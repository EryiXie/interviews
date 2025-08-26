import numpy as np

def _normalize_bearings(x, K):
    """Pixels (N,2) -> normalized homogeneous bearings (N,3)."""
    N = x.shape[0]
    xh = np.hstack([x, np.ones((N,1))])          # (N,3)
    xb = (np.linalg.inv(K) @ xh.T).T             # (N,3)
    return xb

def _dlt_p_from_normalized(Xw, xb):
    """
    Linear DLT to estimate camera matrix P ≈ [R|t] from
    world points Xw (N,3) and normalized bearings xb=(u,v,1) (N,3).
    Returns P (3x4).
    """
    N = Xw.shape[0]
    Xh = np.hstack([Xw, np.ones((N,1))])         # (N,4)
    A = []
    for i in range(N):
        u, v, w = xb[i]   # w≈1
        Xhi = Xh[i]
        # Enforce x × (P X) = 0 with two independent rows (drop the third)
        A.append(np.hstack([ np.zeros(4),      -w*Xhi,     v*Xhi ]))
        A.append(np.hstack([      w*Xhi,   np.zeros(4),    -u*Xhi ]))
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3,4)
    return P

def _Rt_from_P(P):
    """
    Given P ≈ [R|t] up to a global scale, recover a valid rotation via polar
    decomposition and de-scale t using s = trace(R^T M)/3 (M = P[:,:3]).
    """
    M = P[:, :3]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R *= -1
    # If M ≈ s R, then s ≈ trace(R^T M)/3
    s = np.trace(R.T @ M) / 3.0
    if abs(s) < 1e-12:
        s = 1.0
    t = P[:, 3] / s
    return R, t

def _reproj_error_px(R, t, Xw, x, K):
    """Per-point pixel reprojection error (N,)."""
    Xc = (R @ Xw.T + t.reshape(3,1)).T
    z = Xc[:, 2:3]
    uv = (Xc[:, :2] / (z + 1e-12))
    uv = (K[:2, :2] @ uv.T).T + K[:2, 2]
    err = np.linalg.norm(uv - x, axis=1)
    return err

def ransac_pnp(Xw: np.ndarray, x: np.ndarray, K: np.ndarray,
               iters: int = 800, thresh: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustly estimate pose from 2D-3D correspondences.
    Xw: (N,3) world points
    x:  (N,2) pixel coords
    K:  (3,3) intrinsics
    Returns: R (3,3), t (3,), inliers (N,bool)
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    x  = np.asarray(x,  dtype=np.float64)
    N = Xw.shape[0]
    if N < 6:
        raise ValueError("Need at least 6 correspondences for linear PnP in RANSAC.")

    xb = _normalize_bearings(x, K)
    rng = np.random.default_rng(0)

    best_inl = None
    best_cnt = -1
    best_Rt  = None

    for _ in range(iters):
        idx = rng.choice(N, size=6, replace=False)
        try:
            P = _dlt_p_from_normalized(Xw[idx], xb[idx])
            R, t = _Rt_from_P(P)
        except np.linalg.LinAlgError:
            continue

        err = _reproj_error_px(R, t, Xw, x, K)
        inl = err < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
            best_Rt  = (R, t)

    if best_inl is None or best_cnt < 6:
        raise RuntimeError("RANSAC failed to find a valid pose.")

    # Refit on inliers
    P = _dlt_p_from_normalized(Xw[best_inl], xb[best_inl])
    R, t = _Rt_from_P(P)

    # Final inliers with refined pose
    err = _reproj_error_px(R, t, Xw, x, K)
    inl = err < thresh

    # Ensure a proper rotation and reasonable t
    if np.linalg.det(R) < 0:
        R *= -1; t *= -1

    return R, t, inl

if __name__ == "__main__":
    np.random.seed(1)
    K = np.array([[700, 0, 320],
                  [  0,700, 240],
                  [  0,  0,   1]], float)
    # Ground truth pose
    R_gt = np.eye(3)
    t_gt = np.array([0.2, -0.1, 0.4])
    # Scene points
    N = 80
    Xw = np.random.uniform([-1,-1,3],[1,1,7], size=(N,3))
    # Project
    Xc = (R_gt @ Xw.T + t_gt.reshape(3,1)).T
    x  = (K[:2,:2] @ (Xc[:,:2]/Xc[:,2:3]).T).T + K[:2,2]
    # Add noise and outliers
    x += np.random.normal(0, 1.0, x.shape)
    out_idx = np.random.choice(N, 10, replace=False)
    x[out_idx] += np.random.uniform(-50, 50, size=(len(out_idx), 2))

    R, t, inl = ransac_pnp(Xw, x, K, iters=600, thresh=4.0)
    rmse = np.sqrt(np.mean(_reproj_error_px(R, t, Xw[inl], x[inl], K)**2))
    print("Inliers:", int(inl.sum()), "/", N)
    print("det(R):", round(float(np.linalg.det(R)), 6))
    print("RMSE (px) on inliers:", round(float(rmse), 3))
