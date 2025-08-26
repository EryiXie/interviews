import numpy as np

import numpy as np

def _normalize_2d(pts):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c)**2).sum(axis=1)).mean() + 1e-12
    s = np.sqrt(2)/d
    T = np.array([[s, 0, -s*c[0]],
                  [0, s, -s*c[1]],
                  [0, 0, 1]], float)
    ph = np.hstack([pts, np.ones((pts.shape[0],1))])  # (N,3)
    pn = (T @ ph.T).T
    return pn[:, :2], T

def _project(H, pts):  # pts: (N,2) pixels
    ph = np.c_[pts, np.ones(len(pts))]        # (N,3)
    q  = (H @ ph.T).T                          # (N,3)
    w  = q[:, 2] + 1e-12
    return np.c_[q[:, 0]/w, q[:, 1]/w]

def _ste(H, pts0, pts1):  # symmetric transfer error in pixels
    p1 = _project(H, pts0)
    p0 = _project(np.linalg.inv(H), pts1)
    return np.sqrt(((p1 - pts1)**2).sum(1) + ((p0 - pts0)**2).sum(1))

def _build_A(xy, uv):     # on normalized coords
    x,y = xy[:,0], xy[:,1]
    u,v = uv[:,0], uv[:,1]
    N   = len(xy)
    A   = np.zeros((2*N, 9))
    A[0::2] = np.c_[ x, y, np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), -u*x, -u*y, -u]
    A[1::2] = np.c_[np.zeros(N), np.zeros(N), np.zeros(N), -x, -y, -np.ones(N),   v*x,  v*y,  v]
    return A

def _dlt(pts0n, pts1n):   # normalized DLT
    A = _build_A(pts0n, pts1n)
    # reject near-degenerate systems
    if np.linalg.matrix_rank(A) < 8:
        return None
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    Hn = h.reshape(3,3)
    if abs(Hn[2,2]) < 1e-12:
        return None
    return Hn / Hn[2,2]

def _noncollinear4(pts, eps=1e-3):
    # area-based test on normalized coords: any triangle from the 4 has area > eps?
    a = pts[0]; b = pts[1]; c = pts[2]; d = pts[3]
    def tri_area(p,q,r): return 0.5*abs(np.cross(q-p, r-p))
    areas = [tri_area(a,b,c), tri_area(a,b,d), tri_area(a,c,d), tri_area(b,c,d)]
    return max(areas) > eps

def estimate_homography_ransac(pts0, pts1, thresh=4.0, max_it=5000, p_success=0.99):
    assert pts0.shape == pts1.shape and pts0.shape[0] >= 4
    pts0 = pts0.astype(float); pts1 = pts1.astype(float)

    # Normalize each set
    pts0n, T0 = _normalize_2d(pts0)
    pts1n, T1 = _normalize_2d(pts1)

    N = len(pts0)
    best_score = np.inf  # MSAC score (sum of truncated squared errors)
    best_H = None
    best_inliers = None

    # Adaptive iteration budget
    it, max_needed = 0, max_it
    s = 4
    rng = np.random.default_rng()

    while it < min(max_it, max_needed):
        it += 1
        idx = rng.choice(N, size=4, replace=False)
        p0, p1 = pts0n[idx], pts1n[idx]
        if not (_noncollinear4(p0) and _noncollinear4(p1)):
            continue

        Hn = _dlt(p0, p1)
        if Hn is None: 
            continue
        H = np.linalg.inv(T1) @ Hn @ T0  # denormalize

        err = _ste(H, pts0, pts1)
        e2  = err**2
        t2  = (thresh**2)
        msac = np.minimum(e2, t2).sum()
        inliers = err < thresh

        if msac < best_score and inliers.sum() >= 4:
            # LO-refit on current inliers (normalize, refit, denorm)
            p0n_in = (T0 @ np.c_[pts0[inliers], np.ones(inliers.sum())].T).T[:, :2]
            p1n_in = (T1 @ np.c_[pts1[inliers], np.ones(inliers.sum())].T).T[:, :2]
            Hn_ref = _dlt(p0n_in, p1n_in)
            if Hn_ref is None:
                continue
            H_ref = np.linalg.inv(T1) @ Hn_ref @ T0
            err_ref = _ste(H_ref, pts0, pts1)
            e2r = err_ref**2
            msac_ref = np.minimum(e2r, t2).sum()
            if msac_ref < best_score:
                best_score = msac_ref
                best_H = H_ref / (H_ref[2,2] if abs(H_ref[2,2])>1e-12 else 1.0)
                best_inliers = err_ref < thresh

                # update needed iterations based on inlier ratio
                w = best_inliers.mean()
                if w > 0:
                    denom = np.log(1 - w**s)
                    if denom < 0:
                        max_needed = min(max_it, int(np.ceil(np.log(1 - p_success) / denom)))

    if best_H is None:
        # Fallback: DLT on all points (may be garbage if many outliers)
        Hn = _dlt(pts0n, pts1n)
        H  = np.linalg.inv(T1) @ Hn @ T0 if Hn is not None else np.eye(3)
        inliers = _ste(H, pts0, pts1) < thresh
        return H/(H[2,2] if abs(H[2,2])>1e-12 else 1.0), inliers

    return best_H, best_inliers


if __name__ == "__main__":
    import numpy as np

    np.random.seed(7)

    # --- Ground truth homography (rotation + translation + mild projective warp)
    theta = np.deg2rad(12)
    c, s = np.cos(theta), np.sin(theta)
    H_true = np.array([
        [c, -s,  60],
        [s,  c,  40],
        [1e-4, -2e-4, 1.0]
    ], float)

    # --- Generate random points in image0
    N = 80
    pts0 = np.random.uniform([50, 50], [500, 400], size=(N, 2))

    # --- Apply H_true to get pts1
    pts0_h = np.hstack([pts0, np.ones((N, 1))])
    pts1_h = (H_true @ pts0_h.T).T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:3]

    # --- Add Gaussian noise
    pts1 += np.random.normal(0, 1.0, pts1.shape)

    # --- Inject some outliers
    out_idx = np.random.choice(N, 10, replace=False)
    pts1[out_idx] += np.random.uniform(-30, 30, size=(len(out_idx), 2))

    # --- Run your RANSAC homography
    H_est, inliers = estimate_homography_ransac(pts0, pts1, thresh=3.0, max_it=2000)

    # --- Evaluation
    print("True H:\n", np.round(H_true/H_true[2,2], 3))
    print("Estimated H:\n", np.round(H_est/H_est[2,2], 3))
    print("Inliers:", int(inliers.sum()), "/", N)

    # Compute reprojection RMSE on inliers
    pts0_inl = pts0[inliers]
    pts1_inl = pts1[inliers]
    pts0h = np.hstack([pts0_inl, np.ones((len(pts0_inl),1))])
    proj = (H_est @ pts0h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    rmse = np.sqrt(np.mean(np.sum((proj - pts1_inl)**2, axis=1)))
    print("Reprojection RMSE on inliers:", round(float(rmse), 3))