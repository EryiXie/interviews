import numpy as np

def estimate_pose_from_matches(x0: np.ndarray, x1: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x0, x1: (N,2) pixel coordinates
    K: 3x3 intrinsics
    returns: R (3x3), t (3,) with ||t|| = 1
    """
    assert x0.shape == x1.shape and x0.shape[1] == 2
    N = x0.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 correspondences for the 8-point algorithm.")

    # Normalize to camera coordinate via K^-1
    Kinv = np.linalg.inv(K)
    x0h = np.hstack([x0, np.ones(N,1)])
    x1h = np.hstack([x1, np.ones(N,1)])
    x0b = (Kinv @ x0h.T).T
    x1b = (Kinv @ x1h.T).T

    # 8 point alg for E (x1^T E x0 = 0)
    u0, v0 = x0b[:,0], x0b[:, 1]
    u1, v1 = x1b[:,0], x1b[:, 1]
    A = np.stack([
        u0*u1, u0*v1, u1, 
        v0*u1, v0*v1, v1,
        u0, v0, np.ones(N) 
    ], axis=1) # AE = 0
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3,3)

    # Enforce rank-2
    U, S, Vt = np.linalg.svd(E)
    s = (S[0]+ S[1])*0.5
    E = U @ np.diag([s, s, 0.0]) @ Vt

    # Decompose E into R,t
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    W = np.array([
        [0, -1, 0], [1, 0, 0], [0, 0, 1]
    ])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0: R1 *= -1
    if np.linalg.det(R2) < 0: R2 *= -1
    t = U[:,3]
    t = t / (np.linalg.norm(t) + 1e-12)
    candidates = [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]



    return R_best, t_best


# --------- helpers ---------
def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], float)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def _project(K, R, t, X):
    Xc = (R @ X.T + t.reshape(3,1)).T
    x = Xc[:, :2] / Xc[:, 2:3]
    x = (K[:2, :2] @ x.T).T + K[:2, 2]
    return x

def _triangulate_linear(P0, P1, x0b, x1b):
    # x*b are normalized homogeneous (N,3) with last=1
    N = x0b.shape[0]
    X = np.zeros((N,3))
    for i in range(N):
        x, y = x0b[i,0], x0b[i,1]
        xp, yp = x1b[i,0], x1b[i,1]
        A = np.vstack([
            x*P0[2]-P0[0],
            y*P0[2]-P0[1],
            xp*P1[2]-P1[0],
            yp*P1[2]-P1[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        Xi = Vt[-1]
        X[i] = Xi[:3] / (Xi[3] + 1e-12)
    return X

def _cheirality_fraction(R, t, x0, x1, K):
    Kinv = np.linalg.inv(K)
    N = x0.shape[0]
    x0b = (Kinv @ np.hstack([x0, np.ones((N,1))]).T).T
    x1b = (Kinv @ np.hstack([x1, np.ones((N,1))]).T).T
    P0 = np.hstack([np.eye(3), np.zeros((3,1))])
    P1 = np.hstack([R, t.reshape(3,1)])
    X = _triangulate_linear(P0, P1, x0b, x1b)
    z0 = X[:,2]
    X1 = (R @ X.T + t.reshape(3,1)).T
    z1 = X1[:,2]
    good = (z0 > 0) & (z1 > 0)
    return float(np.mean(good))

def _angle_deg(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

# --------- main synthetic tests ---------
if __name__ == "__main__":
    np.random.seed(123)

    # Camera intrinsics
    K = np.array([[700,  5, 320],   # include a tiny skew to ensure generality
                  [  0,700, 240],
                  [  0,  0,   1]], float)

    # Ground-truth relative pose (cam1 w.r.t cam0)
    ang_deg = np.array([3.0, -5.0, 4.0])      # small rotation
    R_true  = _exp_so3(np.deg2rad(ang_deg))
    t_true  = np.array([0.12, -0.02, 0.10])   # arbitrary baseline
    t_true  = t_true / np.linalg.norm(t_true) # direction only (E-pose is up to scale)

    # 3D points in front of cam0
    N = 120
    X = np.random.uniform(low=[-1.0, -0.8, 3.0],
                          high=[ 1.0,  0.8, 7.0], size=(N,3))

    # Perfect projections
    x0 = _project(K, np.eye(3), np.zeros(3), X)
    x1 = _project(K, R_true, t_true, X)

    # Add pixel noise
    noise_px = 0.8
    x0_noisy = x0 + np.random.normal(0, noise_px, x0.shape)
    x1_noisy = x1 + np.random.normal(0, noise_px, x1.shape)

    # Optional: a few gross outliers (comment out if your implementation has no RANSAC)
    # out_idx = np.random.choice(N, size=8, replace=False)
    # x1_noisy[out_idx] += np.random.uniform(-40, 40, size=(len(out_idx), 2))

    # ---- call your implementation ----
    # from your_module import estimate_pose_from_matches
    R_est, t_est = estimate_pose_from_matches(x0_noisy, x1_noisy, K)

    # ---- evaluation ----
    # 1) det(R) ~ +1
    detR = np.linalg.det(R_est)

    # 2) translation direction angle (up to sign ambiguity is resolved by cheirality in a correct impl)
    ang_t = _angle_deg(t_est, t_true)

    # 3) cheirality fraction with estimated pose
    pos_frac = _cheirality_fraction(R_est, t_est, x0_noisy, x1_noisy, K)

    # Print results
    print("det(R_est):", round(float(detR), 6))
    print("angle(t_est, t_true) [deg]:", round(ang_t, 3))
    print("Positive-depth fraction:", round(pos_frac, 3))

    # Simple pass/fail per the spec
    det_ok = (abs(detR - 1.0) < 1e-2)
    angle_ok = (ang_t < 15.0)
    cheir_ok = (pos_frac >= 0.80)

    print("\nPASS det(R)=+1:", det_ok)
    print("PASS angle(t) < 15°:", angle_ok)
    print("PASS ≥80% positive depth:", cheir_ok)

    # --- Bonus: a slightly harder second case (more noise, wider FOV) ---
    np.random.seed(456)
    K2 = np.array([[500, 0, 320],
                   [  0,500, 240],
                   [  0,  0,   1]], float)
    R2 = _exp_so3(np.deg2rad([6.0, -4.0, 8.0]))
    t2 = np.array([0.2, 0.05, 0.12]); t2 /= np.linalg.norm(t2)
    X2 = np.random.uniform([-1.5,-1.0,2.5],[1.5,1.0,6.5], size=(150,3))
    x0b = _project(K2, np.eye(3), np.zeros(3), X2)
    x1b = _project(K2, R2, t2, X2)
    x0b += np.random.normal(0, 1.2, x0b.shape)
    x1b += np.random.normal(0, 1.2, x1b.shape)

    R_est2, t_est2 = estimate_pose_from_matches(x0b, x1b, K2)
    detR2 = np.linalg.det(R_est2)
    ang_t2 = _angle_deg(t_est2, t2)
    pos_frac2 = _cheirality_fraction(R_est2, t_est2, x0b, x1b, K2)

    print("\n[Harder case]")
    print("det(R_est):", round(float(detR2), 6))
    print("angle(t_est, t_true) [deg]:", round(ang_t2, 3))
    print("Positive-depth fraction:", round(pos_frac2, 3))
    print("PASS:", (abs(detR2-1.0)<1e-2) and (ang_t2<15.0) and (pos_frac2>=0.80))