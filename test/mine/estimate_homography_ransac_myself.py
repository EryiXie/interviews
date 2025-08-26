import numpy as np

def estimate_homography_ransac(pts0: np.ndarray, pts1: np.ndarray, thresh: float=3.0, max_it: int=2000) -> tuple[np.ndarray, np.ndarray]:
    """
    pts0, pts1: (N,2) pixels
    return: (H 3x3 with H[2,2]=1), inlier_mask (N,bool)
    """
    
    # 2D pts normalization, then convert to homogenes
    pts0_, T0 = normalize_2d(pts0)
    pts1_, T1 = normalize_2d(pts1)

    #pts0_h = np.stack([pts0_, np.ones(pts0.shape[0], 1)], axis=1)
    #pts1_h = np.stack([pts1_, np.ones(pts1.shape[0], 1)], axis=1)
    
    best_indices = []
    best_cnt = -1
    rng = np.random.default_rng(42)
    # RANSAC looping
    for i in range(0, max_it):
        #random sample 4 points from pts0 and pts1
        indices = rng.choice(N, size=4, replace=False)

        # Homograph slover
        H_curr = homography_solver(pts0_[indices], pts1_[indices])

        # Compute reprojection error with threshold
        err = sym_transfer_error(H_curr, pts0_, pts1_)

        inliers = err < (thresh / np.sqrt(2))
        cnt = int(inliers.sum())
        # Keep the best and run next iteration until max_it
        if cnt > best_cnt:
            best_cnt = cnt
            best_indices = indices
            best_Hn = H_curr
            print(i, best_cnt, best_indices, best_Hn)
    
    if len(best_indices) == 0:
        raise RuntimeError("Failed to find homography")

    Hn_refit = best_Hn
    
    H = np.linalg.inv(T1) @ Hn_refit @ T0
    H = H / (H[2,2] + 1e-12)
    print(H, H_curr)


    # Final inliers using **pixel**-space symmetric error
    err_pix = sym_transfer_error(H, pts0, pts1)
    inliers = err_pix < thresh

    return H, inliers


def normalize_2d(pts):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c)**2).sum(axis=1)).mean() + 1e-12
    s = np.sqrt(2)/d
    T = np.array([[s, 0, -s*c[0]],
                  [0, s, -s*c[1]],
                  [0, 0, 1]], float)
    ph = np.hstack([pts, np.ones((pts.shape[0],1))])  # (N,3)
    pn = (T @ ph.T).T
    return pn[:, :2], T

def homography_solver(p0, p1):
    # AH = 0 
    M = p0.shape[0]
    A = np.zeros((2*M, 9), np.float32)
    # Assemble A matrix
    for i in range(M):
        x, y = p0[i]
        u, v = p1[i]
        A[2*i+1] = [ x, y, 1,  0, 0, 0, -u*x,-u*y,-u]
        A[2*i] = [0,0,0, -x,-y,-1,  v*x, v*y, v]
    # Solve AH = 0 with SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)

    return H / (H[2,2] + 1e-12)

def sym_transfer_error(H, p0, p1):
    # p0,p1: (N,2) (unnormalized or normalized consistently with H)
    N = p0.shape[0]
    p0h = np.hstack([p0, np.ones((N,1))])
    p1h = np.hstack([p1, np.ones((N,1))])

    Hp0 = (H @ p0h.T).T
    Hp0 = Hp0[:, :2] / (Hp0[:, 2:3] + 1e-12)

    Hinv = np.linalg.inv(H)
    Hinvp1 = (Hinv @ p1h.T).T
    Hinvp1 = Hinvp1[:, :2] / (Hinvp1[:, 2:3] + 1e-12)
    
    ef = np.linalg.norm(p1 - Hp0, axis=1)
    eb = np.linalg.norm(p0 - Hinvp1, axis=1)

    return ef + eb # (N,)

if __name__ == "__main__":

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
    H_est, inliers = estimate_homography_ransac(pts0, pts1, thresh=3.0, max_it=5000)

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