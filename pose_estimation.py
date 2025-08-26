import numpy as np
import cv2



def project_pts(K, R, t, pts):
    # pts: Nx3, R: 3x3, t:3x1
    pts_cam = (R @ pts.T + t).T
    pts_img = (K @ pts_cam.T).T
    pts_img = pts_img[:, :2] / pts[:, 2:3]

    return pts_img

# ---------- Core linear algebra helpers ----------
def hat(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=float)

def rodrigues(axis, theta):
    axis = np.asarray(axis, float)
    th = np.linalg.norm(axis) if theta is None else abs(theta)
    if theta is None:
        # interpret 'axis' as twist vector phi
        th = np.linalg.norm(axis)
        if th < 1e-12: return np.eye(3)
        k = axis / th
    else:
        if np.linalg.norm(axis) < 1e-12: return np.eye(3)
        k = axis / np.linalg.norm(axis)
    K = hat(k)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=float)

# ---------- Camera projection ----------
def project(K, R, t, Xw):
    """K: 3x3, R:3x3, t:3, Xw: Nx3 -> pixels Nx2"""
    Xc = (R @ Xw.T + t.reshape(3,1)).T       # Nx3
    uvw = (K @ Xc.T).T                        # Nx3
    return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2] # pixels, depths

# ---------- Essential & Fundamental ----------
def essential_from_Rt(R, t):
    t = t.reshape(3)
    return hat(t) @ R

def fundamental_from_E(E, K1, K2):
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

# ---------- Eight-point algorithm (normalized) ----------
def _normalize_points(x):
    # x: Nx2 pixels -> normalized with similarity transform T
    mean = x.mean(axis=0)
    d = np.sqrt(((x-mean)**2).sum(axis=1)).mean()
    s = np.sqrt(2) / (d + 1e-12)
    T = np.array([[s,0,-s*mean[0]],[0,s,-s*mean[1]],[0,0,1]])
    xh = np.hstack([x, np.ones((x.shape[0],1))])
    xn = (T @ xh.T).T
    return xn[:, :2], T

def eight_point_F(x1, x2):
    # x1,x2: Nx2 pixels (same N>=8)
    x1n, T1 = _normalize_points(x1)
    x2n, T2 = _normalize_points(x2)
    u1,v1 = x1n[:,0], x1n[:,1]
    u2,v2 = x2n[:,0], x2n[:,1]
    A = np.column_stack([u2*u1, u2*v1, u2,
                         v2*u1, v2*v1, v2,
                         u1,    v1,    np.ones_like(u1)])
    _,_,Vt = np.linalg.svd(A)
    Fh = Vt[-1].reshape(3,3)
    # enforce rank-2
    U,S,Vt = np.linalg.svd(Fh)
    S[-1] = 0
    Fh = U @ np.diag(S) @ Vt
    # denormalize
    F = T2.T @ Fh @ T1
    return F / np.linalg.norm(F)

# ---------- Decompose E ----------
def decompose_E(E):
    U,S,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W  @ Vt
    R2 = U @ W.T@ Vt
    t  = U[:,2]
    return [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]

# ---------- Epipolar residual ----------
def epi_residual(F, x1, x2):
    """Sampson or simple algebraic residual; here simple x2^T F x1"""
    x1h = np.hstack([x1, np.ones((x1.shape[0],1))])
    x2h = np.hstack([x2, np.ones((x2.shape[0],1))])
    vals = np.sum(x2h * (F @ x1h.T).T, axis=1)
    return vals


# Intrinsics (both cams)
K = np.array([[800,0,320],[0,800,240],[0,0,1]], float)

# Camera 1 (world frame)
R1 = np.eye(3); t1 = np.zeros(3)

# Camera 2: translate along x by 0.12 m and yaw 4 degrees
baseline = 0.12
axis = np.array([0,0,1])
theta = np.deg2rad(10.0)
R2 = rodrigues(axis, theta)
t2 = np.array([baseline, baseline, 0])

# Generate 3D points in front of cam1
N = 200
Xw = np.column_stack([
    np.random.uniform(-1.0, 1.0, N),   # X
    np.random.uniform(-0.6, 0.6, N),   # Y
    np.random.uniform(2.0, 6.0, N)     # Z
])

# Project to both views
x1, z1 = project(K, R1, t1, Xw)
x2, z2 = project(K, R2, t2, Xw)

print(x1.shape, x2.shape)

# Add pixel noise
noise = lambda n: 0.01*np.random.randn(n,2)
x1n = x1 + noise(N)
x2n = x2 + noise(N)

# Ground-truth E, F
E_gt = essential_from_Rt(R2 @ R1.T, t2 - R2 @ R1.T @ t1)  # equivalently [t]x R
F_gt = fundamental_from_E(E_gt, K, K)

# Estimate F from noisy matches
F_est = eight_point_F(x1n, x2n)

# Residuals
r_gt  = np.abs(epi_residual(F_gt,  x1n, x2n)).mean()
r_est = np.abs(epi_residual(F_est, x1n, x2n)).mean()
print(f"Mean |x2^T F x1|, GT vs Est: {r_gt:.6f} vs {r_est:.6f}")

# Decompose E_est = K^T F_est K and pick cheiral solution
E_est = K.T @ F_est @ np.linalg.inv(K)
cands = decompose_E(E_est)

# Cheirality check: count points with positive depths in both cams
def count_cheiral(R, t):
    # camera 2 pose w.r.t camera 1 world (R,t)
    # depths in cam1 and cam2
    _, zc1 = project(K, np.eye(3), np.zeros(3), Xw)
    _, zc2 = project(K, R, t, Xw)
    return int(np.sum((zc1>0) & (zc2>0)))

best = max(cands, key=lambda Rt: count_cheiral(*Rt))
R_est, t_est = best
print("Chosen solution cheirality count:", count_cheiral(R_est, t_est))
