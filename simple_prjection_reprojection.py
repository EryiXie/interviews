import numpy as np

# --- Camera (example numbers) ---
K = np.array([[800.,   0., 320.],
              [  0., 800., 240.],
              [  0.,   0.,   1.]])

R = np.array([[ 0.998,  0.000,  0.063],
              [ 0.000,  1.000,  0.000],
              [-0.063,  0.000,  0.998]])   # world->camera rotation
t = np.array([0.1, 0.0, 0.2])              # world->camera translation (in meters)

# --- 1) Project a world point Xw -> pixel (u,v) ---
def project_point(Xw, K, R, t):
    Xc = R @ Xw + t                      # to camera coords
    x  = K @ Xc                          # pinhole (homog)
    u, v = x[0]/x[2], x[1]/x[2]
    return np.array([u, v]), Xc[2]       # return pixel and depth Zc

# --- 2a) Back-project pixel (u,v) with known depth Zc -> 3D world point ---
def backproject_with_depth(u, v, Zc, K, R, t):
    # camera ray (normalized by K^{-1})
    Kinv = np.linalg.inv(K)
    dir_cam = Kinv @ np.array([u, v, 1.0])   # direction in cam frame at unit depth
    Xc = dir_cam * Zc                         # scale by desired depth
    Xw = R.T @ (Xc - t)                       # camera->world
    return Xw

# --- 2b) Back-project pixel (u,v) by intersecting with a world plane n^T X + d = 0 ---
def backproject_to_plane(u, v, K, R, t, n, d):
    Kinv = np.linalg.inv(K)
    # Ray in world: Xw(s) = C + s * D
    C = -R.T @ t                              # camera center in world
    D_cam = Kinv @ np.array([u, v, 1.0])      # direction in cam coords
    D = R.T @ D_cam                           # direction in world

    denom = n @ D
    if abs(denom) < 1e-12:
        return None  # ray parallel to plane
    s = -(n @ C + d) / denom
    Xw = C + s * D
    return Xw

# ---------- Demo ----------
Xw_true = np.array([0.5, 0.1, 2.0])          # some world point (meters)
uv, Zc = project_point(Xw_true, K, R, t)
print("Projected pixel:", uv, " depth Zc:", Zc)

# Reproject with known depth:
Xw_rec_depth = backproject_with_depth(uv[0], uv[1], Zc, K, R, t)
print("Back-projected (known depth):", Xw_rec_depth)

# Reproject by intersecting with plane, e.g., ground plane y=0 -> n=[0,1,0], d=0
n = np.array([0.0, 1.0, 0.0]); d = 0.0
Xw_rec_plane = backproject_to_plane(uv[0], uv[1], K, R, t, n, d)
print("Back-projected (to plane y=0):", Xw_rec_plane)