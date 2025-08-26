import numpy as np

def undistort_and_project(xy_norm: np.ndarray, k1: float, k2: float, K: np.ndarray) -> np.ndarray:
    """
    xy_norm: (N,2) normalized coords (x/z, y/z) in camera frame
    k1,k2: radial distortion
    K: 3x3 intrinsics
    return: (N,2) pixel coords
    """
    x = xy_norm[:,0]
    y = xy_norm[:,1]
    r2 = x*x + y*y
    x_d = x * (1 + k1*r2 + k2 * np.power(r2, 2))
    y_d = y * (1 + k1*r2 + k2 * np.power(r2, 2))

    Xh = np.vstack(x_d, y_d, np.ones_like(x_d)) 
    Uh = (K @ Xh).T
    u = Uh[:, 0] / Uh[:, 2]
    v = Uh[:, 1] / Uh[:, 2]
    return np.stack([u, v], axis=1)

