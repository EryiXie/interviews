import numpy as np

def nerf_volume_render(sigmas: np.ndarray, rgbs: np.ndarray, deltas: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Vectorized NeRF volume rendering along one ray.

    Args:
        sigmas: (S,) densities σ_i  (>=0)
        rgbs:   (S,3) colors in [0,1]
        deltas: (S,) step sizes Δ_i (>=0)
    Returns:
        alpha: float  in [0,1], final opacity  sum_i w_i
        rgb:   (3,)   final color              sum_i w_i * c_i
    """
    # --- shape & dtype checks ---
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
    rgbs   = np.asarray(rgbs,   dtype=np.float64)
    S = sigmas.shape[0]
    assert deltas.shape == (S,), f"deltas must be (S,), got {deltas.shape}"
    assert rgbs.shape   == (S,3), f"rgbs must be (S,3), got {rgbs.shape}"

    # Guard against tiny negative inputs from noise
    tau = np.maximum(sigmas * deltas, 0.0)            # τ_i = σ_i Δ_i  (S,)

    # Alpha_i = 1 - exp(-τ_i) (stable; exp(-large) underflows to 0 which is fine)
    alpha_i = 1.0 - np.exp(-tau)                      # (S,)

    # Transmittance T_i = exp(-sum_{j<i} τ_j) via exclusive cumsum (no Python loops)
    cumsum_tau = np.cumsum(tau)                       # (S,)
    exclusive  = np.concatenate(([0.0], cumsum_tau[:-1]))
    T_i = np.exp(-exclusive)                          # (S,)

    # Weights
    w = T_i * alpha_i                                 # (S,)

    # Final outputs
    alpha = float(np.clip(np.sum(w), 0.0, 1.0))       # clip for tiny FP overshoot
    rgb   = (w[:, None] * rgbs).sum(axis=0)           # (3,)
    return alpha, rgb


def _ref_nerf_volume_render(sigmas, rgbs, deltas):
    # Slow but clear reference for verification
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
    rgbs   = np.asarray(rgbs,   dtype=np.float64)
    T = 1.0
    rgb = np.zeros(3, dtype=np.float64)
    alpha = 0.0
    for s, c, d in zip(sigmas, rgbs, deltas):
        tau = max(s*d, 0.0)
        a = 1.0 - np.exp(-tau)
        w = T * a
        rgb += w * c
        alpha += w
        T *= np.exp(-tau)
    return float(alpha), rgb

def _allclose(a, b, tol=1e-6):
    if isinstance(a, tuple):
        return _allclose(a[0], b[0], tol) and _allclose(a[1], b[1], tol)
    if np.isscalar(a):
        return abs(a - b) <= tol
    return np.allclose(a, b, atol=tol, rtol=0)

if __name__ == "__main__":
    # --- Case 1: simple hand-checkable values ---
    sig = np.array([0.1, 1.2, 0.4, 2.0])
    rgb = np.array([[1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1.],
                    [1.,1.,1.]], float)
    dlt = np.array([0.2, 0.2, 0.2, 0.2])
    a1, c1 = nerf_volume_render(sig, rgb, dlt)
    a1_ref, c1_ref = _ref_nerf_volume_render(sig, rgb, dlt)
    print("Case1 alpha:", a1, "rgb:", np.round(c1,6))
    print("Match ref:", _allclose((a1,c1), (a1_ref,c1_ref)))

    # --- Case 2: random moderate magnitudes ---
    rng = np.random.default_rng(0)
    S = 64
    sig = rng.uniform(0.0, 3.0, size=S)
    rgb = rng.uniform(0.0, 1.0, size=(S,3))
    dlt = rng.uniform(0.01, 0.2, size=S)
    a2, c2 = nerf_volume_render(sig, rgb, dlt)
    a2_ref, c2_ref = _ref_nerf_volume_render(sig, rgb, dlt)
    print("Case2 match:", _allclose((a2,c2), (a2_ref,c2_ref)))

    # --- Case 3: extreme τ to test stability (saturating opacity) ---
    sig = np.array([0.0, 50.0, 100.0, 200.0])  # very high densities
    rgb = np.array([[0.2,0.3,0.4],
                    [0.8,0.1,0.0],
                    [0.0,0.9,0.1],
                    [1.0,1.0,1.0]], float)
    dlt = np.array([1.0, 1.0, 1.0, 1.0])       # τ = [0,50,100,200]
    a3, c3 = nerf_volume_render(sig, rgb, dlt)
    a3_ref, c3_ref = _ref_nerf_volume_render(sig, rgb, dlt)
    print("Case3 alpha (should ~1):", a3)
    print("Case3 match:", _allclose((a3,c3), (a3_ref,c3_ref)))