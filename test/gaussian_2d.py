import numpy as np

import numpy as np

def _inv2x2_batch(cov: np.ndarray, jitter_eps: float = 1e-8) -> np.ndarray:
    """
    Stable closed-form inverse for a batch of 2x2 SPD matrices.
    Adds tiny, scale-aware jitter on the diagonal when det is too small.
    """
    # Symmetrize (guards tiny asymmetries)
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    c = cov[:, 1, 0]
    d = cov[:, 1, 1]
    det = a * d - b * c

    # scale-aware jitter: if det is tiny, nudge diagonal
    scale = np.maximum(np.maximum(np.abs(a), np.abs(d)), 1.0)
    eps = (jitter_eps * scale * scale)
    bad = np.abs(det) < eps
    if np.any(bad):
        a = a.copy(); d = d.copy()
        a[bad] += eps[bad]
        d[bad] += eps[bad]
        det = a * d - b * c

    inv = np.empty_like(cov)
    inv[:, 0, 0] =  d / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -c / det
    inv[:, 1, 1] =  a / det
    return inv

def splat_blend_2d(uv: np.ndarray, cov: np.ndarray, colors: np.ndarray, opac: np.ndarray, 
                   px: np.ndarray, py: np.ndarray, max_per_pixel: int = 64,
                   alpha_stop: float = 0.999) -> np.ndarray:
    """
    uv:     (M,2) splat centers in pixels
    cov:    (M,2,2) positive-definite covariances (ellipse in pixel space)
    colors: (M,3) in [0,1]
    opac:   (M,) alpha in [0,1] per splat (pre-multiplied by per-splat visibility)
    px,py:  (P,) pixel coords where to evaluate
    return: (P,3) RGB composited front-to-back using: C += T * (alpha * w) * color; T *= (1 - alpha*w)
            where w = exp(-0.5 * (p-u)^T Σ^{-1} (p-u)).
    Notes:
      - Assumes splats are already sorted front-to-back (near -> far).
      - Per-pixel 3σ culling via Mahalanobis distance (q <= 9).
      - Early exits when accumulated alpha >= alpha_stop.
    """
    uv = np.asarray(uv, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    opac = np.clip(np.asarray(opac, dtype=np.float64), 0.0, 1.0)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    M = uv.shape[0]
    P = px.shape[0]
    out = np.zeros((P, 3), dtype=np.float64)

    # Precompute all inverses once
    inv_cov = _inv2x2_batch(cov)

    three_sigma_sq = 9.0  # 3σ ellipse threshold on Mahalanobis^2

    for j in range(P):
        p = np.array([px[j], py[j]], dtype=np.float64)
        dp = p[None, :] - uv          # (M,2)
        # Mahalanobis^2 for all splats at this pixel (batch-einsum over 2x2)
        q = np.einsum('mi,mij,mj->m', dp, inv_cov, dp)  # (M,)

        # 3σ culling
        mask = q <= three_sigma_sq
        if not np.any(mask):
            continue

        # Respect given (depth) order; cap per-pixel budget
        idx = np.nonzero(mask)[0]
        if idx.size > max_per_pixel:
            idx = idx[:max_per_pixel]

        # Precompute weights for the survivors
        w = np.exp(-0.5 * q[idx])           # (K,)
        a_eff = opac[idx] * w               # effective alpha per survivor

        # Front-to-back compositing with early exit
        T = 1.0
        C = np.zeros(3, dtype=np.float64)
        for k, i in enumerate(idx):
            a = a_eff[k]
            if a <= 0.0:
                continue
            C += T * a * colors[i]
            T *= (1.0 - a)
            if 1.0 - T >= alpha_stop:       # accumulated alpha ~1
                break

        out[j] = C

    return out
