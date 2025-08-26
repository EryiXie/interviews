import numpy as np

# ========== Part 1: Median scale alignment + standard depth metrics ==========

def align_scale_median(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    """
    Median scale alignment used in many monocular depth papers (incl. MiDaS eval variants).
    Returns scalar s* that (robustly) aligns pred to gt via median(gt/pred) over the mask.
    Zeros/negatives in pred or gt are ignored for the ratio.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    m    = np.asarray(mask, dtype=bool)

    # valid where mask is True AND pred>0 AND gt>0 (avoid invalid ratios)
    valid = m & (pred > eps) & (gt > eps)
    if not np.any(valid):
        return 1.0  # fallback: no valid pixels

    ratios = gt[valid] / np.maximum(pred[valid], eps)
    s = float(np.median(ratios))
    return s


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> dict:
    """
    Compute common depth metrics after median scaling:
      - AbsRel = mean( |s*pred - gt| / gt )
      - RMSE   = sqrt( mean( (s*pred - gt)^2 ) )
      - delta thresholds: % pixels where max(s*pred/gt, gt/(s*pred)) < 1.25^k (k=1,2,3)
    Ignores invalid entries where gt<=0 or pred<=0 or mask=False.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    m    = np.asarray(mask, dtype=bool)

    # Align scale
    s = align_scale_median(pred, gt, m, eps=eps)

    sp = s * pred
    valid = m & (gt > eps) & (pred > eps)
    if not np.any(valid):
        return dict(scale=s, AbsRel=np.nan, RMSE=np.nan, delta_1_25=np.nan,
                    delta_1_25_2=np.nan, delta_1_25_3=np.nan, valid_count=0)

    g  = gt[valid]
    p  = sp[valid]

    absrel = float(np.mean(np.abs(p - g) / np.maximum(g, eps)))
    rmse   = float(np.sqrt(np.mean((p - g) ** 2)))

    # delta thresholds
    ratio = np.maximum(p / np.maximum(g, eps), g / np.maximum(p, eps))
    d1   = float(np.mean(ratio < 1.25))
    d2   = float(np.mean(ratio < 1.25 ** 2))
    d3   = float(np.mean(ratio < 1.25 ** 3))

    return dict(scale=s, AbsRel=absrel, RMSE=rmse,
                delta_1_25=d1, delta_1_25_2=d2, delta_1_25_3=d3,
                valid_count=int(valid.sum()))


# ========== Part 2: MiDaS-style SI-Log loss + optional gradient L1 ==========

def silog_loss(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, lam: float = 0.85, eps: float = 1e-8) -> float:
    """
    Scale-Invariant Log RMSE (SI-Log):
      D = log(pred) - log(gt) on the mask (guard with eps)
      loss = sqrt( mean(D^2) - lam * (mean(D))^2 )
    Typical lam ~ 0.85 (as used in MiDaS variants).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    m    = np.asarray(mask, dtype=bool)

    valid = m & (pred > 0) & (gt > 0)
    if not np.any(valid):
        return float('nan')

    d = np.log(np.maximum(pred[valid], eps)) - np.log(np.maximum(gt[valid], eps))
    mu2 = np.mean(d ** 2)
    mu  = np.mean(d)
    val = mu2 - lam * (mu ** 2)
    return float(np.sqrt(max(val, 0.0)))


def grad_l1_loss(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """
    L1 on image gradients (finite differences), masked.
    Uses forward differences in x and y; only pixels where both neighbors are valid contribute.
    """
    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt,   dtype=np.float64)
    M = np.asarray(mask, dtype=bool)

    assert P.shape == G.shape == M.shape, "pred, gt, mask must have same HxW shape"
    H, W = P.shape[-2], P.shape[-1] if P.ndim == 2 else P.shape  # handle HxW only

    # Forward differences: dx = I[:,1:] - I[:,:-1], dy = I[1:,:] - I[:-1,:]
    # Valid where both pixels are masked
    Mx = M[:, 1:] & M[:, :-1] if P.ndim == 2 else (M[1:] & M[:-1])  # We'll implement 2D case below
    My = M[1:, :] & M[:-1, :]

    # 2D-only implementation
    if P.ndim != 2:
        raise ValueError("grad_l1_loss expects 2D arrays (H, W).")

    dxP = P[:, 1:] - P[:, :-1]
    dxG = G[:, 1:] - G[:, :-1]
    vx  = M[:, 1:] & M[:, :-1]
    lx  = np.abs(dxP[vx] - dxG[vx])

    dyP = P[1:, :] - P[:-1, :]
    dyG = G[1:, :] - G[:-1, :]
    vy  = M[1:, :] & M[:-1, :]
    ly  = np.abs(dyP[vy] - dyG[vy])

    count = lx.size + ly.size
    if count == 0:
        return float('nan')
    return float((lx.sum() + ly.sum()) / count)


# ========== Tiny smoke tests ==========

if __name__ == "__main__":
    np.random.seed(0)

    # --- Depth metrics tests ---
    gt = np.array([[1.0, 2.0, 4.0],
                   [8.0, 0.0, 3.0]])
    pred = np.array([[0.6, 1.9, 3.8],
                     [16.0, 1.0, 3.2]])
    mask = np.array([[1, 1, 1],
                     [1, 0, 1]], dtype=bool)

    s = align_scale_median(pred, gt, mask)
    m = depth_metrics(pred, gt, mask)
    print("Median scale s*:", round(s, 4))
    print("Depth metrics:", {k: (round(v,4) if isinstance(v, float) else v) for k, v in m.items()})

    # --- SI-Log and gradient loss tests ---
    gt2 = np.array([[1.0, 2.0, 4.0, 8.0],
                    [1.2, 2.2, 3.8, 8.5]])
    pr2 = gt2 * 0.9  # scaled version → small SI-Log
    ms2 = np.ones_like(gt2, dtype=bool)

    sil = silog_loss(pr2, gt2, ms2, lam=0.85)
    print("SI-Log loss (scaled pred):", round(sil, 6))

    # gradient loss: small differences
    pr3 = gt2.copy(); pr3[:, 2:] += 0.1
    gl = grad_l1_loss(pr3, gt2, ms2)
    print("Grad L1 loss:", round(gl, 6))
