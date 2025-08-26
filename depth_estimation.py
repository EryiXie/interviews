import torch
import torch.nn.functional as F

def _apply_mask(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor | None):
    """
    pred, tgt: [B, H, W] or [B, 1, H, W]
    mask:     same shape (bool or float), 1/True = valid
    Returns pred, tgt, mask as [B, N] flattened with valid entries only.
    """
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
        tgt  = tgt[:, 0]
        if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]

    B = pred.size(0)
    pred = pred.reshape(B, -1)
    tgt  = tgt.reshape(B, -1)

    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    else:
        mask = (mask > 0).reshape(B, -1)

    return pred, tgt, mask


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    lam: float = 1.0,
    reduction: str = "mean",
    eps: float = 1e-6,
):
    """
    Scale-invariant log MSE (Eigen et al.).
    L = mean( (log(pred) - log(gt))^2 ) - lam * (mean( log(pred)-log(gt) ))^2

    Args:
        pred, target: depth maps, shape [B, H, W] or [B, 1, H, W], values > 0
        mask: optional valid mask (bool/float), same shape
        lam:  weight for the bias (second) term; lam=1.0 is the classic form
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        loss (tensor), shape [] if reduced else [B]
    """
    pred, target, mask = _apply_mask(pred, target, mask)
    # ensure positive for log
    pred  = torch.clamp(pred,  min=eps)
    target= torch.clamp(target,min=eps)

    diff = (pred.log() - target.log())  # [B, N]
    # mask invalid
    diff = diff * mask

    n = mask.sum(dim=1).clamp_min(1)  # [B]
    mean_sq = (diff.pow(2).sum(dim=1) / n)               # E[d^2]
    mean    = (diff.sum(dim=1) / n)                      # E[d]
    loss_b  = mean_sq - lam * mean.pow(2)                # per-sample

    if reduction == "mean":
        return loss_b.mean()
    elif reduction == "sum":
        return loss_b.sum()
    else:
        return loss_b


def scale_and_shift_invariant_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-6,
):
    """
    Scale-and-shift invariant L2 (DPT-style):
    Finds per-image alpha, beta minimizing || alpha*pred + beta - target ||^2 over valid pixels,
    then returns the mean squared residual under that optimal (alpha, beta).

    Closed-form:
      Let p=pred, t=target, m=mask in {0,1}. Define:
        S_pp = sum(m * p^2),  S_p = sum(m * p),  S_t = sum(m * t),  S_pt = sum(m * p*t),  N = sum(m)
      Den = S_pp * N - S_p^2
      alpha = (N*S_pt - S_p*S_t) / Den
      beta  = (S_pp*S_t - S_p*S_pt) / Den
    If Den ~ 0, we fall back to alpha=1, beta=0.

    Args:
        pred, target: [B,H,W] or [B,1,H,W]
        mask: optional valid mask
        reduction: 'mean' | 'sum' | 'none'
    """
    pred, target, mask_b = _apply_mask(pred, target, mask)
    m = mask_b.to(pred.dtype)

    # sums per batch
    N   = m.sum(dim=1).clamp_min(1.0)
    p   = pred * m
    t   = target * m
    S_pp= (p * pred).sum(dim=1)         # sum(m * p^2)
    S_p = p.sum(dim=1)                   # sum(m * p)
    S_t = t.sum(dim=1)                   # sum(m * t)
    S_pt= (p * target).sum(dim=1)        # sum(m * p*t)

    Den = S_pp * N - S_p * S_p

    # safe fallback where Den≈0
    safe = Den.abs() > eps
    alpha = torch.where(
        safe, (N * S_pt - S_p * S_t) / Den, torch.ones_like(Den)
    )
    beta = torch.where(
        safe, (S_pp * S_t - S_p * S_pt) / Den, torch.zeros_like(Den)
    )

    # residuals under optimal alpha,beta
    # broadcast alpha,beta to [B, N]
    alpha_b = alpha.unsqueeze(1)
    beta_b  = beta.unsqueeze(1)
    residual = (alpha_b * pred + beta_b - target) * m
    per_sample = (residual.pow(2).sum(dim=1) / N)

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    else:
        return per_sample
