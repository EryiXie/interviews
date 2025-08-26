import numpy as np
import torch

def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray|None=None) -> np.ndarray:
    """
    Q,K,V: (B, T, D). mask: (B, T, T) with -inf where blocked, 0 elsewhere (additive).
    Return: (B, T, D)
    """
    B, T, D = Q.shape

    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)

    scale = np.sqrt(D)
    A = np.matmul(Q, K.transpose(0, 2, 1)) / scale # B, T, T

    if mask is not None:
        A = A + mask

    A_max = np.max(A, axis=1, keepdims=True)
    logits = A - A_max
    np.exp(logits, out=logits)
    denom = np.sum(logits, axis=-1, keepdims=True) + 1e-12
    attn = logits / denom

    out = np.matmul(attn, V)
    return out, attn

if __name__ == "__main__":
    # 1) Symmetry check (no mask): with Q=K and simple V, attention should mix rows
    Q = np.array([[[1., 0.],
                [0., 1.]]])
    K = Q.copy()
    V = np.array([[[1., 2.],
                [3., 4.]]])
    Y = sdpa(Q, K, V, None)
    print("Y:", np.round(Y, 3))  # rows should be softmax mixtures of V

    # 2) Masking test: block attending to the second token
    mask = np.array([[[0., -1e9],
                    [0., -1e9]]])  # very negative ~ -inf to block column 1
    Y_masked = sdpa(Q, K, V, mask)
    print("Y_masked:", np.round(Y_masked, 3))  # should heavily favor the first column/value

    # 3) Big-magnitude stability test
    rng = np.random.default_rng(0)
    Qb = rng.normal(0, 100.0, size=(2, 3, 64))
    Kb = rng.normal(0, 100.0, size=(2, 3, 64))
    Vb = rng.normal(0, 1.0,  size=(2, 3, 64))
    Yb, attn = sdpa(Qb, Kb, Vb, None)
    print("Stable output shape:", Yb.shape)     # (2,3,64) without NaNs/Infs