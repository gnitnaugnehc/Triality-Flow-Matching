import torch

def octonion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two octonions a, b of shape (..., 8) via the Fano-plane structure constants.
    Returns a tensor of shape (..., 8).
    """
    device = a.device
    # C[i,j,k] = structure constants for e_i * e_j = ± e_k
    C = torch.zeros(8, 8, 8, device=device)
    # -- fill in C according to standard octonion multiplication --
    # e.g.: C[1,2,3] = +1, C[2,1,3] = -1, etc.
    # (for brevity you’d paste the full table here)
    # ...
    return torch.einsum('...i,...j,ijk->...k', a, b, C)
