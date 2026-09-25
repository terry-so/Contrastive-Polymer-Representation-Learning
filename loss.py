import torch
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    logits = (z @ z.T) / temperature
    diagonal = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z.device,
    )
    logits = logits.masked_fill(
        diagonal,
        torch.finfo(logits.dtype).min,
    )

    targets = (
        torch.arange(2 * batch_size, device=z.device) + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(logits, targets)
