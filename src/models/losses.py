#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 21-Mar-2023
#
# References:
# 1. solo-learn: https://github.com/vturrisi/solo-learn/tree/main/solo/losses
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
    
class SimCLRLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.1):
        super(SimCLRLoss, self).__init__()

        self.temperature = temperature

    def forward(self, z: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
        """Computes SimCLR's loss given batch of projected features z
        from different views, a positive boolean mask of all positives and
        a negative boolean mask of all negatives.
        Args:
            z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
            indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
                or targets of each crop (supervised).
        Return:
            torch.Tensor: SimCLR loss.
        """

        z = F.normalize(z, dim=-1) # 128, 512

        sim = torch.exp(torch.einsum("if, jf -> ij", z, z) / self.temperature)

        indexes = indexes.unsqueeze(0)

        # positives
        pos_mask = indexes.t() == indexes
        pos_mask.fill_diagonal_(0)

        # negatives
        neg_mask = indexes.t() != indexes

        pos = torch.sum(sim * pos_mask, 1)
        neg = torch.sum(sim * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss
    
class ProtoConLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.1):
        super(ProtoConLoss, self).__init__()

        self.temperature = temperature

    def forward(self, preds: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:

        preds = torch.exp(preds) / self.temperature # 128, <num_protos>

        pos_mask = F.one_hot(indices, num_classes = 10) # 128, <num_protos>
        neg_mask = torch.ones_like(pos_mask) - pos_mask

        pos = torch.sum(preds * pos_mask, 1)
        neg = torch.sum(preds * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss
    
class ProtoUniLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.1):
        super(ProtoUniLoss, self).__init__()

        self.temperature = temperature

    def forward(self, protos: torch.Tensor) -> torch.Tensor:

        sim = torch.exp(torch.einsum("if, jf -> ij", protos, protos) / self.temperature)
        mask = torch.ones_like(sim).fill_diagonal_(0)

        return torch.sum(sim * mask, 1).mean()