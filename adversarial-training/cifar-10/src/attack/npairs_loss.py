import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss

def npairs_loss(X, X_adv, label):
    """
    Args:
        Calculates the npair_loss for this batch using efficient batch construction

        X: Embeddings from Input images; dimension: batch_size * Width * Height
        X_adv: Embeddings from adversarially perturbed input images;  dimension: batch_size * Width * Height
        label: Real output labels for images; dimension: batch_size
    Returns:
        The npairs loss for this batch
    """
    npairloss = NpairLoss()
    loss_npair = npairloss(
        anchor=X_adv,
        positive=X,
        target=label
    )
    print("Npairs loss:", loss_npair)
    return loss_npair
