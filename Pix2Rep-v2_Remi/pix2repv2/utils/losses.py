import torch
from einops import rearrange


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_terms(features_1, features_2, device):
    features_1 = rearrange(features_1, "b c h w -> (b h w) c")
    features_2 = rearrange(features_2, "b c h w -> (b h w) c")

    bn = torch.nn.BatchNorm1d(features_1.shape[1], affine=False).to(device)
    features_1 = bn(features_1)
    features_2 = bn(features_2)

    c = features_1.T @ features_2
    c.div_(features_1.shape[0])

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag_term = off_diagonal(c).pow_(2).sum()
    return on_diag, off_diag_term


def barlow_loss(features_1, features_2, lambda_param, device):
    on_diag, off_diag_term = barlow_terms(features_1, features_2, device)
    return on_diag + lambda_param * off_diag_term


def barlow_loss_3d(features_1, features_2, lambda_param, device):
    bn = torch.nn.BatchNorm1d(features_1.shape[1], affine=False).to(device)

    features_1 = bn(features_1)
    features_2 = bn(features_2)

    c = features_1.T @ features_2
    c.div_(features_1.shape[0])

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag_term = off_diagonal(c).pow_(2).sum()
    return on_diag + lambda_param * off_diag_term
