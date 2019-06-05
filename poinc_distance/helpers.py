import torch
import torch.nn as nn
import numpy as np

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0

def torch_atanh(x):
    x = torch.clamp(x, max=1.-EPS)
    return 0.5*torch.log(1+x+PROJ_EPS) - 0.5*torch.log(1-x+PROJ_EPS)

def torch_clip_by_norm(v, clip_norm):
    v_norm = torch.norm(v, dim=1)
    idxs = v_norm > clip_norm

    v_clip = v
    multiplier = torch.ones((v.size(0), 1)).cuda()
    multiplier[idxs] = clip_norm / (EPS + v_norm[idxs, None])
    v_clip = v * multiplier
    return v_clip

def torch_project_hyp_vecs(x, c):
    return torch_clip_by_norm(x, (1 - PROJ_EPS))

def torch_mob_add(u, v, c):
    v = v + EPS
    hb_dot_u_v = 2. * c * torch.diag(torch.nn.functional.linear(u, v))
    hb_norm_u_sq = c * torch.diag(torch.nn.functional.linear(u, u))
    hb_norm_v_sq = c * torch.diag(torch.nn.functional.linear(v, v))
    denominator = 1. + hb_dot_u_v + hb_norm_v_sq * hb_norm_u_sq

    result = (1. + hb_dot_u_v + hb_norm_v_sq)[:,None] / denominator[:,None] * u + (1. - hb_norm_u_sq)[:,None] / denominator[:,None] * v
    return torch_project_hyp_vecs(result, c)

def torch_exp_map_zero(v, c=1):
    v = v + EPS
    norm_v = torch.norm(v, dim=1) 
    result = torch.tanh(torch.clamp(norm_v, max=MAX_TANH_ARG))[:, None] / (norm_v[:, None]) * v
    return torch_project_hyp_vecs(result, c)

def torch_poinc_dist_sq(u, v, c):
    m = torch_mob_add(-u, v, c) + EPS
    atanh_x = torch.norm(m, dim=1)
    dist_poincare = 2 * torch_atanh(atanh_x)
    return dist_poincare.pow(2)

