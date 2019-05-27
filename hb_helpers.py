import torch
import torch.nn
import numpy as np

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0

def hb_atanh(x):
    return 0.5*torch.log((1+x)/(1-x))


def hp_clip_by_norm(v, clip_norm):
    v_norm = torch.norm(v)
    if v_norm <= clip_norm:
        return v
    else:
        v_clip = v * clip_norm / v_norm
        return v_clip

def hb_project_hyp_vecs(x, c):
    return hp_clip_by_norm(x, (1 - PROJ_EPS))

def hb_mob_add(u, v, c):
    v = v + EPS
    hb_dot_u_v = 2. * c * torch.dot(u, v)
    hb_norm_u_sq = c * torch.dot(u, u)
    hb_norm_v_sq = c * torch.dot(v, v)
    denominator = 1. + hb_dot_u_v + hb_norm_v_sq + hb_norm_u_sq
    result = (1. + hb_dot_u_v + hb_norm_v_sq) / denominator * u + (1. - hb_norm_u_sq) / denominator * v
    return hb_project_hyp_vecs(result, c)

def hb_exp_map_zero(v, c=1):
    v = v + EPS
    norm_v = torch.norm(v)
    result = torch.tanh(norm_v) / (np.sqrt(c) * norm_v) * v
    return hb_project_hyp_vecs(result, c)

def hb_log_map_zero(y, c):
    diff = y + EPS
    norm_diff = torch.norm(diff)
    return 1. / torch.atanh(norm_diff) / norm_diff * diff

def hb_poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    m = hb_mob_add(-u, v, c) + EPS
    atanh_x = torch.norm(m)
    dist_poincare = hb_atanh(atanh_x)
    return dist_poincare ** 2

def pairwise_poinc_distance(u, v, bias=None, c=1.):

    N1 = u.size(0)
    N2 = v.size(0)

    if N1 == 1:
        distance = torch.zeros(N2).cuda()
        for i in range(N2):
            distance[i] = hb_poinc_dist_sq(u[0], v[i], c)
    elif N1 == N2:
        distance = torch.zeros(N2).cuda()
        for i in range(N2):
            distance[i] = hb_poinc_dist_sq(u[i], v[i], c)
    else:
        print('THIS IS BAD')
        return None

    return distance


