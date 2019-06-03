from poinc_distance.helpers import torch_poinc_dist_sq, torch_exp_map_zero
import torch

def poinc_distance(u, v, bias=None, c=1., cuda=True):

    N1, N2 = u.size(0), v.size(0)

    if N1 == 1:
        u_rep = u.repeat(N2, 1)
    elif N1 == N2:
        u_rep = u
    else:
        print('N1 neq N2, this is bad!')
        return None

    u_exp = torch_exp_map_zero(u_rep)
    v_exp = torch_exp_map_zero(v)


    return torch_poinc_dist_sq(u_exp, v_exp, c)
