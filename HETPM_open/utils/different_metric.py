import torch
import torch.nn.functional as F

def entropy(x):
    b = F.softmax(x) * F.log_softmax(x)
    b = -1.0 * b.sum()
    return b


def discrepancy(out1, out2):
    l1loss = torch.nn.L1Loss()
    return l1loss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))


def discrepancy_mse(out1, out2):
    mseloss = torch.nn.MSELoss()
    return mseloss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))


def discrepancy_cos(out1, out2):
    cosloss = torch.nn.CosineSimilarity()
    return 1 - cosloss(F.softmax(out1, dim=1), F.softmax(out2, dim=1))


def cdd(output_t1, output_t2):
    output_t1, output_t2 = F.softmax(output_t1), F.softmax(output_t2)
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    cdd_loss = cdd_loss * 0.01
    return cdd_loss


def discrepancy_slice_wasserstein(p1, p2):
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist
