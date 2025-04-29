import torch
import torch.nn as nn
from utils import reorder_indices, div_up

torch.ops.load_library("../build/lib/libst_pybinding.so")

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class ReorderRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, M: int, BM: int, BN: int, hint: list, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5))
        self.bm = BM
        self.bn = BN
        tm, tn = div_up(M, BM), div_up(dim, BN)
        self.reorder_array = reorder_indices(tm * tn, hint).reshape((tm, tn))
        # self.reorder_array = torch.arange(0, tm * tn, dtype=torch.int, device="cuda").reshape((tm, tn))

    def forward(self, x):
        output = torch.empty((x.size(0), x.size(1)), dtype=torch.float16, device="cuda")
        torch.ops.flashoverlap_op.reorder_rmsnorm(x, output, self.weight, 
            self.bm, self.bn, 8, self.reorder_array)
        return output