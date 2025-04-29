import torch.nn as nn
import torch
import torch.distributed as dist
from utils import reorder_indices, div_up

torch.ops.load_library("../build/lib/libst_pybinding.so")

class RowParallelLayer(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
    
    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        return out

class OverlapRowParallelLayer(nn.Module):
    def __init__(self, rank: int, world_size: int, in_features: int, out_features: int, 
        M: int, BM: int, BN: int, hint: list, cseg: list, algo: int, nccl_id):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.overlap_class = torch.classes.flashoverlap_class.OverlapImpl()
        self.overlap_class.nccl_init(rank, world_size, nccl_id)
        self.overlap_class.cutlass_init()
        self.overlap_class.overlap_init()

        tm, tn = div_up(M, BM), div_up(out_features, BN)
        self.algo = algo
        self.counter = torch.zeros((1, tn), dtype=torch.int, device="cuda")
        self.reorder_array = reorder_indices(tm * tn, hint).reshape((tm, tn))
        # self.reorder_array = torch.arange(0, tm * tn, dtype=torch.int, device="cuda").reshape((tm, tn))

        self.cseg_cpu = torch.tensor(cseg, dtype=torch.int32) 
        self.cseg_gpu = self.cseg_cpu.cuda(rank)
    
    def forward(self, x):
        out = torch.zeros((x.size(0), self.weight.size(0)), dtype=torch.float16, device="cuda")
        self.overlap_class.gemm_allreduce_overlap(
            x, self.weight, out, self.counter, self.reorder_array, 1, self.cseg_cpu, self.cseg_gpu, self.algo, False)
        return out
