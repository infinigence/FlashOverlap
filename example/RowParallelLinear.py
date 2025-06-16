import torch.nn as nn
import torch
import torch.distributed as dist
from utils import reorder_indices, div_up, generate_row_mapping

torch.ops.load_library("../build/lib/libst_pybinding.so")

class RowParallelLayer(nn.Module):
    def __init__(self, in_features, out_features, comm_op, tp_group):
        super().__init__()

        assert comm_op in ["all_reduce", "reduce_scatter"], \
            f"comm_op must be 'all_reduce' or 'reduce_scatter', but got '{comm_op}'"

        self.tp_group = tp_group
        self.comm_op = comm_op
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
    
    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        if self.comm_op == "all_reduce":
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
            return out
        elif self.comm_op == "reduce_scatter":
            cout = torch.empty((x.size(0) // dist.get_world_size(), self.weight.size(0)), 
                dtype=torch.float16, device="cuda")
            dist.reduce_scatter_tensor(cout, out, group=self.tp_group)
            return cout
        else:
            return out

class OverlapRowParallelLayer(nn.Module):
    def __init__(self, rank: int, world_size: int, in_features: int, out_features: int, 
        M: int, config: dict, comm_op: str, nccl_id):
        super().__init__()

        assert comm_op in ["all_reduce", "reduce_scatter"], \
            f"comm_op must be 'all_reduce' or 'reduce_scatter', but got '{comm_op}'"

        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))

        self.world_size = world_size
        self.comm_op = comm_op

        self.overlap_class = torch.classes.flashoverlap_class.OverlapImpl()
        self.overlap_class.nccl_init(rank, world_size, nccl_id)
        self.overlap_class.cutlass_init()
        self.overlap_class.overlap_init()

        BM = config["BM"]
        BN = config["BN"]
        hint = config["hint"]
        cseg = config["cSeg"]

        tm, tn = div_up(M, BM), div_up(out_features, BN)
        self.algo = config["Algo"]
        self.counter = torch.zeros((1, tn), dtype=torch.int, device="cuda")
        self.reorder_array = reorder_indices(tm * tn, hint).reshape((tm, tn))

        self.cseg_cpu = torch.tensor(cseg, dtype=torch.int32) 
        self.cseg_gpu = self.cseg_cpu.cuda(rank)

        if comm_op == "reduce_scatter":
            self.row_array = generate_row_mapping(M, out_features, BM, BN, cseg, world_size)
        else:
            self.row_array = None
        
    def forward(self, x):
        if self.comm_op == "all_reduce":
            out = torch.empty((x.size(0), self.weight.size(0)), dtype=torch.float16, device="cuda")
            self.overlap_class.gemm_allreduce_overlap(
                x, self.weight, out, self.counter, self.reorder_array, 1, self.cseg_cpu, self.cseg_gpu, self.algo, False)
        elif self.comm_op == "reduce_scatter":
            tmp = torch.empty((x.size(0), self.weight.size(0)), dtype=torch.float16, device="cuda")
            out = torch.empty((x.size(0) // self.world_size, self.weight.size(0)), dtype=torch.float16, device="cuda")
            self.overlap_class.gemm_reducescatter_overlap(
                x, self.weight, tmp, out, self.counter, self.reorder_array, self.row_array, 
                1, self.cseg_cpu, self.cseg_gpu, self.algo, False
            )
        else:
            out = torch.empty((x.size(0), self.weight.size(0)), dtype=torch.float16, device="cuda")
        return out
