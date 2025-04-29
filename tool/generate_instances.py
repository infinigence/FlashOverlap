import itertools
import json
import torch

candidates = {
    'ThreadblockM': [128, 256],
    'ThreadblockN': [128, 256],
    'ThreadblockK': [32, 64],
    'WarpM': [64],
    'WarpN': [64],
    'WarpK': [32, 64],
    'InstructionM': [16],
    'InstructionN': [8],
    'InstructionK': [16],
    'NumStages': [3, 4, 5],
    'SwizzleSize': [1, 2, 3, 4, 6, 8],
    'SplitK': [1]
}

all_combinations = itertools.product(*candidates.values())
valid_combinations = []
for combo in all_combinations:
    tbm, tbn, tbk, wm, wn, wk, _, _, _, st, sw, sk = combo
    # if tbm == 128 and tbn == 256 and tbk == 32 and wm == 64 and wn == 64 and wk == 32 and st == 5 and sw == 8:
    #     continue
    if tbm == 128 and tbn == 256 and tbk == 64 and wm == 64 and wn == 64 and wk == 64 and st >= 4:
        continue
    if tbm == 128 and tbn == 256 and tbk == 64 and wm == 64 and wn == 64 and wk == 32:
        continue
    if tbm == 256 and tbn == 128 and tbk == 64 and wm == 64 and wn == 64 and wk == 32:
        continue
    if tbm == 256 and tbn == 128 and tbk == 64 and wm == 64 and wn == 64 and wk == 64 and st >= 4:
        continue
    if tbm == 256 and tbn == 256:
        continue
    if tbm % wm == 0 and tbn % wn == 0 and tbk % wk == 0:
        valid_combinations.append(combo)

# 生成字典：key为参数元组，value为index
index_dict = {combo: idx for idx, combo in enumerate(valid_combinations)}

torch.save(
    index_dict,  
    "../configs/AlgoDict.pt"
)

with open('../src/inc/gemm_instances.inc', 'w') as f_inc, \
     open('../src/tiling/gemm_tiling.cuh', 'w') as f_table:

    # 生成实例化代码
    f_table.write("#include \"gemm_dispatcher.h\"\n\n")

    f_table.write("GemmFuncPtr gemm_func_table[] = {\n")

    for idx, combo in enumerate(valid_combinations):
        args = ', '.join(map(str, combo))
        
        # 写入实例化代码
        f_inc.write(f'CUTLASS_GEMM_SPLITK_INIT({args});\n')
        
        # 写入函数指针表
        f_table.write(f"    &cutlass_gemm_splitk<{args}>,\n")

    f_table.write("};\n")

with open('../src/inc/signal_instances.inc', 'w') as f_inc, \
     open('../src/tiling/signal_tiling.cuh', 'w') as f_table:

    # 生成实例化代码
    f_table.write("#include \"gemm_dispatcher.h\"\n\n")

    f_table.write("SignalFuncPtr signal_func_table[] = {\n")

    for idx, combo in enumerate(valid_combinations):
        args = ', '.join(map(str, combo))
        
        # 写入实例化代码
        f_inc.write(f'CUTLASS_GEMM_SIGNAL_INIT({args});\n')
        
        # 写入函数指针表
        f_table.write(f"    &cutlass_gemm_signal<{args}>,\n")

    f_table.write("};\n")