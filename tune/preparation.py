import os
from tqdm import tqdm
import torch

# Test cases
device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
gpu_name = props.name[7:11].lower()
if gpu_name == 'a800':
    test_cases_ar_rs = [
        (8192, 8192, 2048), (8192, 8192, 4096), (8192, 8192, 8192), (16384, 8192, 2048), (16384, 8192, 4096), (16384, 8192, 8192), (16384, 16384, 2048), (16384, 16384, 4096), (16384, 16384, 8192)
    ]
    test_cases_a2a = [
        (4096, 8192, 4096), (4096, 8192, 8192), (8192, 8192, 4096), (8192, 8192, 8192), (8192, 16384, 4096),  (8192, 16384, 8192)
    ]
else:
    test_cases_ar_rs = [
        (2048, 8192, 8192), (2048, 8192, 16384),  (3072, 8192, 8192), (3072, 8192, 16384),  (4096, 8192, 8192), (4096, 8192, 16384),  (8192, 8192, 8192), (8192, 8192, 16384),
    ]
    test_cases_a2a = [
        (512, 8192, 8192), (512, 8192, 16384), (1024, 8192, 8192), (1024, 8192, 16384),  (2048, 8192, 8192), (2048, 8192, 16384), (2560, 8192, 8192), (2560, 8192, 16384),
    ]


print("Capturing the bandwidth curve ...")
for i in [2, 4, 8]:
    os.system('python3 bandwidth.py --comm_op all_reduce --world_size %d' % i)
    os.system('python3 bandwidth.py --comm_op reduce_scatter --world_size %d' % i)

print("Start tuning ...")
for case in tqdm(test_cases_ar_rs):
    m, n, k = case[0], case[1], case[2]
    os.system('python3 profile_config.py --m %d --n %d --k %d' % (m, n, k))
    for i in [2, 4, 8]:
        os.system('python3 search.py --m %d --n %d --k %d --comm_op all_reduce --world_size %d' % (m, n, k, i))
        os.system('python3 search.py --m %d --n %d --k %d --comm_op reduce_scatter --world_size %d' % (m, n, k, i))

for case in tqdm(test_cases_a2a):
    m, n, k = case[0], case[1], case[2]
    for i in [2, 4, 8]:
        os.system('python3 search_a2a.py --m %d --n %d --k %d --world_size %d' % (m, n, k, i))
