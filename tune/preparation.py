import os
from tqdm import tqdm

# Use representative test cases
# test_cases_ar_rs_a800 = [
#     (8192, 8192, 2048), (8192, 8192, 4096), (8192, 8192, 8192), (16384, 8192, 2048), (16384, 8192, 4096), (16384, 8192, 8192), (32768, 8192, 2048), (32768, 8192, 4096), (32768, 8192, 8192)
# ]
# test_cases_a2a_a800 = [
#     (4096, 8192, 4096), (4096, 8192, 8192), (8192, 8192, 4096), (8192, 8192, 8192), (16384, 8192, 4096),  (16384, 8192, 8192)
# ]
test_cases_ar_rs_a800 = [
    (8192, 8192, 2048)
]
test_cases_a2a_a800 = [
    (4096, 8192, 4096)
]

# print("Capturing the bandwidth curve ...")
# for i in [2, 4, 8]:
#     os.system('python3 bandwidth.py --comm_op all_reduce --world_size %d' % i)
#     os.system('python3 bandwidth.py --comm_op reduce_scatter --world_size %d' % i)

# print("Start tuning ...")
# for case in tqdm(test_cases_ar_rs_a800):
#     m, n, k = case[0], case[1], case[2]
#     os.system('python3 profile_config.py --m %d --n %d --k %d' % (m, n, k))
#     for i in [2, 4, 8]:
#         os.system('python3 search.py --m %d --n %d --k %d --comm_op all_reduce --world_size %d' % (m, n, k, i))
#         os.system('python3 search.py --m %d --n %d --k %d --comm_op reduce_scatter --world_size %d' % (m, n, k, i))

for case in tqdm(test_cases_a2a_a800):
    m, n, k = case[0], case[1], case[2]
    for i in [2, 4, 8]:
        os.system('python3 search_a2a.py --m %d --n %d --k %d --world_size %d' % (m, n, k, i))
