/***************************************************************************************************
 * Adapted from SwiftTransformer(https://github.com/LLMServe/SwiftTransformer/blob/main/src/csrc/util/py_nccl.cc)
 **************************************************************************************************/

#include "nccl_utils.h"

std::vector<int64_t> generate_nccl_id() {
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    std::vector<int64_t> ret;
    ret.resize(NCCL_UNIQUE_ID_BYTES / sizeof(int64_t));
    memcpy(ret.data(), nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
    return ret;
}
