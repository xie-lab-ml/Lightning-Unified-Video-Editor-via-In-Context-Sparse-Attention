# Stub: single-GPU inference.
def get_sequence_parallel_state():
    return False


class _NcclInfo:
    sp_size = 1
    rank_within_group = 0
    group = None


nccl_info = _NcclInfo()
