# Stub: single-GPU inference doesn't need distributed collectives.
def all_gather(x, dim=0):
    return [x]

def all_to_all_4D(x, scatter_dim=2, gather_dim=1):
    return x

def broadcast(x, src=0):
    return x
