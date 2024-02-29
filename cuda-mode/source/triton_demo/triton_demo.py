import torch
import triton
from triton import language as tl

ones = lambda *size: torch.ones(*size).float().cuda()
zeros = lambda *size: torch.zeros(*size).float().cuda()
arange = lambda n: torch.arange(n).float().cuda()
rand = lambda size: torch.rand(*size).abs().float().cuda()


@triton.jit
def triton_hello_world(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    # Use arange to build the shape for loading
    pid = tl.program_id(0)
    Ks = tl.arange(0, K) # K
    Ls = tl.arange(0, L)[:, None] # L x 1
    lid = pid * L

    # Load from memory
    x = tl.load(X + Ks) # K
    y = tl.load(Y + (Ls+lid)*K + Ks) # L x K
    z = x + y # L x K

    # Store
    tl.store(Z + (Ls+lid)*K + Ks, z) # L x K

L = 2**10
x, y = arange(4), ones(L, 4)
z = zeros(L, 4)
triton_hello_world[(1,)](x, y, z, 4, L)


print(z)


@triton.jit
def plus_fn(a, b):
    # This is a helper function where a and b are tensors.
    return a + b

@triton.jit
def cumsum_tt(X, H_0, Y, H, K: tl.constexpr):
    # Which block an I?
    pid = tl.program_id(0)

    # How far into the sequence am I?
    kid = K * pid
    Ks = tl.arange(0, K)

    # Load in K x's per block and 1 starting h
    x = tl.load(X + Ks + kid)

    # Load the first value as H_0 and the rest 0
    h_0 = tl.load(H_0 + Ks * 0 + pid, Ks == 0, 0)

    # Allow for a starting value.
    x = plus_fn(h_0, x)

    # Compute scan
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    # Write out K y's per block and 1 h
    tl.store(Y + Ks + kid, y)

    # Write out only the last value to H
    tl.store(H + Ks * 0 + pid, hs, mask=Ks == (K-1))

K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS

x = arange(SEQLEN)
y = zeros(SEQLEN)

h = zeros(BLOCKS)
cumsum_tt[(BLOCKS,)](x, h, y, h, K=K)

def cumsum_block(x, y, K):
    seqlen = y.shape[0]
    BLOCKS = seqlen // K
    h = zeros(2, BLOCKS)
    cumsum_tt[(BLOCKS,)](x, h[0], y, h[0], K=K)
    h[1, 1:] = h[0].cumsum(0)[:-1]
    cumsum_tt[(BLOCKS,)](x, h[1], y, h[1], K=K)

cumsum_block(x, y, K)



print(y)


print(x.sum())