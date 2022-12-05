# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch.special import expm1
import math
from einops import rearrange, reduce

# convert to bit representations and back
def decimal_to_bits(x, vocab_size, bits):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    x = x.clamp(0, vocab_size-1)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b c h w -> b c 1 h w')

    bits = ((x & mask) != 0).float()
    # bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits.squeeze(-1).squeeze(-1) # batch_size x seq_length x bits x 1 x 1 -> batch_size x seq_length x bits
    bits = bits * 2 - 1
    return bits

def bits_to_decimal(x, vocab_size, bits):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device
    x = rearrange(x, 'b s d -> b d s 1') # the same as img: batch_size x bit_dim x weight x height

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = bits)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    # dec = (dec / (vocab_size-1)).clamp(0., 1.)
    dec = dec.squeeze(-1)
    dec = rearrange(dec, 'b d s -> b s d')
    dec = dec.squeeze(-1)

    return dec


# bit diffusion class

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))