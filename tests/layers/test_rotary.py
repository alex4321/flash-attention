# Copyright (c) 2023, Tri Dao.

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding as RotaryEmbeddingNeoX
from transformers.models.gpt_neox.modeling_gpt_neox import rotate_half as rotate_half_gpt_neox
from transformers.models.gptj.modeling_gptj import rotate_every_two
from flash_attn.layers.rotary import apply_rotary_emb_func, apply_rotary_emb_qkv_
from flash_attn.layers.rotary import RotaryEmbedding


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):  # no changes in this function
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def apply_rotary_pos_emb_neox(q, k, cos, sin, offset):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half_gpt_neox(q) * sin)
    k_embed = (k * cos) + (rotate_half_gpt_neox(k) * sin)
    return q_embed, k_embed


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb_gptj(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


# NeoX-style rotary embedding
@pytest.mark.parametrize('seqlen_offset', [0, 711])
@pytest.mark.parametrize('rotary_emb_fraction', [0.5, 1.0])
def test_rotary(rotary_emb_fraction, seqlen_offset):
    device = 'cuda'
    dtype = torch.float16
    rtol, atol = (1e-3, 5e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen_total = 2048
    seqlen = seqlen_total - seqlen_offset
    nheads = 16
    headdim = 128
    rotary_dim = int(headdim * rotary_emb_fraction)
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                      requires_grad=True)
    qkv_og = qkv.clone().detach()  # Our implementation modifies qkv inplace
    rotary = RotaryEmbedding(rotary_dim, device=device)
    rotary_neox = RotaryEmbeddingNeoX(rotary_dim, seqlen_total, device=device)
    # Doesn't matter what tensor we pass in, rotary_neox only uses the device of the tensor
    cos_neox, sin_neox = rotary_neox(qkv, seq_len=seqlen_total)
    cos_neox, sin_neox = cos_neox.to(dtype=dtype), sin_neox.to(dtype=dtype)
    q_pt = rearrange(qkv[:, :, 0, :, :rotary_dim],
                     'b s h d -> b h s d').detach().clone().requires_grad_(True)
    k_pt = rearrange(qkv[:, :, 1, :, :rotary_dim],
                     'b s h d -> b h s d').detach().clone().requires_grad_(True)
    q_neox, k_neox = apply_rotary_pos_emb_neox(q_pt, k_pt, cos_neox, sin_neox, offset=seqlen_offset)
    out = rotary(qkv, seqlen_offset=seqlen_offset)
    assert torch.allclose(rotary._cos_cached, cos_neox[..., :rotary_dim // 2].to(dtype=dtype),
                          rtol=rtol, atol=atol)
    assert torch.allclose(rotary._sin_cached, sin_neox[..., :rotary_dim // 2].to(dtype=dtype),
                          rtol=rtol, atol=atol)
    assert torch.allclose(rearrange(q_neox, 'b h s d -> b s h d'), out[:, :, 0, :, :rotary_dim],
                          rtol=rtol, atol=atol)
    assert torch.allclose(rearrange(k_neox, 'b h s d -> b s h d'), out[:, :, 1, :, :rotary_dim],
                          rtol=rtol, atol=atol)
    assert torch.equal(out[:, :, 0:2, :, rotary_dim:], qkv_og[:, :, 0:2, :, rotary_dim:])
    assert torch.equal(out[:, :, 2], qkv_og[:, :, 2])

    g = torch.randn_like(out)
    g_og = g.clone().detach()  # Our implementation modifies g inplace
    out.backward(g)
    q_neox.backward(rearrange(g_og[:, :, 0, :, :rotary_dim], 'b s h d -> b h s d'))
    k_neox.backward(rearrange(g_og[:, :, 1, :, :rotary_dim], 'b s h d -> b h s d'))
    assert torch.allclose(rearrange(q_pt.grad, 'b h s d -> b s h d'),
                          qkv.grad[:, :, 0, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.allclose(rearrange(k_pt.grad, 'b h s d -> b s h d'),
                          qkv.grad[:, :, 1, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.equal(qkv.grad[:, :, 0:2, :, rotary_dim:], g_og[:, :, 0:2, :, rotary_dim:])
    assert torch.equal(qkv.grad[:, :, 2], g_og[:, :, 2])


# GPT-J-style rotary embedding
@pytest.mark.parametrize('seqlen_offset', [0, 711])
@pytest.mark.parametrize('rotary_emb_fraction', [0.5, 1.0])
def test_rotary_interleaved(rotary_emb_fraction, seqlen_offset):
    device = 'cuda'
    dtype = torch.float16
    rtol, atol = (1e-3, 5e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen_total = 2048
    seqlen = seqlen_total - seqlen_offset
    nheads = 16
    headdim = 128
    rotary_dim = int(headdim * rotary_emb_fraction)
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                      requires_grad=True)
    qkv_og = qkv.clone().detach()  # Our implementation modifies qkv inplace
    rotary = RotaryEmbedding(rotary_dim, interleaved=True, device=device)
    sincos_gptj = fixed_pos_embedding(qkv[..., :rotary_dim], seq_dim=1, seq_len=seqlen_total)
    sincos_gptj = tuple(x.to(dtype=dtype) for x in sincos_gptj)
    q_pt = qkv[:, :, 0, :, :rotary_dim].detach().clone().requires_grad_(True)
    k_pt = qkv[:, :, 1, :, :rotary_dim].detach().clone().requires_grad_(True)
    q_gptj = apply_rotary_pos_emb_gptj(q_pt, sincos_gptj, offset=seqlen_offset)
    k_gptj = apply_rotary_pos_emb_gptj(k_pt, sincos_gptj, offset=seqlen_offset)
    out = rotary(qkv, seqlen_offset=seqlen_offset)
    assert torch.allclose(rotary._cos_cached, sincos_gptj[1], rtol=rtol, atol=atol)
    assert torch.allclose(rotary._sin_cached, sincos_gptj[0], rtol=rtol, atol=atol)
    assert torch.allclose(q_gptj, out[:, :, 0, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.allclose(k_gptj, out[:, :, 1, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.equal(out[:, :, 0:2, :, rotary_dim:], qkv_og[:, :, 0:2, :, rotary_dim:])
    assert torch.equal(out[:, :, 2], qkv_og[:, :, 2])

    g = torch.randn_like(out)
    g_og = g.clone().detach()  # Our implementation modifies g inplace
    out.backward(g)
    q_gptj.backward(g_og[:, :, 0, :, :rotary_dim])
    k_gptj.backward(g_og[:, :, 1, :, :rotary_dim])
    assert torch.allclose(q_pt.grad, qkv.grad[:, :, 0, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.allclose(k_pt.grad, qkv.grad[:, :, 1, :, :rotary_dim], rtol=rtol, atol=atol)
    assert torch.equal(qkv.grad[:, :, 0:2, :, rotary_dim:], g_og[:, :, 0:2, :, rotary_dim:])
    assert torch.equal(qkv.grad[:, :, 2], g_og[:, :, 2])
