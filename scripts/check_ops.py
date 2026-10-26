#!/usr/bin/env python3
"""Verify the compiled MTR CUDA extensions (KNN + local attention) on the GPU.

Run inside the mtr container from UniTraj/unitraj (the working directory of the
compose service). Checks that the imported .so files are the freshly compiled ones,
then runs each kernel against a pure-PyTorch reference.
"""
from __future__ import annotations
import os
import sys
import time

import torch

OPS = '/opt/valeo4cast/UniTraj/unitraj/models/mtr/ops'
sys.path.insert(0, OPS)

from knn import knn_cuda  # noqa: E402
from attention import attention_cuda  # noqa: E402
from knn.knn_utils import knn_batch, knn_batch_mlogk  # noqa: E402
from attention.attention_utils import attention_weight_computation, attention_value_computation  # noqa: E402

print('GPU:', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
for mod in (knn_cuda, attention_cuda):
    age_h = (time.time() - os.path.getmtime(mod.__file__)) / 3600
    print(f'{mod.__name__}: {mod.__file__}  (built {age_h:.1f} h ago)')
    assert age_h < 48, 'stale prebuilt .so from the repository is being imported'

torch.manual_seed(0)
dev = 'cuda'

# --- KNN: two batches of points, 8 queries each, k nearest keys per query --------------------
n_per, m_per, k = 300, 32, 16  # k must not exceed queries per batch
xyz = torch.rand(2 * n_per, 3, device=dev)
batch_idxs = torch.repeat_interleave(torch.arange(2, device=dev), n_per).int()
query_xyz = torch.rand(2 * m_per, 3, device=dev)
query_offsets = torch.tensor([0, m_per, 2 * m_per], dtype=torch.int32, device=dev)

# Reference: for each key point, the k nearest queries within the same batch (that is the
# convention of knn_batch: idx is (n, k)).
ref = torch.empty(2 * n_per, k, dtype=torch.long, device=dev)
for b in range(2):
    keys = xyz[b * n_per:(b + 1) * n_per]
    qs = query_xyz[b * m_per:(b + 1) * m_per]
    d = torch.cdist(keys, qs)
    ref[b * n_per:(b + 1) * n_per] = d.topk(min(k, m_per), largest=False).indices + b * m_per

# The kernel may report query indices either globally (over all m queries) or locally
# (relative to the first query of the same batch); accept whichever it implements and say so.
ref_global = ref.sort(dim=1).values
ref_local = (ref - (batch_idxs.long() * m_per)[:, None]).sort(dim=1).values
for name, fn in (('knn_batch', knn_batch), ('knn_batch_mlogk', knn_batch_mlogk)):
    idx = fn(xyz.contiguous(), query_xyz.contiguous(), batch_idxs.contiguous(), query_offsets, k)
    torch.cuda.synchronize()
    got = idx.long().sort(dim=1).values
    agree_global = (got == ref_global).float().mean().item()
    agree_local = (got == ref_local).float().mean().item()
    convention = 'batch-local' if agree_local > agree_global else 'global'
    print(f'{name}: idx {tuple(idx.shape)}, agreement with torch.cdist reference = '
          f'{max(agree_local, agree_global):.3f} ({convention} query indices)')
    assert max(agree_local, agree_global) > 0.99, f'{name} disagrees with reference'

# --- Local attention: weights and values against a dense reference -------------------------
bs, nq, nk, local, nhead, hdim = 2, 10, 40, 6, 4, 8
query_batch_cnt = torch.full((bs,), nq, dtype=torch.int32, device=dev)
key_batch_cnt = torch.full((bs,), nk, dtype=torch.int32, device=dev)
index_pair_batch = torch.repeat_interleave(torch.arange(bs, device=dev), nq).int()
index_pair = torch.stack([torch.randperm(nk, device=dev)[:local] for _ in range(bs * nq)]).int()
q = torch.randn(bs * nq, nhead, hdim, device=dev)
kf = torch.randn(bs * nk, nhead, hdim, device=dev)
v = torch.randn(bs * nk, nhead, hdim, device=dev)

w = attention_weight_computation(query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, q, kf)
# reference: w[i, j, h] = q[i, h] . k[batch_offset + index_pair[i, j], h]
key_global = index_pair.long() + (index_pair_batch.long() * nk)[:, None]
ref_w = torch.einsum('ihd,ijhd->ijh', q, kf[key_global])
err_w = (w - ref_w).abs().max().item()
print(f'attention_weight_computation: out {tuple(w.shape)}, max abs err vs dense = {err_w:.2e}')
assert err_w < 1e-4

out = attention_value_computation(query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, w.contiguous(), v)
ref_out = torch.einsum('ijh,ijhd->ihd', w, v[key_global])
err_v = (out - ref_out).abs().max().item()
print(f'attention_value_computation:  out {tuple(out.shape)}, max abs err vs dense = {err_v:.2e}')
assert err_v < 1e-4
print('SUCCESS: compiled KNN and attention kernels run on sm_120 and match PyTorch references.')
