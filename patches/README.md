# Patches

Both patches apply to exllamav3-rocm `360c902`. `scripts/setup.sh` applies them.

## 0001-attention-host-rsqrtf.patch

[`0001-attention-host-rsqrtf.patch`](0001-attention-host-rsqrtf.patch)

The build fails with ROCm 7.2.4 (clang 22). `attention.cu` calls `rsqrtf()` in host code, and clang only
declares a `__device__` version:

```
attention.hip:910:44: error: no matching function for call to 'rsqrtf'
note: candidate function not viable: call to __device__ function from __host__ function
```

The patch replaces the call with `1.0f / sqrtf(dim)`, which gives the same value.

Delete any stale `attention.hip` before rebuilding. Otherwise the old hipified copy is compiled again.

## 0002-dflash2-codebook-weight-suffix.patch

[`0002-dflash2-codebook-weight-suffix.patch`](0002-dflash2-codebook-weight-suffix.patch)

r0b0tlab's DFlash2 draft stores the candidate-selector codebooks as `...predecessor_codebook.weight` and
`...successor_codebook.weight` (nn.Embedding names). The port only looks for the bare names, so the draft fails
to load with a missing-tensor error.

The patch makes the loader try the bare name first, then the name with `.weight`.

## License

The patches in this directory are MIT.
