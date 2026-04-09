# FLUX.2 [klein] 9B KV Cache

FLUX.2 [klein] 9B KV is a variant of the klein 9B model optimized for fast image editing. It uses KV caching to avoid redundantly recomputing attention over reference image tokens at every denoising step, providing significant speedups when working with reference images.

## How it works

In standard image editing, each denoising step concatenates reference image tokens with the noisy output tokens and runs full attention over the entire sequence. Since reference tokens don't change across steps, this is wasteful.

The KV cache variant splits denoising into two phases:

1. **Step 0 — `forward_kv_extract`**: Full forward pass with reference tokens included. Extracts and caches the key/value projections for the reference tokens.
2. **Steps 1+ — `forward_kv_cached`**: Forward pass with only the output and text tokens. Reuses the cached reference KVs via concatenation in the attention layers.

This means reference tokens are only processed once, regardless of the number of denoising steps.

## Speedup

Speedup depends on the ratio of reference tokens to output tokens. More references and smaller output resolutions yield larger gains:

| #Refs (1024x1024 each) | 512x512 | 768x768 | 1024x1024 | 1440x1440 |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 1.78x | 1.57x | 1.40x | 1.21x |
| 2 | 2.16x | 1.97x | 1.77x | 1.46x |
| 3 | 2.43x | 2.21x | 1.99x | 1.69x |
| 4 | 2.66x | 2.44x | 2.22x | 1.85x |

## Installation

Follow the standard [installation instructions](../README.md#local-installation).

## Usage

### Environment variables


```bash
export KLEIN_9B_KV_MODEL_PATH="/path/to/flux-2-klein-9b-kv.safetensors"
export AE_MODEL_PATH="/path/to/ae.safetensors"  # optional, auto-downloads if unset
```

### CLI

The CLI automatically uses the KV cache path when you select the `flux.2-klein-9b-kv` model:

```bash
PYTHONPATH=src python scripts/cli.py --model_name flux.2-klein-9b-kv
```

Then provide reference images for editing:

```
> input_images="ref1.jpg,ref2.jpg"
> prompt="a cat wearing sunglasses"
> run
```
