# LLaMA Transformer — From Scratch

**PyTorch | Large Language Models | Transformer Architecture**

## Overview

A from-scratch implementation of a decoder-only LLaMA-style Transformer in PyTorch, focusing on core architectural components and efficient autoregressive inference.

## Key Implementations

- **RMSNorm:** Root Mean Square Layer Normalization.
- **RoPE:** Rotary Positional Embeddings for position-aware attention.
- **SwiGLU:** Gated Feed-Forward Network.
- **Grouped-Query Attention (GQA):** Efficient attention with shared key-value heads.
- **KV Cache:** Incremental autoregressive decoding without recomputing previous key-value states.

## Architecture

The implementation combines LLaMA's core architectural components into a causal decoder-only Transformer, supporting autoregressive next-token prediction.

## Repository Structure

```text
.
├── README.md
└── llama2_7B.py
