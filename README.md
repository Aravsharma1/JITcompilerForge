# Forge: Profile-Guided JIT Compiler for LLM Decode Kernels

Forge is a profile-guided just-in-time compiler for LLM decode kernels. It profiles live inference workloads, identifies the current operating point of the serving system, autotunes Triton kernel parameters for that workload, caches compiled kernel variants, and hot-swaps validated kernels at safe decode-step boundaries.

The project focuses on the decode phase of LLM inference, where generation happens one token at a time and performance is often limited by memory bandwidth rather than raw compute throughput.

## Current Prototype Status

The repository now includes a small Python prototype of the Forge control plane:

```text
runtime profiler
deterministic autotuner
JSON kernel cache
step-boundary hot-swap manager
serving-loop simulation
basic tests
```

The current kernel is a simulated decode-attention kernel, not a real Triton/CUDA implementation yet. This lets the architecture run on a normal development machine before adding GPU-specific code.

Run the demo:

```bash
python3 scripts/run_server.py
```

Run tests:

```bash
python3 -m pytest
```

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Insight](#core-insight)
3. [Problem Statement](#problem-statement)
4. [Why Decode Is Memory-Bandwidth-Bound](#why-decode-is-memory-bandwidth-bound)
5. [What Forge Builds](#what-forge-builds)
6. [System Architecture](#system-architecture)
7. [Major Components](#major-components)
8. [Runtime Flow](#runtime-flow)
9. [Triton Kernel Template](#triton-kernel-template)
10. [Profile-Guided Optimization](#profile-guided-optimization)
11. [Autotuning Strategy](#autotuning-strategy)
12. [Kernel Cache](#kernel-cache)
13. [Hot-Swap Mechanism](#hot-swap-mechanism)
14. [Benchmarking Plan](#benchmarking-plan)
15. [Expected Results](#expected-results)
16. [Why This Is a Compiler Project](#why-this-is-a-compiler-project)
17. [Why This Is Different From AgentIR](#why-this-is-different-from-agentir)
18. [Tech Stack](#tech-stack)
19. [Repository Structure](#repository-structure)
20. [Setup](#setup)
21. [Usage](#usage)
22. [Roadmap](#roadmap)
23. [Resume and Interview Framing](#resume-and-interview-framing)

---

## Project Overview

Forge is a runtime optimization system for LLM inference. It specializes GPU kernels for the live decode workload currently being served.

Instead of relying on a single statically compiled attention kernel, Forge continuously observes the serving workload and uses that profile to compile better kernels for the current batch size, KV cache length, and hardware conditions.

At a high level, Forge does the following:

```text
Profile live decode workload
        ↓
Build workload specification
        ↓
Search over Triton kernel configurations
        ↓
Compile and benchmark candidate kernels
        ↓
Select the best validated kernel
        ↓
Cache the compiled kernel
        ↓
Hot-swap it into the serving loop
```

The goal is to improve decode throughput and latency by adapting kernel configuration to the workload instead of assuming one static configuration is optimal for all conditions.

---

## Core Insight

The central insight behind Forge is:

> The decode phase of LLM inference is often memory-bandwidth-bound, not compute-bound, and the optimal GPU kernel configuration changes dynamically with batch size and KV cache length.

During decode, the model generates one token at a time. Each decode step repeatedly reads from the KV cache and model memory. Since the GPU spends significant time moving data from memory, kernel performance depends heavily on how the kernel performs memory access, tiling, vectorization, and pipelining.

A static kernel configuration cannot be optimal for every serving condition. A workload with small batch size and long sequence length stresses the GPU differently from a workload with large batch size and short sequence length.

Forge exploits this by recompiling Triton kernels at serving time using live profile data.

---

## Problem Statement

Most LLM serving systems use generic or statically tuned kernels. These kernels are usually selected ahead of time based on representative benchmarks. However, real serving workloads are dynamic.

For example, a serving system may experience:

```text
Period 1: bursty short requests
batch size: high
KV cache length: short to medium

Period 2: sustained long requests
batch size: lower
KV cache length: long

Period 3: mixed traffic
batch size: unstable
KV cache length: varied
```

The same attention kernel configuration is unlikely to be optimal across all three periods.

Forge addresses this problem by building a profile-guided JIT system that adapts the kernel configuration as the workload shifts.

---

## Why Decode Is Memory-Bandwidth-Bound

LLM inference has two major phases:

1. **Prefill**
   - Processes the input prompt.
   - Usually involves larger matrix multiplications.
   - Often has more parallelism.

2. **Decode**
   - Generates one token at a time.
   - Repeatedly attends over previous tokens stored in the KV cache.
   - Often has less compute per memory byte loaded.

During decode, the GPU frequently reads from the KV cache. The amount of data moved from high-bandwidth memory can dominate runtime. This means performance is constrained by memory traffic and memory reuse rather than only arithmetic throughput.

The important consequence is:

```text
Better kernel tiling and memory access patterns can improve decode performance.
```

That is the opportunity Forge targets.

---

## What Forge Builds

Forge consists of the following pieces:

```text
1. Triton decode attention kernel template
2. Runtime profiler
3. Background autotuner
4. Kernel cache
5. Hot-swap mechanism
6. Benchmark harness
```

The Triton kernel template is parameterized by compile-time constants:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

Forge changes these parameters based on the observed workload and recompiles the kernel through Triton's JIT compilation flow.

---

## System Architecture

The architecture follows the design below:

```text
                            ┌───────────────────────────────────────┐
                            │ serving loop                           │
                            │ decode step + batch tensor             │
                            │ active kernel pointer                  │
                            └───────────────────────────────────────┘
                                  │                         ▲
                                  │ step metrics             │ reads kernel pointer
                                  ▼                         │
┌───────────────────────┐     workload spec      ┌───────────────────────┐
│ profiler              │ ─────────────────────▶ │ autotuner              │
│ batch size tracking   │                        │ background thread      │
│ sequence length dist  │                        │ search + benchmark     │
└───────────────────────┘                        └───────────────────────┘
                                                          │
                                                          │ lookup/cache hit
                                                          ▼
                                                ┌───────────────────────┐
                                                │ kernel cache           │
                                                │ keyed by               │
                                                │ batch, seq_len, model  │
                                                └───────────────────────┘
                                                          │
                                                          │ new kernel
                                                          ▼
                                                ┌───────────────────────┐
                                                │ hot-swap               │
                                                │ double buffer          │
                                                │ step-boundary swap     │
                                                └───────────────────────┘
```

The serving loop never blocks while the optimizer is searching. The autotuner runs in the background, compiles and benchmarks candidate kernels, and only updates the serving loop when a validated kernel is ready.

---

## Major Components

### 1. Serving Loop

The serving loop is the main execution path of the LLM inference system. It repeatedly performs decode steps.

At each decode step, it:

```text
1. Reads the active kernel pointer.
2. Launches the currently active decode kernel.
3. Produces the next token for each active request.
4. Emits step metrics to the profiler.
5. Checks whether a validated replacement kernel is ready.
```

The serving loop must remain stable and non-blocking. Forge's background optimization should not interrupt token generation.

---

### 2. Profiler

The profiler collects runtime information from recent decode steps.

It tracks:

```text
batch size
KV cache length
sequence length distribution
possibly GPU utilization
recent workload history
```

Rather than reacting to a single step, the profiler maintains a rolling window over the last `N` decode steps. This creates a more stable view of the workload.

Example profile:

```json
{
  "batch_size_bucket": "8-16",
  "seq_len_bucket": "1024-2048",
  "model_config_hash": "llama_7b_head128",
  "window_size": 256
}
```

The profiler detects when the workload distribution shifts significantly. Once the shift exceeds a threshold, it emits a workload specification to the autotuner.

---

### 3. Autotuner

The autotuner runs in a background thread. Its job is to find the best Triton kernel configuration for the current workload specification.

It searches over:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

For each candidate configuration, it:

```text
1. Compiles the Triton kernel with the candidate parameters.
2. Benchmarks the candidate for a fixed number of decode steps.
3. Measures throughput and latency.
4. Records the result.
5. Chooses the best validated configuration.
```

The proposal targets a short search of approximately 8 to 16 candidate configurations, with each candidate benchmarked for about 50 decode steps.

---

### 4. Kernel Cache

The kernel cache avoids unnecessary recompilation.

It is keyed by:

```text
(batch_size_bucket, seq_len_bucket, model_config_hash)
```

Each cache entry stores:

```text
best configuration
compiled kernel
benchmark metadata
validation status
timestamp
```

Example cache entry:

```json
{
  "key": {
    "batch_size_bucket": "8-16",
    "seq_len_bucket": "1024-2048",
    "model_config_hash": "llama_7b_head128"
  },
  "value": {
    "BLOCK_M": 16,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "num_warps": 4,
    "num_stages": 3,
    "throughput_tokens_per_sec": 12450.7,
    "p50_latency_ms": 8.1,
    "p95_latency_ms": 10.4
  }
}
```

If Forge sees the same workload bucket again, it can reuse the cached kernel instead of repeating the full autotuning process.

---

### 5. Hot-Swap Mechanism

The hot-swap mechanism safely replaces the active kernel used by the serving loop.

The key rule is:

```text
Never replace the active kernel in the middle of a decode step.
```

Forge uses a double-buffered design:

```text
active kernel: currently used by the serving loop
staging kernel: compiled and benchmarked by the autotuner
```

Once the staging kernel is validated, the hot-swap mechanism updates the active kernel pointer at the next decode-step boundary.

This avoids unsafe mid-step replacement and keeps serving behavior stable.

---

## Runtime Flow

A typical Forge runtime sequence looks like this:

### 1. Startup

Forge initializes the serving loop with a default kernel.

```text
active_kernel = default_decode_kernel
kernel_cache = load_existing_cache_or_empty()
```

### 2. Profiling Begins

The profiler observes recent decode steps.

```text
step 1: batch=8, seq_len=900
step 2: batch=10, seq_len=950
step 3: batch=9, seq_len=1000
```

It summarizes the workload into buckets.

```text
batch_size_bucket = 8-16
seq_len_bucket = 512-1024
```

### 3. Autotuning Is Triggered

The profiler sends the workload specification to the autotuner.

```text
workload_spec = {
  batch_size_bucket: 8-16,
  seq_len_bucket: 512-1024,
  model_config_hash: ...
}
```

### 4. Candidate Kernels Are Compiled

The autotuner tries candidate configurations.

```text
candidate 1: BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3
candidate 2: BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3
candidate 3: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=4
...
```

### 5. Candidates Are Benchmarked

Each candidate is benchmarked for a short window.

```text
candidate 1: 11800 tokens/sec
candidate 2: 12650 tokens/sec
candidate 3: 12120 tokens/sec
```

### 6. Best Kernel Is Selected

The autotuner selects the best validated candidate.

```text
winner = candidate 2
```

### 7. Kernel Is Cached

The result is stored in the cache.

```text
(8-16, 512-1024, model_hash) -> candidate 2 kernel
```

### 8. Kernel Is Hot-Swapped

At the next decode-step boundary, Forge switches to the new kernel.

```text
active_kernel = candidate_2_kernel
```

### 9. Workload Shift Occurs

If the workload later changes:

```text
old workload: batch=8-16, seq_len=512-1024
new workload: batch=1-4, seq_len=4096-8192
```

The profiler triggers a new tuning cycle.

---

## Triton Kernel Template

Forge's decode kernel is written as a Triton template. The kernel is not fixed at write time. It is specialized at compile time using meta-parameters.

Key parameters:

```text
BLOCK_M: tile size along query/batch dimension
BLOCK_N: tile size along sequence/KV dimension
BLOCK_K: tile size along head dimension
num_warps: number of warps assigned per program
num_stages: software pipeline depth
```

A simplified conceptual launch might look like:

```python
decode_attention_kernel[grid](
    q,
    k_cache,
    v_cache,
    output,
    block_table,
    batch_size,
    seq_len,
    BLOCK_M=16,
    BLOCK_N=64,
    BLOCK_K=64,
    num_warps=4,
    num_stages=3,
)
```

Each different parameter set causes Triton to produce a different compiled kernel variant.

---

## Profile-Guided Optimization

Forge applies profile-guided optimization to GPU kernels at serving time.

Traditional profile-guided optimization works like this:

```text
1. Run the program.
2. Collect profile data.
3. Recompile the program using profile-guided decisions.
4. Run the optimized version.
```

Forge applies the same idea to LLM decode kernels:

```text
1. Run the serving loop.
2. Collect batch size and KV cache length profiles.
3. Recompile Triton kernels for the observed workload.
4. Hot-swap the optimized kernel into the serving loop.
```

The profile guides the kernel configuration.

---

## Autotuning Strategy

A naive grid search over all kernel configurations would be too slow. Forge instead performs a short search.

The proposed search uses:

```text
8 to 16 candidate configurations
50 decode steps per benchmark
Pareto-optimal selection
nearest-neighbor cache warm-starting
Bayesian optimization or lightweight search
```

### Candidate Evaluation

Each candidate is evaluated using metrics such as:

```text
tokens/sec
decode latency
p50 latency
p95 latency
GPU utilization
benchmark stability
```

### Warm-Starting

If a nearby workload has already been tuned, Forge can start from the cached configuration.

Example:

```text
cached workload: batch=8-16, seq_len=1024-2048
new workload: batch=8-16, seq_len=2048-4096
```

The previous best configuration may be a strong starting point.

---

## Kernel Cache

The kernel cache exists to avoid repeated compilation.

Cache key:

```text
(batch_size_bucket, seq_len_bucket, model_config_hash)
```

Cache value:

```text
compiled kernel
kernel parameters
benchmark result
validation metadata
```

The cache helps in two cases:

1. **Repeated workload**
   - Same bucket appears again.
   - Forge reuses the cached kernel.

2. **Nearby workload**
   - Similar bucket appears.
   - Forge uses nearest-neighbor warm-starting.

This makes Forge faster over time because the system accumulates knowledge about which kernel configurations work well for different workload regions.

---

## Hot-Swap Mechanism

The hot-swap mechanism ensures that kernel replacement is safe.

A bad swap would look like this:

```text
decode step starts with kernel A
system replaces kernel A with kernel B mid-step
execution state becomes inconsistent
```

Forge avoids this by swapping only at step boundaries.

Safe swap:

```text
decode step N runs with kernel A
decode step N finishes
hot-swap checks validated staging kernel
active kernel pointer updated to kernel B
decode step N+1 runs with kernel B
```

The mechanism uses the concept of:

```text
active kernel pointer
staging kernel
validation flag
step-boundary swap
double buffering
```

---

## Benchmarking Plan

Forge should be evaluated under both stable and shifting workloads.

### Stable Workload Benchmark

Goal:

```text
Show that a workload-specialized kernel is faster than a static generic kernel.
```

Example workload:

```text
batch size: 8-16
seq_len: 1024-2048
duration: fixed benchmark window
```

Metrics:

```text
tokens/sec
average decode latency
p50 latency
p95 latency
GPU utilization
```

### Shifting Workload Benchmark

Goal:

```text
Show that Forge adapts when the workload changes.
```

Example:

```text
0-60 seconds:
  batch size: high
  seq_len: short

60-120 seconds:
  batch size: low
  seq_len: long

120-180 seconds:
  batch size: medium
  seq_len: medium
```

Expected graph:

```text
x-axis: time
y-axis: throughput

static baseline: mostly flat
Forge: adapts upward after each workload shift
```

### Cache Benchmark

Goal:

```text
Show that cached kernels reduce retuning overhead.
```

Metrics:

```text
cache hit rate
time to optimized kernel
number of trials required
compile overhead
benchmark overhead
```

---

## Expected Results

The proposal's benchmark story is:

```text
Under stable workloads:
Forge's tuned kernel is 15 to 30 percent faster than a generic static kernel.

Under shifting workloads:
Forge detects workload changes and tracks the optimal configuration within approximately 30 seconds.
```

The main result should be a time-series chart showing throughput over time.

The baseline remains mostly flat because it uses a static kernel configuration. Forge improves after each workload shift because it recompiles and hot-swaps a better kernel.

---

## Why This Is a Compiler Project

Forge is a compiler project because it takes a parameterized program and emits specialized lower-level executable code for a target.

Compiler analogy:

```text
source program
        ↓
target specification
        ↓
optimized machine code
```

Forge analogy:

```text
Triton kernel template
        ↓
batch size + sequence length + hardware profile
        ↓
optimized GPU kernel
```

The important compiler concepts are:

```text
JIT compilation
profile-guided optimization
target-specific code generation
compile-time constants
runtime specialization
compiled code caching
safe code replacement
```

The project uses compiler ideas in a live LLM serving system.

---

## Why This Is Different From AgentIR

Forge is not an agent orchestration or routing project.

The distinction is:

```text
AgentIR: control plane
Forge: data plane compiler
```

AgentIR focuses on decisions such as:

```text
where work should go
how work should be scheduled
which route should be chosen
how agents should coordinate
```

Forge focuses on:

```text
making the actual GPU execution faster
optimizing decode kernels
compiling specialized kernel variants
hot-swapping runtime code
```

The two systems are orthogonal and composable. AgentIR can decide where work should be sent. Forge can make the execution of that work faster.

---

## Tech Stack

### Core

```text
Python
PyTorch
Triton
CUDA-compatible NVIDIA GPU
NumPy
pandas
matplotlib
```

### GPU and Profiling

```text
NVIDIA CUDA runtime
nvidia-smi
Nsight Systems
Nsight Compute
PyTorch profiler
Triton debugging/profiling tools
```

### Autotuning

```text
Optuna
scikit-optimize
custom lightweight Bayesian search
nearest-neighbor cache lookup
```

### Systems

```text
Python threading
thread-safe queues
locks or atomic-style kernel handle
JSON or SQLite cache metadata
CSV/JSON benchmark logs
```

### Optional

```text
vLLM for baseline comparison
Docker for reproducible environment
FastAPI only if exposing a demo serving API
```

---

## Repository Structure

A possible repository layout:

```text
forge/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── search_space.yaml
│   └── benchmark_workloads.yaml
├── forge/
│   ├── __init__.py
│   ├── serving/
│   │   ├── loop.py
│   │   ├── scheduler.py
│   │   └── kernel_handle.py
│   ├── profiler/
│   │   ├── profiler.py
│   │   ├── workload_spec.py
│   │   └── shift_detector.py
│   ├── autotuner/
│   │   ├── tuner.py
│   │   ├── search.py
│   │   ├── benchmark.py
│   │   └── candidate.py
│   ├── cache/
│   │   ├── kernel_cache.py
│   │   └── metadata_store.py
│   ├── kernels/
│   │   ├── decode_attention.py
│   │   └── kernel_configs.py
│   ├── hotswap/
│   │   ├── swap_manager.py
│   │   └── validation.py
│   └── utils/
│       ├── timing.py
│       ├── logging.py
│       └── shapes.py
├── benchmarks/
│   ├── stable_workload.py
│   ├── shifting_workload.py
│   ├── compare_static.py
│   └── plot_results.py
├── scripts/
│   ├── run_server.py
│   ├── run_autotune.py
│   └── run_benchmarks.py
└── results/
    ├── raw/
    └── plots/
```

---

## Setup

### 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
triton
numpy
pandas
matplotlib
optuna
pyyaml
```

### 3. Verify GPU Access

```bash
nvidia-smi
```

### 4. Verify PyTorch CUDA

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 5. Verify Triton

```python
import triton
import triton.language as tl

print(triton.__version__)
```

---

## Usage

### Run a Static Baseline

```bash
python benchmarks/compare_static.py --config configs/default.yaml
```

### Run Forge With Profiling and Autotuning

```bash
python scripts/run_server.py --config configs/default.yaml
```

### Run a Stable Workload Benchmark

```bash
python benchmarks/stable_workload.py --config configs/benchmark_workloads.yaml
```

### Run a Shifting Workload Benchmark

```bash
python benchmarks/shifting_workload.py --config configs/benchmark_workloads.yaml
```

### Plot Results

```bash
python benchmarks/plot_results.py --input results/raw/shifting_workload.json
```

---

## Roadmap

### Phase 1: Kernel Template

Build the Triton decode attention kernel template parameterized by:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

Validate correctness against a PyTorch reference implementation.

### Phase 2: Benchmark Harness

Build a benchmarking harness that measures:

```text
tokens/sec
decode latency
p50 latency
p95 latency
GPU utilization
```

### Phase 3: Profiler

Implement rolling-window tracking for:

```text
batch size distribution
sequence length distribution
KV cache length distribution
```

### Phase 4: Autotuner

Implement candidate search over 8 to 16 configurations.

Add:

```text
candidate generation
benchmark execution
Pareto-optimal selection
nearest-neighbor warm-starting
```

### Phase 5: Kernel Cache

Implement cache lookup keyed by:

```text
(batch_size_bucket, seq_len_bucket, model_config_hash)
```

### Phase 6: Hot-Swap

Implement double-buffered kernel replacement.

Ensure swaps happen only at decode-step boundaries.

### Phase 7: End-to-End Benchmark

Run stable and shifting workload experiments.

Produce the final throughput-over-time graph.

---

## Resume and Interview Framing

### One-Line Description

```text
Built Forge, a profile-guided JIT compiler for LLM decode kernels that profiles live batch and KV-cache distributions, autotunes Triton kernel parameters, caches compiled variants, and hot-swaps optimized kernels at decode-step boundaries.
```

### Technical Explanation

```text
Forge takes a parameterized Triton decode kernel template and a target specification derived from live serving profiles, then emits an optimized GPU kernel for that workload. It applies compiler-style profile-guided optimization to LLM inference by continuously measuring batch size and KV cache length distributions, recompiling kernels when the workload shifts, and safely replacing active kernels through a step-boundary hot-swap mechanism.
```

### Why It Matters

```text
LLM decode performance is often memory-bandwidth-bound, and the best GPU kernel configuration depends on runtime workload shape. Forge adapts kernel tiling, warp count, and pipeline depth dynamically instead of relying on one static configuration.
```

### Concepts Demonstrated

```text
ML systems
GPU kernel programming
Triton JIT compilation
LLM inference optimization
KV cache behavior
profile-guided optimization
runtime specialization
autotuning
hot-swapping
performance benchmarking
```

---

## Summary

Forge is a data-plane compiler for LLM inference. It improves decode performance by observing the live workload, compiling specialized Triton kernels for the current operating point, caching compiled variants, and safely hot-swapping better kernels into the serving loop.

The project combines compiler ideas, GPU systems, and LLM serving infrastructure:

```text
profile-guided optimization
+
JIT compilation
+
Triton GPU kernels
+
LLM decode workloads
+
runtime hot-swapping
```

The result is a high-signal MLSys project with a concrete bottleneck, a concrete optimization mechanism, and a measurable benchmark story.
