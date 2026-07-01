# Forge Architecture

Forge is a prototype for speeding up LLM text generation by tuning the GPU
kernels used during decode.

This document explains the project from the ground up. It intentionally uses
plain language first, then maps each idea to the code in this repository.

## 1. The Basic LLM Flow

An LLM does not directly work with words. The prompt is converted into numbers,
then the runtime executes the model's math on those numbers.

```text
User prompt
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Tensors
    |
    v
Runtime executes model calculations
    |
    v
Scores for possible next tokens
    |
    v
Chosen next token
    |
    v
Text output
```

Important terms:

- **Tokenizer**: turns text into token IDs.
- **Token ID**: a number representing a piece of text.
- **Tensor**: an array of numbers used by the model.
- **Model**: the learned weights plus the structure of the calculations.
- **Runtime**: the software that executes the model.
- **GPU kernel**: a small compiled program that runs on the GPU.

The model itself does not launch GPU kernels. The runtime executes the model,
and the runtime launches GPU kernels to do the heavy tensor calculations.

## 2. Input, Output, And Next-Token Prediction

The model generates text one token at a time.

```text
Prompt:
"Explain gravity simply."

Output so far:
"Gravity is"

Current context:
"Explain gravity simply. Gravity is"

Next-token prediction:
" the"
```

The model is not usually predicting the next token inside the original input.
During response generation, it predicts the next token in the output, using both:

```text
original prompt + output generated so far
```

The repeated loop looks like this:

```text
context so far
    |
    v
runtime executes model
    |
    v
scores for possible next tokens
    |
    v
choose one token
    |
    v
append token to output
    |
    v
repeat
```

## 3. Prefill And Decode

LLM inference has two main phases.

```text
                      LLM Inference
                           |
             +-------------+-------------+
             |                           |
             v                           v
          Prefill                      Decode
   read the user prompt        generate output tokens
   process many tokens         one token at a time
```

### Prefill

Prefill processes the prompt.

```text
User prompt:
"Explain gravity simply."

Prefill:
process all prompt tokens
build internal saved information
```

### Decode

Decode generates the answer one token at a time.

```text
Step 1: generate "Gravity"
Step 2: generate " is"
Step 3: generate " the"
Step 4: generate " force"
```

Forge focuses on the decode phase.

## 4. The KV Cache

During generation, the model needs information from previous tokens. Recomputing
all previous-token information every time would be slow, so the runtime stores
some of it in the KV cache.

```text
Previous tokens
    |
    v
Saved internal number data
    |
    v
KV cache
```

KV means:

```text
K = keys
V = values
```

For this project, the beginner-friendly meaning is enough:

```text
KV cache = saved previous-token information
```

During each decode step, GPU kernels read from the KV cache to help predict the
next token.

```text
current token information
        |
        v
decode attention kernel <---- KV cache from previous tokens
        |
        v
new tensor used for next-token prediction
```

As the prompt and generated answer get longer, the KV cache gets larger. Reading
and processing that saved data can become a major cost during decode.

## 5. Where GPU Kernels Fit

A GPU is the hardware. A GPU kernel is a small program running on that hardware.

```text
Runtime
  |
  | launches
  v
GPU kernel
  |
  | runs on
  v
GPU hardware
```

One user request does not equal one GPU kernel. One request can cause many GPU
kernels to run.

```text
one user request
    |
    v
many decode steps
    |
    v
many GPU kernel launches per step
```

Forge is interested in decode attention kernels, especially the ones that use
the KV cache.

## 6. What Forge Is Trying To Do

Different serving situations can need different GPU-kernel settings.

Examples:

```text
Situation A:
many short requests at once

Situation B:
few long requests

Situation C:
mixed request lengths
```

The same kernel configuration may not be fastest for all situations.

Forge proposes this loop:

```text
observe the current decode workload
    |
    v
summarize it as a workload spec
    |
    v
try several kernel configurations
    |
    v
benchmark them
    |
    v
pick the best one
    |
    v
cache the result
    |
    v
swap the better kernel into the serving loop
```

This is called profile-guided JIT compilation:

- **Profile-guided**: runtime measurements guide the optimization.
- **JIT**: just-in-time, meaning compilation/tuning happens while the system is running.

## 7. Full System Diagram

```text
                             +----------------------+
                             | User request         |
                             +----------+-----------+
                                        |
                                        v
                             +----------------------+
                             | LLM server/runtime   |
                             | tokenize, batch, run |
                             +----------+-----------+
                                        |
                                        v
                             +----------------------+
                             | Serving loop         |
                             | decode step N        |
                             +----------+-----------+
                                        |
                           launches active kernel
                                        |
                                        v
                             +----------------------+
                             | Decode GPU kernel    |
                             | uses KV cache        |
                             +----------+-----------+
                                        |
                                        v
                             +----------------------+
                             | Next-token scores    |
                             +----------------------+


     Runtime metrics
          |
          v
+-------------------+     workload spec      +-------------------+
| Runtime profiler  | ---------------------> | Autotuner         |
| batch, seq length |                        | tries candidates  |
+-------------------+                        +---------+---------+
                                                     |
                                                     v
                                           +-------------------+
                                           | Kernel cache      |
                                           | remember winners  |
                                           +---------+---------+
                                                     |
                                                     v
                                           +-------------------+
                                           | Hot-swap manager  |
                                           | stage new kernel  |
                                           +---------+---------+
                                                     |
                                      swap at decode boundary
                                                     |
                                                     v
                                           +-------------------+
                                           | Active kernel     |
                                           +-------------------+
```

## 8. Repository Components

The current implementation is a CPU-side simulation of the Forge control plane.
It does not yet contain real Triton/CUDA kernels.

```text
forge/
├── profiler/
│   ├── profiler.py
│   └── workload_spec.py
├── kernels/
│   ├── decode_attention.py
│   └── kernel_configs.py
├── autotuner/
│   ├── benchmark.py
│   ├── candidate.py
│   ├── search.py
│   └── tuner.py
├── cache/
│   └── kernel_cache.py
├── hotswap/
│   └── swap_manager.py
├── serving/
│   └── loop.py
└── utils/
    └── buckets.py
```

## 9. Component Breakdown

### 9.1 Runtime Profiler

Code:

```text
forge/profiler/profiler.py
forge/profiler/workload_spec.py
```

The profiler records recent decode steps.

It tracks:

```text
batch size
sequence length
latency
```

Then it summarizes the recent workload into buckets.

Example:

```text
batch size = 8
sequence length = 900

becomes:

batch bucket = 5-8
sequence bucket = 513-1024
```

That summary is called a `WorkloadSpec`.

```text
decode-step metrics
        |
        v
RuntimeProfiler
        |
        v
WorkloadSpec
```

### 9.2 Workload Spec

Code:

```text
forge/profiler/workload_spec.py
```

A workload spec is a compact description of the current workload.

It contains:

```text
batch_size_bucket
seq_len_bucket
model_config_hash
window_size
```

It also creates a cache key like:

```text
toy_llm_head128|5-8|513-1024
```

This key lets Forge remember which kernel worked best for that kind of workload.

### 9.3 Kernel Configs

Code:

```text
forge/kernels/kernel_configs.py
```

A kernel config describes compile-time settings for a kernel variant.

The current prototype uses settings like:

```text
block_m
block_n
block_k
num_warps
num_stages
```

For now, treat these as tuning knobs. Different knob values can make the kernel
faster or slower for different workload shapes.

```text
KernelConfig A
KernelConfig B
KernelConfig C
        |
        v
Autotuner tests them
```

### 9.4 Decode Attention Kernel

Code:

```text
forge/kernels/decode_attention.py
```

In a future version, this will become a real Triton decode attention kernel.

Right now, it is a simulation. It pretends to launch a kernel and returns
deterministic timing numbers.

```text
batch size + sequence length + kernel config
        |
        v
simulated decode kernel
        |
        v
latency and tokens/sec
```

This lets the rest of the architecture work before real GPU code is added.

### 9.5 Autotuner

Code:

```text
forge/autotuner/tuner.py
forge/autotuner/search.py
forge/autotuner/benchmark.py
forge/autotuner/candidate.py
```

The autotuner receives a workload spec, tests candidate configs, and chooses the
fastest one.

```text
WorkloadSpec
     |
     v
candidate configs
     |
     v
benchmark each candidate
     |
     v
choose highest tokens/sec
     |
     v
TuningResult
```

The result contains:

```text
best config
latency
tokens/sec
validated flag
```

### 9.6 Kernel Cache

Code:

```text
forge/cache/kernel_cache.py
```

The cache stores the best result for a workload spec.

```text
WorkloadSpec cache key
        |
        v
best KernelConfig
```

This avoids retuning the same workload again.

Example:

```text
toy_llm_head128|5-8|513-1024
    -> block_m=32, block_n=128, block_k=128, num_warps=8, num_stages=4
```

The current prototype stores this as JSON in:

```text
results/raw/kernel_cache.json
```

### 9.7 Hot-Swap Manager

Code:

```text
forge/hotswap/swap_manager.py
```

The hot-swap manager keeps two kernel slots:

```text
active kernel  = currently used by decode
staging kernel = validated replacement waiting to be swapped in
```

The key safety rule:

```text
do not swap kernels in the middle of a decode step
```

The safe swap flow:

```text
decode step N uses old kernel
        |
        v
step finishes
        |
        v
swap manager checks staging kernel
        |
        v
decode step N+1 uses new kernel
```

### 9.8 Serving Loop

Code:

```text
forge/serving/loop.py
```

The serving loop connects everything.

For each decode step, it:

```text
1. launches the active kernel
2. records metrics in the profiler
3. periodically asks for a tuned kernel
4. checks the cache first
5. runs the autotuner on cache miss
6. stages the tuned kernel
7. swaps at the step boundary
```

Diagram:

```text
decode_step(batch_size, seq_len)
        |
        v
active kernel launch
        |
        v
record profiler metrics
        |
        v
time to tune?
        |
        +---- no ----> return result
        |
       yes
        |
        v
build workload spec
        |
        v
cache hit?
        |
        +---- yes ---> stage cached kernel
        |
        +---- no ----> autotune, cache result, stage kernel
        |
        v
swap at step boundary
        |
        v
return result
```

## 10. End-To-End Prototype Flow

When you run:

```bash
python3 scripts/run_server.py
```

the prototype does this:

```text
create profiler
create cache
create autotuner
create serving loop with default kernel
run simulated decode steps
periodically tune a better kernel
swap tuned kernel into active slot
print latency and throughput
```

Expected behavior:

```text
early steps use:
decode_attention[default]

after tuning:
decode_attention[toy_llm_head128|...]
```

That shows the control-plane behavior working:

```text
profile -> tune -> cache -> hot-swap
```

## 11. What Is Real Now Vs Future Work

Real in the current prototype:

```text
project structure
workload specs
rolling profiler
candidate search
simulated benchmarking
JSON cache
hot-swap manager
serving-loop simulation
tests
```

Not real yet:

```text
actual Triton kernel
actual CUDA/GPU execution
real LLM inference runtime
real KV-cache tensors
real GPU benchmarking
```

The current code is useful because it proves the architecture before adding GPU
complexity.

## 12. Final Mental Model

Keep this version in your head:

```text
LLM generation means predicting one next token at a time.

The runtime executes the model using tensors.

The GPU runs small programs called kernels to process those tensors.

During decode, some kernels repeatedly use the KV cache.

Forge watches the decode workload and chooses better kernel settings.

The better kernel is cached and safely swapped into the serving loop.
```

