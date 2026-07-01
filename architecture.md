# Forge Architecture

Forge is a prototype for speeding up LLM text generation by tuning the GPU
kernels used during decode.

This document explains the project from the ground up. It intentionally uses
plain language first, then maps each idea to the code in this repository.

## 1. The Basic LLM Flow

An LLM does not directly calculate with words. A tokenizer converts the prompt
into numbers, and an LLM-serving program coordinates the remaining work.

```text
User prompt
    |
    v
Tokenizer
    |
    v
Token IDs (one number for each text piece)
    |
    v
Tensors (arrays containing numbers)
    |
    v
LLM-serving program launches many GPU kernels
    |
    v
GPU kernels calculate with tensors and model weights
    |
    v
Final tensor containing one score per possible next token
    |
    v
LLM-serving program chooses one token using those scores
    |
    v
Text output
```

Important terms:

- **Tokenizer**: turns text into token IDs.
- **Token ID**: a number representing a piece of text.
- **Tensor**: an array of numbers used by the model.
- **Model weights**: billions of learned numbers stored by the LLM.
- **Model structure**: instructions that define which calculations must happen.
- **Model**: the model weights plus the model structure.
- **LLM-serving program**: software such as PyTorch or vLLM that coordinates
  tokenization, tensors, GPU kernel launches, and token selection.
- **GPU kernel**: a small compiled program that runs on the GPU.

### What "model calculations" means

Model calculations are calculations performed on tensors.

```text
input tensors
    +
model weights
    |
    v
GPU kernels perform calculations
    |
    v
new tensors
```

These calculations repeatedly combine and transform arrays of numbers. Many GPU
kernels are needed for one pass through the LLM.

The model is not a program independently controlling the computer. The
LLM-serving program follows the model structure and launches the required GPU
kernels. The GPU kernels perform the actual tensor calculations on GPU hardware.

The responsibilities are:

```text
Tokenizer:
turns text into token IDs

LLM-serving program:
controls the overall process and launches GPU kernels

GPU:
physical hardware that performs calculations

GPU kernels:
small programs that perform specific tensor calculations

Model weights:
learned numbers used by those calculations
```

## 2. Input, Output, And Next-Token Prediction

The LLM generates its response one token at a time.

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

It is not predicting a missing token inside the original prompt. During response
generation, it predicts the next token in the output using both:

```text
original prompt + output generated so far
```

### What are next-token scores?

Suppose the tokenizer knows 50,000 possible tokens. After the GPU kernels finish
the model calculations, the final tensor contains roughly one score for each
possible token.

```text
Possible token     Score

" the"             12.8
" a"               10.1
"."                 8.4
" running"          2.2
" banana"          -3.5
```

A higher score means that the model considers that token a better continuation
of the current text.

The GPU kernels produce this final score tensor by calculating with:

```text
original prompt information
+
generated output information
+
model weights
```

The LLM-serving program then uses the scores to select one token. It may select
the highest-scoring token, or randomly select while giving higher-scoring tokens
a greater chance.

The complete repeated loop is:

```text
original prompt + output generated so far
    |
    v
LLM-serving program launches GPU kernels
    |
    v
GPU kernels perform tensor calculations
    |
    v
final tensor contains possible-token scores
    |
    v
LLM-serving program chooses one token
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

Prefill processes all tokens in the user's prompt.

```text
1. Tokenizer converts the entire prompt into token IDs.
2. Token IDs are turned into tensors.
3. The LLM-serving program launches GPU kernels.
4. GPU kernels process all prompt tokens.
5. KV-cache information is created for all prompt tokens.
6. Scores for the first output token are produced.
7. The LLM-serving program chooses the first output token.
```

Yes: the KV cache is initially filled during prefill.

### Decode

Decode generates the remaining answer one token at a time.

```text
1. Process the newest generated token.
2. Read previous-token information from the KV cache.
3. Launch GPU kernels that perform tensor calculations.
4. Produce scores for the next possible token.
5. Choose one token.
6. Add KV-cache information for that new token.
7. Repeat.
```

The KV cache is therefore:

```text
initially filled with prompt information during prefill
+
extended with one new token during every decode step
```

Forge focuses on the decode phase.

## 4. The KV Cache

During generation, the model needs information from previous tokens. Recomputing
all previous-token information every time would be slow, so the LLM-serving
program keeps some of it in GPU memory as the KV cache.

```text
Prefill processes prompt tokens
    |
    v
KV cache is initially filled
    |
    v
Decode generates one token
    |
    v
KV cache is extended for that token
    |
    v
Repeat for every generated token
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
next token. They also produce the new information that gets added to the cache.

```text
current token information
        |
        v
decode attention GPU kernel <---- existing KV cache
        |
        +----> tensor used for next-token prediction
        |
        +----> new token information added to KV cache
```

As the prompt and generated answer get longer, the KV cache gets larger. Reading
and processing that saved data can become a major cost during decode.

## 5. Where GPU Kernels Fit

A GPU is the hardware. A GPU kernel is a small program running on that hardware.

```text
LLM-serving program
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

The model does not launch kernels. The exact relationship is:

```text
model structure says which calculations are required
        |
        v
LLM-serving program follows those instructions
        |
        v
LLM-serving program launches GPU kernels
        |
        v
GPU kernels calculate with tensors and model weights
```

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
                             | LLM-serving program  |
                             | tokenize and control |
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
                             +----------+-----------+
                                        |
                                        v
                             +----------------------+
                             | Serving program      |
                             | chooses next token   |
                             +----------------------+


     Decode-step metrics
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

### 9.1 Decode Workload Profiler

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
real LLM-serving program integration
real KV-cache tensors
real GPU benchmarking
```

The current code is useful because it proves the architecture before adding GPU
complexity.

## 12. Final Mental Model

Keep this version in your head:

```text
LLM generation means predicting one next token at a time.

The model is learned weights plus instructions for tensor calculations.

The LLM-serving program follows those instructions and launches GPU kernels.

GPU kernels perform calculations on tensors using the model weights.

Prefill processes the prompt, fills the initial KV cache, and produces scores
for the first output token.

Decode produces later output tokens one at a time. During every decode step,
GPU kernels read the KV cache and add information for the newly generated token.

The LLM-serving program chooses each token from the final score tensor.

Forge watches the decode workload and chooses better kernel settings.

The better kernel is cached and safely swapped into the serving loop.
```
