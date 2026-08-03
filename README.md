# glamin-experiment-1

An experimental C++ inference runtime that integrates a language model with
Glamin executable geometry through a stable C ABI.

The first milestone is a cross-language generation-swap demonstration: a C++
process queries Glamin, atomically changes the mounted geometry generation, and
observes a deterministic behavior change without restarting.

The model milestone now runs that boundary inside a pinned llama.cpp graph. A
Qwen3 hidden state traverses Glamin at one fixed layer, receives a
generation-qualified residual, and continues through the remaining layers.

## Documentation

- [Documentation index](docs/README.md)
- [Project overview](docs/project-overview.md)
- [Fixed model hook](docs/model-hook.md)
- [Persistent hook artifact](docs/hook-artifact.md)
- [Activation-derived associative memory](docs/activation-memory.md)
- [Experiment definition](EXPERIMENT.md)

## Build the Reference Runtime

The reference executable proves host-side immutable generation pinning and live
activation semantics. The project also builds a real C++ RAII wrapper around
Glamin's Fortran worker runtime, flat indexes, immutable generations, and ABI
version 4 persistent compiler-artifact loading with selected-space contract
validation.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/gx1_generation_swap_demo
./build/gx1_glamin_runtime_smoke
./build/gx1_glamin_generation_swap_demo
```

To exercise persistent loading:

```bash
make -C third_party/glamin spec-compile spec-embed
./build/gx1_glamin_persistent_generation_demo \
  third_party/glamin/build/specs geometry.auth
```

To run the real-model hidden-state swap with the available Qwen3 4B model:

```bash
./build/gx1_llama_hidden_state_demo \
  /home/lynn/.qwench/models/Qwen_Qwen3-4B-Q4_K_M.gguf
```

The demonstration constructs two complete hash-bound hook artifacts, keeps the
weights loaded while activating A, B, and A again, and prints each artifact
identity. It verifies that B changes the final logits and rollback exactly
reproduces A.

To derive eight associations from teacher activations, mount them through
Glamin, and test construction views, held-out natural-question recall, and hard
negatives:

```bash
./build/gx1_llama_activation_memory_experiment \
  /home/lynn/.qwench/models/Qwen_Qwen3-4B-Q4_K_M.gguf
```

The default hook is the penultimate transformer layer, whose tensor still
contains every prompt row. The hook projects all token rows, selects the
globally nearest Glamin result, and applies its residual only to the final
answer-position row. A second argument can probe a different zero-based layer;
see the
[activation-memory report](docs/activation-memory.md) for the precise boundary.
