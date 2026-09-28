# AXEN Polyglot Architecture

This directory hosts non-Python language components of AXEN. Each language performs **one specific, honest job**.

## Core Polyglot Rules

1. **Python Core Independence**:
   - Python runs the core AI assistant, training, and model serving.
   - The entire system **must remain fully functional** even if the entire `polyglot/` folder is deleted.

2. **Per-Language Structure**:
   Each language implementation lives in its own subdirectory:
   ```
   polyglot/<language>/
     README.md      # Purpose, compiler/runtime prerequisites, setup instructions
     test.*         # Automated test or verification script
     run.*          # Execution script or entrypoint
   ```

3. **Strict Inter-Process Communication (IPC)**:
   Languages communicate with the Python core exclusively via standard, well-defined protocols:
   - **HTTP REST / JSON**
   - **stdin / stdout JSON streaming**
   - **gRPC / Protocol Buffers**
   - **File-based data exchange** (e.g. SQLite, binary shards, memory-mapped arrays)

4. **Job Classification**:
   Every language is assigned an honest classification:
   - `BUILD`: Production-ready working code (e.g., C++ llama.cpp inference, Rust data cleaner, Go gateway).
   - `DEMO`: Educational learning implementation or benchmark (e.g., Julia training dynamics, MATLAB attention visualizer, Prolog logic solver).
   - `EXPLAIN`: Written technical analysis / documentation when a dedicated runtime has no practical justification.
   - `SKIP`: Explicitly skipped when not applicable (with a clear one-line rationale).

5. **Prerequisites & Toolchains**:
   - Compilers and toolchains will only be installed after explicit permission.
   - For tools that run poorly on native Windows (e.g., Mojo, Triton), WSL2 is recommended.
