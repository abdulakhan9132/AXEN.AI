"""
axen/tools/computer/assembly.py
==================================
Educational assembly language module for AXEN.

IMPORTANT ARCHITECTURAL PRINCIPLE:
  Assembly language is NOT universal. Instructions, registers,
  calling conventions, and encodings differ across architectures.

  AXEN clearly distinguishes:
    - x86-64 (Intel/AMD — most common on desktops/servers)
    - ARM64 / AArch64 (Apple Silicon, mobile, embedded)
    - RISC-V (open ISA, growing in embedded/server)

This module:
  - Documents common instruction concepts (educational)
  - Shows architecture-specific register names and conventions
  - Explains the compilation pipeline (C → Assembly → Machine Code → Binary → CPU)
  - Demonstrates the fetch-decode-execute cycle
  - Provides an educational CPU abstraction model

This module does NOT:
  - Assemble real machine code
  - Emulate a real CPU
  - Fabricate instruction encodings without proper context
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "x86-64": {
        "full_name": "x86-64 (AMD64 / Intel 64)",
        "word_size": 64,
        "endianness": "Little-endian",
        "registers": {
            "general_purpose": ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI",
                                  "RSP", "RBP", "R8", "R9", "R10", "R11",
                                  "R12", "R13", "R14", "R15"],
            "instruction_pointer": "RIP",
            "flags": "RFLAGS",
            "stack_pointer": "RSP",
            "frame_pointer": "RBP",
        },
        "calling_convention": "System V AMD64 ABI (Linux/macOS): first 6 args in RDI, RSI, RDX, RCX, R8, R9; return in RAX",
        "instruction_set": "CISC (Complex Instruction Set Computing)",
        "common_use": "Desktop, laptop, server CPUs (Intel Core, AMD Ryzen/EPYC)",
    },
    "arm64": {
        "full_name": "ARM64 / AArch64",
        "word_size": 64,
        "endianness": "Little-endian (default, can be big-endian)",
        "registers": {
            "general_purpose": [f"X{i}" for i in range(31)] + ["XZR"],
            "instruction_pointer": "PC (Program Counter)",
            "flags": "NZCV (Negative, Zero, Carry, Overflow)",
            "stack_pointer": "SP",
            "link_register": "X30 (LR)",
            "frame_pointer": "X29 (FP)",
        },
        "calling_convention": "AAPCS64: first 8 args in X0–X7; return in X0",
        "instruction_set": "RISC (Reduced Instruction Set Computing) — fixed 32-bit instruction width",
        "common_use": "Apple Silicon (M1/M2/M3), Android devices, AWS Graviton, Raspberry Pi 4+",
    },
    "risc-v": {
        "full_name": "RISC-V (RV64)",
        "word_size": 64,
        "endianness": "Little-endian (standard)",
        "registers": {
            "general_purpose": (
                ["x0 (zero)", "x1 (ra)", "x2 (sp)", "x3 (gp)", "x4 (tp)"] +
                [f"x{i} (t{i-5})" for i in range(5, 8)] +
                [f"x{i} (s{i-8})" for i in range(8, 10)] +
                [f"x{i} (a{i-10})" for i in range(10, 18)] +
                [f"x{i} (s{i-16})" for i in range(18, 28)] +
                [f"x{i} (t{i-25})" for i in range(28, 32)]
            ),
            "instruction_pointer": "PC (Program Counter)",
            "stack_pointer": "x2 (sp)",
        },
        "calling_convention": "RISC-V ABI: args in a0–a7 (x10–x17); return in a0/a1",
        "instruction_set": "RISC — 32-bit base instructions, modular extensions (M, F, D, C, ...)",
        "common_use": "Open-source hardware, SiFive CPUs, embedded systems, research",
    },
}


# ---------------------------------------------------------------------------
# Common instruction concepts (architecture-agnostic conceptual level)
# ---------------------------------------------------------------------------

INSTRUCTION_CONCEPTS = {
    "MOV": {
        "concept": "Move/Load — copy a value from source to destination.",
        "x86_64_example": "mov rax, 42       ; load 42 into RAX",
        "arm64_example":   "mov x0, #42       ; load 42 into X0",
        "riscv_example":   "li a0, 42          ; load immediate 42 into a0",
        "note": "Does not erase the source.",
    },
    "ADD": {
        "concept": "Add — adds two values, stores result.",
        "x86_64_example": "add rax, rbx      ; RAX = RAX + RBX",
        "arm64_example":   "add x0, x0, x1    ; X0 = X0 + X1",
        "riscv_example":   "add a0, a0, a1    ; a0 = a0 + a1",
        "note": "Sets arithmetic flags (Carry, Overflow, Zero, Sign) on most architectures.",
    },
    "SUB": {
        "concept": "Subtract — subtracts source from destination.",
        "x86_64_example": "sub rax, 5        ; RAX = RAX - 5",
        "arm64_example":   "sub x0, x0, #5   ; X0 = X0 - 5",
        "riscv_example":   "addi a0, a0, -5   ; a0 = a0 + (-5) [RISC-V has no SUBI]",
        "note": "Sets flags used by conditional jumps.",
    },
    "MUL": {
        "concept": "Multiply — multiplies two registers.",
        "x86_64_example": "imul rax, rbx     ; RAX = RAX × RBX (signed)",
        "arm64_example":   "mul x0, x0, x1    ; X0 = X0 × X1",
        "riscv_example":   "mul a0, a0, a1    ; a0 = a0 × a1 (requires M extension)",
        "note": "Result may be split into two registers for wide multiplication.",
    },
    "DIV": {
        "concept": "Divide — integer division.",
        "x86_64_example": "idiv rbx           ; quotient in RAX, remainder in RDX",
        "arm64_example":   "sdiv x0, x0, x1   ; X0 = X0 / X1 (signed)",
        "riscv_example":   "div a0, a0, a1    ; a0 = a0 / a1 (requires M extension)",
        "note": "Division by zero behavior is architecture-defined (trap or undefined).",
    },
    "CMP": {
        "concept": "Compare — performs subtraction but discards result; only sets flags.",
        "x86_64_example": "cmp rax, 10       ; sets flags based on RAX - 10",
        "arm64_example":   "cmp x0, #10       ; sets NZCV based on X0 - 10",
        "riscv_example":   "RISC-V uses branch instructions directly (beq, blt, bge, ...)",
        "note": "Used before conditional jump instructions (JE, JNE, JL, JG, etc.).",
    },
    "JMP": {
        "concept": "Unconditional jump — transfer control to another address.",
        "x86_64_example": "jmp 0x401000      ; jump to address 0x401000",
        "arm64_example":   "b label            ; branch to label",
        "riscv_example":   "j label            ; jump (pseudo: jal x0, label)",
        "note": "Used for loops, function calls, and control flow.",
    },
    "CALL": {
        "concept": "Call a subroutine — push return address onto stack, then jump.",
        "x86_64_example": "call my_function   ; push RIP, jump to my_function",
        "arm64_example":   "bl my_function     ; branch with link (saves PC in LR=X30)",
        "riscv_example":   "jal ra, my_function ; jump and link (saves return addr in ra=x1)",
        "note": "The callee uses RET/RET/jalr to return.",
    },
    "RET": {
        "concept": "Return from subroutine — restore control to the caller.",
        "x86_64_example": "ret               ; pop RIP from stack, jump there",
        "arm64_example":   "ret               ; jump to address in LR (X30)",
        "riscv_example":   "ret               ; pseudo: jalr x0, ra, 0",
        "note": "Must match the CALL or PUSH of the return address.",
    },
    "PUSH": {
        "concept": "Push a value onto the stack.",
        "x86_64_example": "push rax          ; RSP -= 8; [RSP] = RAX",
        "arm64_example":   "str x0, [sp, #-16]!  ; pre-decrement SP by 16, store X0",
        "riscv_example":   "addi sp, sp, -8 ; sw a0, 0(sp)  ; manual push",
        "note": "Stack grows downward on most architectures.",
    },
    "POP": {
        "concept": "Pop a value from the stack.",
        "x86_64_example": "pop rax           ; RAX = [RSP]; RSP += 8",
        "arm64_example":   "ldr x0, [sp], #16 ; load X0 from SP, post-increment SP",
        "riscv_example":   "lw a0, 0(sp) ; addi sp, sp, 8  ; manual pop",
        "note": "Reverses the PUSH operation.",
    },
    "LOAD": {
        "concept": "Load a value from memory into a register.",
        "x86_64_example": "mov rax, [rbx]    ; load value at address in RBX into RAX",
        "arm64_example":   "ldr x0, [x1]      ; load 64-bit value at address X1 into X0",
        "riscv_example":   "ld a0, 0(a1)      ; load doubleword from address a1",
        "note": "Memory access is much slower than register access.",
    },
    "STORE": {
        "concept": "Store a register value into memory.",
        "x86_64_example": "mov [rbx], rax    ; store RAX at address in RBX",
        "arm64_example":   "str x0, [x1]      ; store X0 at address X1",
        "riscv_example":   "sd a0, 0(a1)      ; store doubleword from a0 to address a1",
        "note": "Triggers cache/memory write, which can be expensive.",
    },
}


# ---------------------------------------------------------------------------
# AssemblyEducation
# ---------------------------------------------------------------------------

class AssemblyEducation:
    """
    Educational assembly language concepts module.

    Usage::
        ae = AssemblyEducation()
        ae.explain_instruction("MOV")
        ae.explain_architecture("x86-64")
        ae.explain_compilation_pipeline("C")
        ae.fetch_decode_execute_cycle()
    """

    # -----------------------------------------------------------------------
    # Instruction explanations
    # -----------------------------------------------------------------------

    def explain_instruction(self, mnemonic: str, arch: Optional[str] = None) -> dict:
        """
        Explain an assembly instruction concept.

        mnemonic: e.g. "MOV", "ADD", "CALL"
        arch: optional filter for architecture-specific example
        """
        key = mnemonic.upper().strip()
        info = INSTRUCTION_CONCEPTS.get(key)
        if info is None:
            return {
                "mnemonic": key,
                "found": False,
                "note": (
                    f"'{key}' is not in AXEN's instruction reference. "
                    "Supported: " + ", ".join(sorted(INSTRUCTION_CONCEPTS.keys()))
                ),
            }

        result: dict = {
            "mnemonic": key,
            "concept": info["concept"],
            "note": info.get("note", ""),
        }

        if arch:
            a = arch.lower()
            if "x86" in a or "amd64" in a or "x64" in a:
                result["example"] = info.get("x86_64_example", "")
                result["architecture"] = "x86-64"
            elif "arm" in a or "aarch64" in a:
                result["example"] = info.get("arm64_example", "")
                result["architecture"] = "ARM64"
            elif "risc" in a or "riscv" in a:
                result["example"] = info.get("riscv_example", "")
                result["architecture"] = "RISC-V"
        else:
            result["examples"] = {
                "x86-64": info.get("x86_64_example", "N/A"),
                "ARM64": info.get("arm64_example", "N/A"),
                "RISC-V": info.get("riscv_example", "N/A"),
            }
            result["note"] += " Examples shown for x86-64, ARM64, and RISC-V."

        return result

    def list_instructions(self) -> list[str]:
        """Return all documented instruction mnemonics."""
        return sorted(INSTRUCTION_CONCEPTS.keys())

    # -----------------------------------------------------------------------
    # Architecture explanations
    # -----------------------------------------------------------------------

    def explain_architecture(self, arch: str) -> dict:
        """Return detailed information about an architecture."""
        key = arch.lower().replace(" ", "-").replace("_", "-")
        # Normalize
        if key in ("x86-64", "x64", "amd64", "intel64"):
            key = "x86-64"
        elif key in ("arm64", "aarch64", "apple-silicon"):
            key = "arm64"
        elif key in ("risc-v", "riscv", "rv64"):
            key = "risc-v"

        info = ARCHITECTURES.get(key)
        if info is None:
            return {
                "found": False,
                "note": (
                    f"Architecture '{arch}' not recognized. "
                    "Supported: " + ", ".join(sorted(ARCHITECTURES.keys()))
                ),
            }
        return {**info, "architecture": key}

    def list_architectures(self) -> list[str]:
        """Return all documented architectures."""
        return list(ARCHITECTURES.keys())

    # -----------------------------------------------------------------------
    # Compilation pipeline
    # -----------------------------------------------------------------------

    def explain_compilation_pipeline(self, language: str = "C") -> dict:
        """
        Explain how high-level code becomes binary machine instructions.

        Shows the full stack:
          Source Code → Preprocessing → Compilation → Assembly → Linking → Machine Code → Binary → CPU
        """
        lang = language.upper()
        specific_notes = {
            "C": "C is compiled directly to assembly via a compiler (GCC, Clang, MSVC).",
            "C++": "Similar to C — compiled to assembly, with additional name mangling for templates/classes.",
            "PYTHON": "Python is typically interpreted (CPython) — compiled to bytecode, not native machine code directly. PyPy uses JIT compilation to native code.",
            "JAVASCRIPT": "Interpreted by a JS engine (V8, SpiderMonkey) — JIT compiled to native machine code at runtime.",
            "RUST": "Compiled to native machine code via LLVM backend — similar pipeline to C/C++.",
            "JAVA": "Compiled to JVM bytecode → JIT compiled to native code by the JVM at runtime.",
        }

        return {
            "language": lang,
            "pipeline": [
                {
                    "stage": 1,
                    "name": "Source Code",
                    "description": f"Human-readable {lang} source code.",
                    "example": "int x = 5 + 3;" if lang in ("C", "C++") else f"{lang} source file",
                },
                {
                    "stage": 2,
                    "name": "Preprocessing (C/C++)",
                    "description": "Macro expansion, #include file merging, conditional compilation.",
                    "example": "#include <stdio.h> → inserts stdio.h content",
                    "applies_to": ["C", "C++"],
                },
                {
                    "stage": 3,
                    "name": "Compilation",
                    "description": (
                        "Compiler (GCC, Clang, MSVC) translates source code to "
                        "assembly language for the target architecture."
                    ),
                    "tool_example": "gcc -S source.c -o source.s",
                    "output": "Assembly (.s or .asm file)",
                },
                {
                    "stage": 4,
                    "name": "Assembly",
                    "description": (
                        "Assembler (GNU as, NASM) converts assembly text to "
                        "object code — machine instructions in binary format."
                    ),
                    "tool_example": "as source.s -o source.o",
                    "output": "Object file (.o or .obj) — contains machine code + relocation info",
                },
                {
                    "stage": 5,
                    "name": "Linking",
                    "description": (
                        "Linker (ld, lld) combines object files and libraries "
                        "into a final executable. Resolves external references."
                    ),
                    "tool_example": "gcc source.o -o program",
                    "output": "Executable binary (ELF on Linux, PE on Windows, Mach-O on macOS)",
                },
                {
                    "stage": 6,
                    "name": "Machine Instructions",
                    "description": (
                        "The CPU loads the executable. The loader places code "
                        "segments in memory. The program counter (IP/PC) is set "
                        "to the entry point."
                    ),
                },
                {
                    "stage": 7,
                    "name": "Binary / Machine Code",
                    "description": (
                        "Instructions are binary-encoded according to the ISA. "
                        "Example: ADD RAX, 5 on x86-64 encodes to specific bytes."
                    ),
                    "note": "The exact byte encoding depends on the ISA opcode table, operand sizes, and prefixes.",
                },
                {
                    "stage": 8,
                    "name": "CPU Execution",
                    "description": (
                        "The CPU fetches, decodes, and executes each instruction. "
                        "The ALU performs arithmetic/logic. Results are written back "
                        "to registers or memory."
                    ),
                },
            ],
            "language_note": specific_notes.get(lang, f"{lang} compilation details vary by implementation."),
            "important": (
                "AXEN will NOT fabricate exact opcode bytes unless the architecture, "
                "assembler, compiler version, and full instruction context are known."
            ),
        }

    # -----------------------------------------------------------------------
    # Fetch-Decode-Execute cycle
    # -----------------------------------------------------------------------

    def fetch_decode_execute_cycle(self) -> dict:
        """
        Explain the CPU's fundamental instruction execution cycle.

        Returns the conceptual FDE cycle with an educational CPU model.
        """
        return {
            "name": "Fetch-Decode-Execute (FDE) Cycle",
            "description": (
                "The fundamental operating loop of every CPU. "
                "Repeats continuously from power-on until shutdown."
            ),
            "stages": [
                {
                    "stage": "FETCH",
                    "description": (
                        "The CPU reads the instruction at the address stored in the "
                        "Program Counter (PC/IP). The instruction is loaded into the "
                        "Instruction Register (IR). The PC is incremented to point to "
                        "the next instruction."
                    ),
                    "registers_involved": ["PC (Program Counter)", "MAR (Memory Address Register)",
                                           "MDR (Memory Data Register)", "IR (Instruction Register)"],
                    "memory_access": "Yes — reads instruction bytes from memory or instruction cache.",
                },
                {
                    "stage": "DECODE",
                    "description": (
                        "The Control Unit decodes the instruction in IR. "
                        "It identifies the opcode (what to do) and the operands "
                        "(what data to operate on — registers, immediate values, memory addresses)."
                    ),
                    "components": ["Control Unit", "Instruction Decoder"],
                    "output": "Control signals sent to ALU, register file, and memory bus.",
                },
                {
                    "stage": "EXECUTE",
                    "description": (
                        "The ALU (Arithmetic Logic Unit) performs the operation. "
                        "For memory instructions, the Memory Access stage loads or stores data. "
                        "For branches, the PC may be updated to a new address."
                    ),
                    "components": ["ALU", "FPU (for floating point)", "Branch Unit"],
                    "sub_stages": {
                        "Memory Access": "Load or store from/to memory (for LOAD/STORE instructions).",
                        "Write Back": "Write the result back to the destination register.",
                    },
                },
                {
                    "stage": "WRITE BACK",
                    "description": (
                        "The result computed by the ALU (or read from memory) "
                        "is written back into the destination register in the register file."
                    ),
                    "registers_involved": ["Destination register (e.g. RAX, X0, a0)"],
                },
            ],
            "cycle_rate": "A modern 5 GHz CPU completes billions of cycles per second. Each cycle can complete multiple instructions via pipelining.",
            "pipelining": (
                "Modern CPUs overlap FDE stages using a pipeline. "
                "While one instruction is EXECUTING, the next is being DECODED, "
                "and the one after that is being FETCHED — simultaneously."
            ),
            "out_of_order": (
                "Superscalar CPUs execute multiple instructions simultaneously. "
                "Out-of-order execution can execute instructions in a different sequence "
                "than written if data dependencies allow, then re-order results at retirement."
            ),
        }

    # -----------------------------------------------------------------------
    # Conceptual CPU model (for future emulator foundation)
    # -----------------------------------------------------------------------

    def educational_cpu_model(self) -> dict:
        """
        Return an educational abstraction of a simple CPU.

        This serves as the foundation for a future educational CPU emulator.
        Components are conceptual — not tied to a specific real architecture.
        """
        return {
            "name": "AXEN Educational CPU Model",
            "description": (
                "A simplified, educational abstraction of a CPU. "
                "Not intended to emulate any real hardware — useful for understanding concepts."
            ),
            "components": {
                "Registers": {
                    "PC": "Program Counter — address of next instruction",
                    "IR": "Instruction Register — holds current instruction being executed",
                    "ACC": "Accumulator — general-purpose data register",
                    "R0–R7": "General-purpose registers",
                    "SP": "Stack Pointer",
                    "FLAGS": "Condition flags: Z (Zero), N (Negative), C (Carry), V (Overflow)",
                },
                "Memory": {
                    "type": "Byte-addressable linear address space",
                    "segments": {
                        "Text/Code": "Stores machine instructions",
                        "Data": "Stores global and static variables",
                        "Heap": "Dynamic allocation",
                        "Stack": "Function calls, local variables",
                    },
                },
                "ALU": {
                    "operations": ["ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR", "NOT",
                                    "SHL", "SHR", "CMP"],
                    "inputs": "Two operands from registers or immediate values",
                    "outputs": "Result + updated FLAGS register",
                },
                "Control Unit": {
                    "role": "Orchestrates fetch, decode, execute, writeback",
                    "inputs": "Instruction from IR",
                    "outputs": "Control signals to ALU, register file, memory",
                },
                "Memory Bus": {
                    "role": "Data pathway between CPU and main memory",
                    "components": ["Address bus", "Data bus", "Control bus"],
                },
            },
            "instruction_cycle": "FETCH → DECODE → EXECUTE → WRITE BACK",
            "emulator_note": (
                "This model is structured to allow a simple educational CPU emulator "
                "to be built on top. Future AXEN versions may support step-through "
                "simulation of simple instruction sequences."
            ),
        }

    # -----------------------------------------------------------------------
    # Opcode concepts
    # -----------------------------------------------------------------------

    def explain_opcode(self) -> dict:
        """Explain what an opcode is and how machine instructions are encoded."""
        return {
            "name": "Opcode (Operation Code)",
            "description": (
                "An opcode is the binary value that uniquely identifies an operation "
                "in the CPU's instruction set. Each machine instruction begins with an opcode "
                "that tells the CPU what to do."
            ),
            "structure": {
                "x86_64_example": {
                    "instruction": "ADD RAX, RBX",
                    "opcode": "0x48 0x01 0xD8",
                    "breakdown": {
                        "0x48": "REX prefix (indicates 64-bit operands)",
                        "0x01": "Opcode for ADD r/m, r",
                        "0xD8": "ModRM byte (specifies RAX as destination, RBX as source)",
                    },
                    "caveat": (
                        "Exact encoding depends on assembler, operand sizes, and prefixes. "
                        "AXEN does not fabricate encodings without full context."
                    ),
                },
            },
            "key_points": [
                "Opcodes are architecture-specific — the same operation has different encodings on x86-64 vs ARM64 vs RISC-V.",
                "x86-64 uses variable-length instructions (1–15 bytes).",
                "ARM64 uses fixed-width 32-bit instructions.",
                "RISC-V base instructions are 32-bit; compressed (C) extension uses 16-bit.",
                "An assembler's job is to translate human-readable mnemonics to opcodes.",
                "A disassembler reverses this — translates binary opcodes back to mnemonics.",
            ],
        }
