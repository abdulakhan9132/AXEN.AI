"""
axen/tools/computer/memory.py
===============================
Educational memory concepts module for AXEN Computer Systems.

Provides structured explanations and deterministic calculations for:
  - Bits, bytes, nibbles
  - Memory addresses and address spaces
  - Binary memory representation
  - Hexadecimal memory dumps (formatted)
  - Stack vs Heap concepts
  - CPU registers
  - Pointers
  - Endianness (big-endian, little-endian)
  - Memory size conversions (KB, MB, GB, TB)

This module is educational — it explains and demonstrates concepts,
NOT emulate actual hardware memory.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

BYTE = 1
KILOBYTE = 1024
MEGABYTE = 1024 * 1024
GIGABYTE = 1024 ** 3
TERABYTE = 1024 ** 4

UNIT_MAP = {
    "bit": 1 / 8,
    "byte": 1,
    "nibble": 0.5,
    "kb": KILOBYTE,
    "kilobyte": KILOBYTE,
    "mb": MEGABYTE,
    "megabyte": MEGABYTE,
    "gb": GIGABYTE,
    "gigabyte": GIGABYTE,
    "tb": TERABYTE,
    "terabyte": TERABYTE,
}


# ---------------------------------------------------------------------------
# MemoryConcepts
# ---------------------------------------------------------------------------

class MemoryConcepts:
    """
    Educational memory representation and concepts.

    Usage::
        mc = MemoryConcepts()
        mc.explain_byte_storage(42)
        # → {"value": 42, "binary": "00101010", ...}

        mc.explain_endianness(0x12345678, bits=32)
        # → {"value": "0x12345678", "big_endian": [...], "little_endian": [...]}
    """

    # -----------------------------------------------------------------------
    # Bits, Bytes, Nibbles
    # -----------------------------------------------------------------------

    def bit_byte_info(self) -> dict:
        """Return fundamental bit/byte/nibble definitions."""
        return {
            "bit": {
                "description": "Smallest unit of digital information. Either 0 or 1.",
                "example": "1 or 0 — represents one binary digit (on/off signal).",
            },
            "nibble": {
                "description": "4 bits = half a byte. Represents one hex digit.",
                "example": "1010 = 0xA = 10",
                "hex_digits": "A nibble can represent 0–15 (one hex digit 0–F).",
            },
            "byte": {
                "description": "8 bits. The fundamental unit of addressable memory.",
                "example": "01000001 = 65 = 'A' in ASCII",
                "range": "0–255 unsigned, -128–127 signed (two's complement).",
            },
            "word": {
                "description": "Architecture-dependent. Commonly 16-bit, 32-bit, or 64-bit.",
                "note": "On modern 64-bit systems, a 'word' is typically 64 bits (8 bytes).",
            },
        }

    def explain_byte_storage(self, value: int) -> dict:
        """
        Explain how a decimal integer is stored as a byte (8-bit unsigned).

        For values > 255, uses the minimum required bytes.
        """
        if value < 0:
            raise ValueError(
                "Use IntegerRepresentation.to_twos_complement() for negative values."
            )

        bits_needed = max(value.bit_length(), 1)
        bytes_needed = max((bits_needed + 7) // 8, 1)

        byte_representations = []
        temp = value
        for i in range(bytes_needed - 1, -1, -1):
            byte_val = (value >> (i * 8)) & 0xFF
            byte_representations.append({
                "byte_index": bytes_needed - 1 - i,
                "decimal": byte_val,
                "binary": bin(byte_val)[2:].zfill(8),
                "hexadecimal": hex(byte_val)[2:].upper().zfill(2),
            })

        return {
            "value": value,
            "decimal": value,
            "binary": bin(value)[2:].zfill(bits_needed),
            "hexadecimal": hex(value)[2:].upper(),
            "bits_needed": bits_needed,
            "bytes_needed": bytes_needed,
            "byte_representation": byte_representations,
            "note": (
                f"{value} decimal = {bin(value)[2:]} binary. "
                f"Requires {bytes_needed} byte{'s' if bytes_needed > 1 else ''}."
            ),
        }

    # -----------------------------------------------------------------------
    # Endianness
    # -----------------------------------------------------------------------

    def explain_endianness(self, value: int, bits: int = 32) -> dict:
        """
        Show how a multi-byte value is stored in big-endian vs little-endian order.

        value: the integer to represent
        bits: total bit width (must be 16, 32, or 64)
        """
        if bits not in (16, 32, 64):
            raise ValueError("Endianness demonstration requires bits = 16, 32, or 64.")

        num_bytes = bits // 8
        byte_vals = [(value >> (i * 8)) & 0xFF for i in range(num_bytes - 1, -1, -1)]
        # byte_vals is now in big-endian order (most significant first)

        big_endian = [{"address_offset": i, "byte": hex(b)[2:].upper().zfill(2)}
                      for i, b in enumerate(byte_vals)]
        little_endian = [{"address_offset": i, "byte": hex(b)[2:].upper().zfill(2)}
                         for i, b in enumerate(reversed(byte_vals))]

        return {
            "value": hex(value)[2:].upper().zfill(num_bytes * 2),
            "bits": bits,
            "num_bytes": num_bytes,
            "big_endian": {
                "description": "Most significant byte stored at the lowest address (network byte order).",
                "bytes": big_endian,
                "example_arch": "SPARC, IBM mainframes, network protocols (TCP/IP).",
            },
            "little_endian": {
                "description": "Least significant byte stored at the lowest address.",
                "bytes": little_endian,
                "example_arch": "x86, x86-64 (Intel/AMD), ARM (default mode).",
            },
            "note": (
                "Endianness matters when reading multi-byte values from memory, "
                "files, or network streams across different architectures."
            ),
        }

    # -----------------------------------------------------------------------
    # Memory addresses
    # -----------------------------------------------------------------------

    def explain_address_space(self, bits: int = 32) -> dict:
        """
        Explain the address space for a given address bus width.

        bits: address bus width (e.g. 32 or 64)
        """
        total_addresses = 1 << bits
        total_bytes = total_addresses  # one byte per address (byte-addressable)
        total_gb = total_bytes / (1024 ** 3)

        return {
            "address_bits": bits,
            "total_addresses": total_addresses,
            "total_addressable_bytes": total_bytes,
            "total_addressable_gb": round(total_gb, 2),
            "example_range": f"0x{'0' * (bits // 4)} to 0x{'F' * (bits // 4)}",
            "note": (
                f"A {bits}-bit address bus can reference {total_addresses:,} unique "
                f"memory locations = {total_gb:.2f} GB of byte-addressable memory. "
                "Modern 64-bit CPUs typically use 48 actual bits → 256 TB virtual space."
            ),
        }

    def format_memory_dump(self, data: list[int], start_address: int = 0) -> list[dict]:
        """
        Format a list of byte values as a hexadecimal memory dump.

        data: list of integers (0–255), each representing one byte
        start_address: starting address for display

        Returns list of dicts with: address (hex), hex_value, binary, ascii_char.
        """
        result = []
        for i, byte_val in enumerate(data):
            if not (0 <= byte_val <= 255):
                raise ValueError(f"Invalid byte value {byte_val} at index {i}. Must be 0–255.")
            addr = start_address + i
            char = chr(byte_val) if 32 <= byte_val <= 126 else "."
            result.append({
                "address": hex(addr)[2:].upper().zfill(8),
                "hex_value": hex(byte_val)[2:].upper().zfill(2),
                "binary": bin(byte_val)[2:].zfill(8),
                "decimal": byte_val,
                "ascii_char": char,
            })
        return result

    # -----------------------------------------------------------------------
    # Stack and Heap
    # -----------------------------------------------------------------------

    def explain_stack(self) -> dict:
        """Return an educational explanation of the stack."""
        return {
            "name": "Stack",
            "description": (
                "A region of memory that grows and shrinks automatically as functions "
                "are called and return. It uses a Last-In, First-Out (LIFO) discipline."
            ),
            "managed_by": "CPU and operating system automatically.",
            "stores": [
                "Function return addresses",
                "Local variables",
                "Function parameters (in some calling conventions)",
                "Saved register values",
            ],
            "key_register": "Stack Pointer (SP or RSP on x86-64) — always points to the top of the stack.",
            "growth_direction": "Typically grows downward (toward lower addresses) on most architectures.",
            "operations": {
                "PUSH": "Decrements SP, then writes value at new SP address.",
                "POP": "Reads value at SP address, then increments SP.",
            },
            "risks": "Stack overflow — if too much data is pushed (e.g., deep recursion).",
        }

    def explain_heap(self) -> dict:
        """Return an educational explanation of the heap."""
        return {
            "name": "Heap",
            "description": (
                "A region of memory used for dynamic memory allocation. "
                "Unlike the stack, the heap is manually managed by the programmer "
                "(or automatically via garbage collection in managed languages)."
            ),
            "managed_by": "Programmer (malloc/free in C) or garbage collector (Python, Java, Go).",
            "stores": [
                "Dynamically allocated objects",
                "Large data structures",
                "Data whose lifetime extends beyond a single function call",
            ],
            "operations": {
                "allocate": "malloc() / new / Python object creation",
                "deallocate": "free() / delete / garbage collection",
            },
            "risks": [
                "Memory leaks — allocated memory not freed",
                "Dangling pointers — using memory after it's freed",
                "Heap fragmentation — many small allocations leaving unusable gaps",
            ],
        }

    # -----------------------------------------------------------------------
    # Registers
    # -----------------------------------------------------------------------

    def explain_registers(self, arch: str = "x86-64") -> dict:
        """
        Return educational information about CPU registers.

        arch: one of "x86-64", "arm64", "generic"
        """
        generic = {
            "description": (
                "Registers are extremely fast storage locations inside the CPU itself. "
                "They are orders of magnitude faster than RAM because they are built "
                "directly into the processor's silicon."
            ),
            "typical_size": "8, 16, 32, or 64 bits depending on architecture.",
            "categories": {
                "General Purpose": "Hold data for computation (integers, addresses).",
                "Program Counter (PC/IP)": "Holds the address of the NEXT instruction to execute.",
                "Stack Pointer (SP)": "Points to the top of the stack.",
                "Base Pointer (BP/FP)": "Points to the base of the current stack frame.",
                "Status/Flags Register": "Holds condition flags: Zero, Carry, Sign, Overflow, etc.",
                "Instruction Register (IR)": "Holds the currently executing instruction (inside CPU).",
                "Memory Address Register (MAR)": "Holds address for memory access.",
                "Memory Data Register (MDR)": "Holds data being read from or written to memory.",
            },
        }

        if arch.lower() in ("x86-64", "x64"):
            generic["architecture"] = "x86-64"
            generic["general_purpose_registers"] = {
                "RAX": "Accumulator — return values, arithmetic",
                "RBX": "Base — general purpose",
                "RCX": "Counter — loops, string ops",
                "RDX": "Data — I/O, multiplication overflow",
                "RSI": "Source Index — string/array ops",
                "RDI": "Destination Index — string/array ops",
                "RBP": "Base Pointer — stack frame base",
                "RSP": "Stack Pointer — top of stack",
                "R8–R15": "Additional 64-bit general purpose registers (added in x86-64)",
                "RIP": "Instruction Pointer — address of next instruction",
                "RFLAGS": "Status flags register",
            }
            generic["register_widths"] = {
                "note": "x86-64 registers have sub-registers for backward compatibility.",
                "example": "RAX (64-bit) → EAX (lower 32-bit) → AX (lower 16-bit) → AH/AL (high/low 8-bit)",
            }
        elif arch.lower() == "arm64":
            generic["architecture"] = "ARM64 (AArch64)"
            generic["general_purpose_registers"] = {
                "X0–X7": "Arguments and return values",
                "X8": "Indirect result location register",
                "X9–X15": "Caller-saved temporaries",
                "X16–X17": "Intra-procedure-call scratch registers",
                "X18": "Platform register (OS-specific)",
                "X19–X28": "Callee-saved registers",
                "X29 (FP)": "Frame Pointer",
                "X30 (LR)": "Link Register — holds return address",
                "SP": "Stack Pointer",
                "PC": "Program Counter (not directly accessible as a GPR)",
                "NZCV": "Condition flags (Negative, Zero, Carry, oVerflow)",
            }

        return generic

    # -----------------------------------------------------------------------
    # Memory size conversions
    # -----------------------------------------------------------------------

    def convert_memory_size(self, value: float, from_unit: str, to_unit: str) -> dict:
        """
        Convert between memory size units.

        from_unit / to_unit: "bit", "byte", "kb", "mb", "gb", "tb"
        """
        f_key = from_unit.lower()
        t_key = to_unit.lower()
        if f_key not in UNIT_MAP:
            raise ValueError(f"Unknown unit: '{from_unit}'. Valid: {list(UNIT_MAP.keys())}")
        if t_key not in UNIT_MAP:
            raise ValueError(f"Unknown unit: '{to_unit}'. Valid: {list(UNIT_MAP.keys())}")

        value_in_bytes = value * UNIT_MAP[f_key]
        result = value_in_bytes / UNIT_MAP[t_key]

        return {
            "input": f"{value} {from_unit}",
            "output": f"{result:g} {to_unit}",
            "value": result,
            "from_unit": from_unit,
            "to_unit": to_unit,
        }

    # -----------------------------------------------------------------------
    # Pointers
    # -----------------------------------------------------------------------

    def explain_pointer(self, arch: str = "x86-64") -> dict:
        """Return an educational explanation of pointers."""
        width = "64-bit (8 bytes)" if "64" in arch else "32-bit (4 bytes)"
        return {
            "name": "Pointer",
            "description": (
                "A pointer is a variable that stores the memory address of another variable. "
                "Instead of holding a data value directly, it 'points to' where data lives in memory."
            ),
            "size": f"On {arch}: {width} — same width as the address bus.",
            "example": {
                "scenario": "int x = 42; stored at address 0x7FFF1234",
                "pointer": "int* p = &x; → p holds the value 0x7FFF1234",
                "dereference": "*p → reads the value at 0x7FFF1234 → returns 42",
            },
            "use_cases": [
                "Dynamic memory allocation",
                "Passing large data structures efficiently",
                "Implementing linked lists, trees, graphs",
                "Function pointers (callbacks)",
            ],
            "dangers": [
                "Null pointer dereference",
                "Dangling pointers (pointing to freed memory)",
                "Buffer overflows via pointer arithmetic",
            ],
        }
