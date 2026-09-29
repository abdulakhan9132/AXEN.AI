"""
axen/tools/computer/__init__.py
================================
AXEN Computer Systems / Low-Level Computing Module.

Provides deterministic tools for:
  - Number system conversions (binary, hex, octal, decimal)
  - Bitwise operations (AND, OR, XOR, NOT, shifts)
  - Bit manipulation (set, clear, toggle, check, extract, mask)
  - ASCII / Unicode character encoding
  - Hexadecimal arithmetic and representation
  - Integer representation (signed, unsigned, two's complement)
  - Memory concepts (bits, bytes, addresses, endianness, stack/heap)
  - Assembly language concepts (educational, multi-architecture)
  - CPU architecture concepts (fetch-decode-execute, registers, ALU)

Architecture:
    USER REQUEST
         ↓
    INTENT CLASSIFIER → computer_systems
         ↓
    ComputerController.handle(request)
         ↓
    Deterministic Sub-Tool  (binary.py, bitwise.py, ascii.py, …)
         ↓
    Structured Result
         ↓
    TinyLlama (optional enriched explanation)
         ↓
    "Computer Systems Verified" response
"""

from .binary import BinaryConverter
from .hexadecimal import HexConverter
from .bitwise import BitwiseOps
from .ascii_tool import AsciiTool
from .number_systems import NumberSystemConverter
from .integer_representation import IntegerRepresentation
from .memory import MemoryConcepts
from .assembly import AssemblyEducation
from .computer_controller import ComputerController, build_computer_tool

__all__ = [
    "BinaryConverter",
    "HexConverter",
    "BitwiseOps",
    "AsciiTool",
    "NumberSystemConverter",
    "IntegerRepresentation",
    "MemoryConcepts",
    "AssemblyEducation",
    "ComputerController",
    "build_computer_tool",
]
