"""
tests/test_computer_systems.py
================================
Comprehensive test suite for AXEN Computer Systems / Low-Level Computing Module.

Tests cover:
  - Binary conversions (decimal ↔ binary ↔ hex ↔ octal)
  - Binary arithmetic (+, -, ×, ÷)
  - Bitwise operations (AND, OR, XOR, NOT, shifts)
  - Bit manipulation (set, clear, toggle, check, extract, mask)
  - ASCII encoding (char → binary, decimal → char, string → binary)
  - Hexadecimal operations and arithmetic
  - Two's complement (signed integer representation)
  - Integer ranges (signed/unsigned)
  - Memory concepts (endianness, byte storage)
  - Number system detection and info
  - Invalid input error handling
  - Overflow detection
  - Assembly education
  - Classifier routing (computer_systems intent)
"""

from __future__ import annotations

import pytest
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axen.tools.computer.binary import BinaryConverter
from axen.tools.computer.hexadecimal import HexConverter
from axen.tools.computer.bitwise import BitwiseOps
from axen.tools.computer.ascii_tool import AsciiTool
from axen.tools.computer.number_systems import NumberSystemConverter
from axen.tools.computer.integer_representation import IntegerRepresentation
from axen.tools.computer.memory import MemoryConcepts
from axen.tools.computer.assembly import AssemblyEducation
from axen.tools.computer.computer_controller import ComputerController


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def bc():
    return BinaryConverter()

@pytest.fixture
def hc():
    return HexConverter()

@pytest.fixture
def bw():
    return BitwiseOps()

@pytest.fixture
def at():
    return AsciiTool()

@pytest.fixture
def ns():
    return NumberSystemConverter()

@pytest.fixture
def ir():
    return IntegerRepresentation()

@pytest.fixture
def mc():
    return MemoryConcepts()

@pytest.fixture
def ae():
    return AssemblyEducation()

@pytest.fixture
def ctrl():
    return ComputerController()


# ===========================================================================
# 1. BINARY CONVERSIONS
# ===========================================================================

class TestBinaryConversions:

    def test_decimal_to_binary_25(self, bc):
        assert bc.decimal_to_binary(25) == "11001"

    def test_decimal_to_binary_42(self, bc):
        assert bc.decimal_to_binary(42) == "101010"

    def test_decimal_to_binary_255(self, bc):
        assert bc.decimal_to_binary(255) == "11111111"

    def test_decimal_to_binary_0(self, bc):
        assert bc.decimal_to_binary(0) == "0"

    def test_decimal_to_binary_1(self, bc):
        assert bc.decimal_to_binary(1) == "1"

    def test_binary_to_decimal_101010(self, bc):
        assert bc.binary_to_decimal("101010") == 42

    def test_binary_to_decimal_11111111(self, bc):
        assert bc.binary_to_decimal("11111111") == 255

    def test_binary_to_decimal_0(self, bc):
        assert bc.binary_to_decimal("0") == 0

    def test_binary_to_decimal_11001(self, bc):
        assert bc.binary_to_decimal("11001") == 25

    def test_binary_to_hex_101010(self, bc):
        assert bc.binary_to_hex("101010") == "2A"

    def test_binary_to_hex_11111111(self, bc):
        assert bc.binary_to_hex("11111111") == "FF"

    def test_hex_to_binary_FF(self, bc):
        result = bc.hex_to_binary("FF")
        assert result == "11111111"

    def test_hex_to_binary_2A(self, bc):
        result = bc.hex_to_binary("2A")
        assert result == "101010"

    def test_binary_to_octal(self, bc):
        # 42 = 101010 binary = 52 octal
        assert bc.binary_to_octal("101010") == "52"

    def test_octal_to_binary(self, bc):
        # 52 octal = 101010 binary
        assert bc.octal_to_binary("52") == "101010"

    def test_format_conversion_42(self, bc):
        result = bc.format_conversion(42)
        assert result["decimal"] == 42
        assert result["binary"] == "101010"
        assert result["hexadecimal"] == "2A"
        assert result["octal"] == "52"


# ===========================================================================
# 2. BINARY ARITHMETIC
# ===========================================================================

class TestBinaryArithmetic:

    def test_binary_add_1010_0110(self, bc):
        result = bc.binary_add("1010", "0110")
        assert result["result"] == "10000"
        assert result["decimal_result"] == 16
        assert result["decimal_a"] == 10
        assert result["decimal_b"] == 6

    def test_binary_add_shows_steps(self, bc):
        result = bc.binary_add("1010", "0110", show_steps=True)
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_binary_add_zero(self, bc):
        result = bc.binary_add("0", "0")
        assert result["result"] == "0"
        assert result["decimal_result"] == 0

    def test_binary_subtract_simple(self, bc):
        # 10000 - 0110 = 1010
        result = bc.binary_subtract("10000", "0110")
        assert result["decimal_result"] == 10
        assert not result["is_negative"]

    def test_binary_multiply_101_11(self, bc):
        # 5 * 3 = 15
        result = bc.binary_multiply("101", "11")
        assert result["decimal_result"] == 15
        assert result["result"] == "1111"

    def test_binary_divide_1010_10(self, bc):
        # 10 / 2 = 5 remainder 0
        result = bc.binary_divide("1010", "10")
        assert result["decimal_quotient"] == 5
        assert result["decimal_remainder"] == 0

    def test_binary_divide_by_zero_raises(self, bc):
        with pytest.raises(ZeroDivisionError):
            bc.binary_divide("1010", "0")


# ===========================================================================
# 3. BITWISE OPERATIONS
# ===========================================================================

class TestBitwiseOps:

    def test_AND_1010_1100(self, bw):
        result = bw.AND(0b1010, 0b1100)
        assert result["result_bin"] == "1000"
        assert result["result_decimal"] == 8

    def test_OR_1010_1100(self, bw):
        result = bw.OR(0b1010, 0b1100)
        assert result["result_bin"] == "1110"
        assert result["result_decimal"] == 14

    def test_XOR_1010_1100(self, bw):
        result = bw.XOR(0b1010, 0b1100)
        assert result["result_bin"] == "0110"
        assert result["result_decimal"] == 6

    def test_NOT_1010_4bit(self, bw):
        # NOT 1010 using 4 bits = 0101
        result = bw.NOT(0b1010, bits=4)
        assert result["result_bin"] == "0101"
        assert result["result_decimal"] == 5

    def test_NOT_00000000_8bit(self, bw):
        result = bw.NOT(0b00000000, bits=8)
        assert result["result_bin"] == "11111111"
        assert result["result_decimal"] == 255

    def test_NOT_requires_width(self, bw):
        # Verify that NOT uses explicit bit width — 8 by default
        result = bw.NOT(0b1010, bits=8)
        # NOT 00001010 in 8-bit = 11110101 = 245
        assert result["result_decimal"] == 245

    def test_left_shift(self, bw):
        result = bw.left_shift(0b0001, 3)
        assert result["result_decimal"] == 8
        assert result["result_bin"] == "1000"

    def test_right_shift(self, bw):
        result = bw.right_shift(0b1000, 2)
        assert result["result_decimal"] == 2
        assert result["result_bin"] == "10"

    def test_from_binary_strings_AND(self, bw):
        result = bw.from_binary_strings("AND", "1010", "1100")
        assert result["result_bin"] == "1000"

    def test_from_binary_strings_OR(self, bw):
        result = bw.from_binary_strings("OR", "1010", "1100")
        assert result["result_bin"] == "1110"

    def test_from_binary_strings_XOR(self, bw):
        result = bw.from_binary_strings("XOR", "1010", "1100")
        assert result["result_bin"] == "0110"

    def test_from_binary_strings_NOT(self, bw):
        result = bw.from_binary_strings("NOT", "1010", bits=4)
        assert result["result_bin"] == "0101"


# ===========================================================================
# 4. BIT MANIPULATION
# ===========================================================================

class TestBitManipulation:

    def test_set_bit(self, bw):
        # Set bit 2 of 1000 (8) → 1100 (12)
        result = bw.set_bit(0b1000, 2)
        assert result["result_decimal"] == 0b1100
        assert "1100" in result["result_bin"]

    def test_clear_bit(self, bw):
        # Clear bit 3 of 1111 (15) → 0111 (7)
        result = bw.clear_bit(0b1111, 3)
        assert result["result_decimal"] == 7

    def test_toggle_bit(self, bw):
        # Toggle bit 0 of 1010 → 1011
        result = bw.toggle_bit(0b1010, 0)
        assert result["result_decimal"] == 0b1011

    def test_check_bit_set(self, bw):
        # Bit 3 of 1000 (8) is set
        result = bw.check_bit(0b1000, 3)
        assert result["is_set"] is True

    def test_check_bit_clear(self, bw):
        # Bit 2 of 1000 (8) is NOT set
        result = bw.check_bit(0b1000, 2)
        assert result["is_set"] is False

    def test_extract_bits(self, bw):
        # Extract bits 5:3 of 0b10110100 → bits 5,4,3 = 110 = 6
        result = bw.extract_bits(0b10110100, 5, 3)
        assert result["result_decimal"] == 6

    def test_mask_bits(self, bw):
        # 0b11110000 mask on 0b11001100 → 0b11000000 = 192
        result = bw.mask_bits(0b11001100, 0b11110000)
        assert result["result_decimal"] == 0b11000000

    def test_invalid_shift_negative(self, bw):
        with pytest.raises(ValueError):
            bw.left_shift(1, -1)

    def test_invalid_not_width(self, bw):
        with pytest.raises(ValueError):
            bw.NOT(0b1010, bits=7)  # 7 is not in (4, 8, 16, 32, 64)


# ===========================================================================
# 5. ASCII
# ===========================================================================

class TestAscii:

    def test_A_to_info(self, at):
        result = at.char_to_info("A")
        assert result["decimal"] == 65
        assert result["binary"] == "01000001"
        assert result["hexadecimal"] == "41"
        assert result["char"] == "A"

    def test_decimal_65_to_char(self, at):
        result = at.decimal_to_char(65)
        assert result["char"] == "A"

    def test_ascii_a_lowercase(self, at):
        result = at.char_to_info("a")
        assert result["decimal"] == 97
        assert result["binary"] == "01100001"

    def test_string_to_binary(self, at):
        result = at.string_to_binary("Hi")
        assert len(result) == 2
        h_char = result[0]
        i_char = result[1]
        assert h_char["decimal"] == 72
        assert i_char["decimal"] == 105

    def test_string_to_binary_compact(self, at):
        result = at.string_to_binary_compact("Hi")
        assert "01001000" in result  # H
        assert "01101001" in result  # i

    def test_string_to_hex(self, at):
        result = at.string_to_hex("Hi")
        assert "48" in result  # H = 0x48
        assert "69" in result  # i = 0x69

    def test_binary_to_string(self, at):
        result = at.binary_to_string("01001000 01101001")
        assert result == "Hi"

    def test_space_char(self, at):
        result = at.char_to_info(" ")
        assert result["decimal"] == 32

    def test_null_char(self, at):
        result = at.char_to_info("\x00")
        assert result["decimal"] == 0
        assert not result["is_printable"]

    def test_empty_string_raises(self, at):
        with pytest.raises(ValueError):
            at.string_to_binary("")

    def test_multi_char_input_raises(self, at):
        with pytest.raises(ValueError):
            at.char_to_info("AB")  # should be single char


# ===========================================================================
# 6. HEXADECIMAL
# ===========================================================================

class TestHexadecimal:

    def test_decimal_to_hex_255(self, hc):
        assert hc.decimal_to_hex(255) == "FF"

    def test_hex_to_decimal_FF(self, hc):
        assert hc.hex_to_decimal("FF") == 255

    def test_hex_to_decimal_2A(self, hc):
        assert hc.hex_to_decimal("2A") == 42

    def test_decimal_to_hex_0(self, hc):
        assert hc.decimal_to_hex(0) == "0"

    def test_hex_to_binary_FF(self, hc):
        assert hc.hex_to_binary("FF") == "11111111"

    def test_binary_to_hex(self, hc):
        assert hc.binary_to_hex("11111111") == "FF"

    def test_hex_add(self, hc):
        result = hc.hex_add("1A", "0F")
        assert result["decimal_result"] == 26 + 15
        assert result["result"] == "0x29"

    def test_hex_subtract(self, hc):
        result = hc.hex_subtract("FF", "0F")
        assert result["decimal_result"] == 255 - 15

    def test_hex_multiply(self, hc):
        result = hc.hex_multiply("10", "A")
        # 16 * 10 = 160 = 0xA0
        assert result["decimal_result"] == 160

    def test_hex_AND(self, hc):
        result = hc.hex_and("FF", "0F")
        # 0xFF & 0x0F = 0x0F = 15
        assert result["decimal_result"] == 0xFF & 0x0F
        assert result["decimal_result"] == 15

    def test_hex_OR(self, hc):
        result = hc.hex_or("F0", "0F")
        assert result["decimal_result"] == 0xFF

    def test_hex_XOR(self, hc):
        result = hc.hex_xor("FF", "F0")
        assert result["decimal_result"] == 0x0F

    def test_hex_NOT_8bit(self, hc):
        result = hc.hex_not("0F", bits=8)
        assert result["decimal_result"] == 0xF0

    def test_invalid_hex_raises(self, hc):
        with pytest.raises(ValueError):
            hc.hex_to_decimal("ZZ")

    def test_nibble_breakdown(self, hc):
        result = hc.nibble_breakdown("2A")
        assert len(result) == 2
        assert result[0]["decimal"] == 2   # '2'
        assert result[1]["decimal"] == 10  # 'A'

    def test_hex_to_octal(self, hc):
        # 0xFF = 255 = 377 octal
        result = hc.hex_to_octal("FF")
        assert result == "377"


# ===========================================================================
# 7. TWO'S COMPLEMENT / INTEGER REPRESENTATION
# ===========================================================================

class TestIntegerRepresentation:

    def test_twos_complement_minus5_8bit(self, ir):
        result = ir.to_twos_complement(-5, bits=8)
        assert result["binary"] == "11111011"
        assert result["value"] == -5
        assert result["bits"] == 8

    def test_twos_complement_minus1_8bit(self, ir):
        result = ir.to_twos_complement(-1, bits=8)
        assert result["binary"] == "11111111"

    def test_twos_complement_zero_8bit(self, ir):
        result = ir.to_twos_complement(0, bits=8)
        assert result["binary"] == "00000000"

    def test_twos_complement_positive_8bit(self, ir):
        result = ir.to_twos_complement(42, bits=8)
        assert result["binary"] == "00101010"

    def test_twos_complement_minus128_8bit(self, ir):
        # -128 is the minimum for 8-bit signed
        result = ir.to_twos_complement(-128, bits=8)
        assert result["binary"] == "10000000"

    def test_twos_complement_overflow_raises(self, ir):
        with pytest.raises(OverflowError):
            ir.to_twos_complement(128, bits=8)  # max is 127 for signed 8-bit

    def test_from_twos_complement_11111011(self, ir):
        result = ir.from_twos_complement("11111011", bits=8)
        assert result["decimal"] == -5
        assert result["is_negative"] is True

    def test_from_twos_complement_positive(self, ir):
        result = ir.from_twos_complement("01000001", bits=8)
        assert result["decimal"] == 65
        assert result["is_negative"] is False

    def test_unsigned_range_8bit(self, ir):
        result = ir.unsigned_range(8)
        assert result["min"] == 0
        assert result["max"] == 255

    def test_signed_range_8bit(self, ir):
        result = ir.signed_range(8)
        assert result["min"] == -128
        assert result["max"] == 127

    def test_signed_range_16bit(self, ir):
        result = ir.signed_range(16)
        assert result["min"] == -32768
        assert result["max"] == 32767

    def test_unsigned_overflow_raises(self, ir):
        with pytest.raises(OverflowError):
            ir.to_unsigned(256, bits=8)

    def test_negative_to_unsigned_raises(self, ir):
        with pytest.raises(ValueError):
            ir.to_unsigned(-1, bits=8)

    def test_to_twos_complement_has_steps(self, ir):
        result = ir.to_twos_complement(-5, bits=8, show_steps=True)
        assert "steps" in result
        assert len(result["steps"]) >= 4

    def test_sign_extend_4bit_to_8bit(self, ir):
        # -1 in 8-bit two's complement = 11111111, sign-extended to 16-bit = 1111111111111111
        # Using valid supported widths: 8 → 16
        result = ir.sign_extend("11111111", from_bits=8, to_bits=16)
        assert result["decimal_value"] == -1
        # Sign bit is 1, extended binary should be 16 bits of 1s
        assert result["extended_binary"] == "1111111111111111"
        assert result["sign_bit"] == "1"


    def test_overflow_check_passes(self, ir):
        result = ir.check_overflow(127, bits=8, signed=True)
        assert not result["overflows"]

    def test_overflow_check_fails(self, ir):
        result = ir.check_overflow(128, bits=8, signed=True)
        assert result["overflows"]


# ===========================================================================
# 8. NUMBER SYSTEMS
# ===========================================================================

class TestNumberSystems:

    def test_convert_decimal_42(self, ns):
        result = ns.convert(42, from_base="decimal")
        assert result["decimal"] == 42
        assert result["binary"] == "101010"
        assert result["hexadecimal"] == "2A"
        assert result["octal"] == "52"

    def test_convert_binary_to_all(self, ns):
        result = ns.convert("101010", from_base="binary")
        assert result["decimal"] == 42

    def test_convert_hex_to_all(self, ns):
        result = ns.convert("FF", from_base="hexadecimal")
        assert result["decimal"] == 255
        assert result["binary"] == "11111111"

    def test_convert_octal_to_all(self, ns):
        result = ns.convert("52", from_base="octal")
        assert result["decimal"] == 42

    def test_detect_base_hex(self, ns):
        assert ns.detect_base("0xFF") == "hexadecimal"
        assert ns.detect_base("0x2A") == "hexadecimal"
        assert ns.detect_base("FF") == "hexadecimal"

    def test_detect_base_binary_prefix(self, ns):
        assert ns.detect_base("0b101010") == "binary"

    def test_positional_breakdown(self, ns):
        breakdown = ns.positional_breakdown("101010", "binary")
        total = sum(d["contribution"] for d in breakdown)
        assert total == 42  # 101010 binary = 42 decimal

    def test_explain_system_binary(self, ns):
        info = ns.explain_system("binary")
        assert info["radix"] == 2
        assert "0" in info["digits"] and "1" in info["digits"]

    def test_explain_system_hex(self, ns):
        info = ns.explain_system("hexadecimal")
        assert info["radix"] == 16

    def test_invalid_base_raises(self, ns):
        with pytest.raises(ValueError):
            ns.convert("42", from_base="base36")

    def test_convert_0(self, ns):
        result = ns.convert(0, from_base="decimal")
        assert result["decimal"] == 0
        assert result["binary"] == "0"


# ===========================================================================
# 9. MEMORY CONCEPTS
# ===========================================================================

class TestMemoryConcepts:

    def test_bit_byte_info(self, mc):
        info = mc.bit_byte_info()
        assert "bit" in info
        assert "byte" in info
        assert "nibble" in info

    def test_byte_storage_42(self, mc):
        result = mc.explain_byte_storage(42)
        assert result["value"] == 42
        assert result["binary"] == "101010"
        assert result["bytes_needed"] == 1

    def test_byte_storage_256(self, mc):
        result = mc.explain_byte_storage(256)
        assert result["bytes_needed"] == 2

    def test_endianness_32bit(self, mc):
        result = mc.explain_endianness(0x12345678, bits=32)
        assert "big_endian" in result
        assert "little_endian" in result
        # Big-endian: most significant byte first → 12
        assert result["big_endian"]["bytes"][0]["byte"] == "12"
        # Little-endian: least significant byte first → 78
        assert result["little_endian"]["bytes"][0]["byte"] == "78"

    def test_explain_stack(self, mc):
        result = mc.explain_stack()
        assert result["name"] == "Stack"
        assert "LIFO" in result["description"] or "Last" in result["description"]

    def test_explain_heap(self, mc):
        result = mc.explain_heap()
        assert result["name"] == "Heap"

    def test_explain_registers_x86(self, mc):
        result = mc.explain_registers(arch="x86-64")
        assert "RAX" in result.get("general_purpose_registers", {})
        assert "RSP" in result.get("general_purpose_registers", {})

    def test_explain_registers_arm64(self, mc):
        result = mc.explain_registers(arch="arm64")
        assert "X0–X7" in result.get("general_purpose_registers", {})

    def test_address_space_32bit(self, mc):
        result = mc.explain_address_space(bits=32)
        assert result["address_bits"] == 32
        assert result["total_addressable_gb"] == pytest.approx(4.0, abs=0.1)

    def test_memory_dump(self, mc):
        dump = mc.format_memory_dump([0x41, 0x42, 0x43], start_address=0x100)
        assert len(dump) == 3
        assert dump[0]["ascii_char"] == "A"
        assert dump[1]["ascii_char"] == "B"
        assert dump[0]["address"] == "00000100"

    def test_memory_size_kb_to_bytes(self, mc):
        result = mc.convert_memory_size(1, "kb", "byte")
        assert result["value"] == 1024

    def test_memory_size_gb_to_mb(self, mc):
        result = mc.convert_memory_size(1, "gb", "mb")
        assert result["value"] == 1024

    def test_invalid_memory_unit(self, mc):
        with pytest.raises(ValueError):
            mc.convert_memory_size(1, "petabyte", "byte")


# ===========================================================================
# 10. ASSEMBLY EDUCATION
# ===========================================================================

class TestAssemblyEducation:

    def test_explain_mov(self, ae):
        result = ae.explain_instruction("MOV")
        assert result["mnemonic"] == "MOV"
        assert "copy" in result["concept"].lower() or "move" in result["concept"].lower()

    def test_explain_add(self, ae):
        result = ae.explain_instruction("ADD")
        assert result["mnemonic"] == "ADD"

    def test_explain_call(self, ae):
        result = ae.explain_instruction("CALL")
        assert "subroutine" in result["concept"].lower() or "function" in result["concept"].lower() or "return" in result["concept"].lower()

    def test_explain_unknown_instruction(self, ae):
        result = ae.explain_instruction("NOPE")
        assert result["found"] is False

    def test_explain_architecture_x86(self, ae):
        result = ae.explain_architecture("x86-64")
        assert result["word_size"] == 64
        assert result["endianness"] == "Little-endian"

    def test_explain_architecture_arm64(self, ae):
        result = ae.explain_architecture("arm64")
        assert result["word_size"] == 64

    def test_explain_architecture_riscv(self, ae):
        result = ae.explain_architecture("risc-v")
        assert "RISC" in result["instruction_set"]

    def test_explain_unknown_arch(self, ae):
        result = ae.explain_architecture("mips")
        assert result.get("found") is False

    def test_compilation_pipeline_c(self, ae):
        result = ae.explain_compilation_pipeline("C")
        assert len(result["pipeline"]) >= 6
        stages = [s["name"] for s in result["pipeline"]]
        assert "Source Code" in stages
        assert "Assembly" in stages

    def test_fetch_decode_execute(self, ae):
        result = ae.fetch_decode_execute_cycle()
        assert "stages" in result
        stage_names = [s["stage"] for s in result["stages"]]
        assert "FETCH" in stage_names
        assert "DECODE" in stage_names
        assert "EXECUTE" in stage_names

    def test_educational_cpu_model(self, ae):
        result = ae.educational_cpu_model()
        assert "components" in result
        assert "ALU" in result["components"]
        assert "Registers" in result["components"]

    def test_explain_opcode(self, ae):
        result = ae.explain_opcode()
        assert "opcode" in result["name"].lower()
        assert "key_points" in result

    def test_list_instructions(self, ae):
        instructions = ae.list_instructions()
        assert "MOV" in instructions
        assert "ADD" in instructions
        assert "CALL" in instructions
        assert "RET" in instructions


# ===========================================================================
# 11. COMPUTER CONTROLLER (Integration)
# ===========================================================================

class TestComputerController:

    def test_controller_decimal_to_binary(self, ctrl):
        result = ctrl.handle("Convert 42 to binary")
        assert result["success"] is True
        assert "101010" in result["summary"]

    def test_controller_binary_to_decimal(self, ctrl):
        result = ctrl.handle("What is 101010 in decimal?")
        assert result["success"] is True
        assert "42" in result["summary"]

    def test_controller_255_to_hex(self, ctrl):
        result = ctrl.handle("Convert 255 to hex")
        assert result["success"] is True
        assert "FF" in result["summary"]

    def test_controller_FF_to_decimal(self, ctrl):
        result = ctrl.handle("What is FF in decimal?")
        assert result["success"] is True
        assert "255" in result["summary"]

    def test_controller_bitwise_and(self, ctrl):
        result = ctrl.handle("1010 AND 1100")
        assert result["success"] is True
        assert "1000" in result["summary"]

    def test_controller_bitwise_or(self, ctrl):
        result = ctrl.handle("1010 OR 1100")
        assert result["success"] is True
        assert "1110" in result["summary"]

    def test_controller_bitwise_xor(self, ctrl):
        result = ctrl.handle("1010 XOR 1100")
        assert result["success"] is True
        assert "0110" in result["summary"]

    def test_controller_not_4bit(self, ctrl):
        result = ctrl.handle("NOT 1010 using 4 bits")
        assert result["success"] is True
        assert "0101" in result["summary"]

    def test_controller_ascii_A(self, ctrl):
        result = ctrl.handle("Convert \"A\" to binary")
        assert result["success"] is True
        assert "01000001" in result["summary"]

    def test_controller_ascii_65(self, ctrl):
        result = ctrl.handle("ASCII 65")
        assert result["success"] is True
        assert "A" in result["summary"]

    def test_controller_twos_complement(self, ctrl):
        result = ctrl.handle("represent -5 using 8-bit two's complement")
        assert result["success"] is True
        assert "11111011" in result["summary"]

    def test_controller_verified_label(self, ctrl):
        result = ctrl.handle("Convert 42 to binary")
        assert "Computer Systems Verified" in result["label"]

    def test_controller_verified_flag(self, ctrl):
        result = ctrl.handle("1010 AND 1100")
        assert result["verified"] is True

    def test_controller_sub_domain_binary_conversion(self, ctrl):
        result = ctrl.handle("Convert 25 to binary")
        assert result["sub_domain"] == "binary_conversion"

    def test_controller_sub_domain_bitwise(self, ctrl):
        result = ctrl.handle("1010 AND 1100")
        assert result["sub_domain"] == "bitwise_ops"

    def test_controller_sub_domain_ascii(self, ctrl):
        result = ctrl.handle("ASCII of A")
        assert result["sub_domain"] == "ascii_encoding"


# ===========================================================================
# 12. CLASSIFIER ROUTING
# ===========================================================================

class TestClassifierRouting:

    def test_classify_binary_conversion(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("Convert 255 to binary")
        assert intent.intent == "computer_systems"
        assert intent.primary_tool == "computer"

    def test_classify_hex_query(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("What is 0xFF in decimal?")
        assert intent.intent == "computer_systems"

    def test_classify_bitwise(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("1010 AND 1100")
        assert intent.intent == "computer_systems"

    def test_classify_twos_complement(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("What is two's complement?")
        assert intent.intent == "computer_systems"

    def test_classify_cpu_register(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("What is a CPU register?")
        assert intent.intent == "computer_systems"

    def test_classify_assembly(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("Explain assembly language")
        assert intent.intent == "computer_systems"

    def test_classify_binary_to_decimal(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("What is 101010 in decimal?")
        assert intent.intent == "computer_systems"

    def test_classify_ascii_conversion(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("Convert Hello to binary")
        assert intent.intent == "computer_systems"

    def test_classify_requires_tool_true(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("Convert 42 to binary")
        assert intent.requires_tool is True

    def test_classify_requires_verification_true(self):
        from axen.agent.classifier import TaskClassifier
        clf = TaskClassifier()
        intent = clf.analyze_intent("1010 AND 1100")
        assert intent.requires_verification is True


# ===========================================================================
# 13. INVALID INPUT AND EDGE CASES
# ===========================================================================

class TestInvalidInputs:

    def test_invalid_binary_string(self, bc):
        with pytest.raises(ValueError):
            bc.binary_to_decimal("12345")  # 2,3,4,5 not valid binary

    def test_invalid_binary_string_letters(self, bc):
        with pytest.raises(ValueError):
            bc.binary_to_decimal("ABCD")

    def test_invalid_hex_string(self, hc):
        with pytest.raises(ValueError):
            hc.hex_to_decimal("ZZZZ")

    def test_invalid_hex_arithmetic(self, hc):
        with pytest.raises(ValueError):
            hc.hex_add("GG", "0F")

    def test_empty_binary_raises(self, bc):
        with pytest.raises(ValueError):
            bc.binary_to_decimal("")

    def test_negative_to_unsigned_raises(self, ir):
        with pytest.raises(ValueError):
            ir.to_unsigned(-1, bits=8)

    def test_overflow_8bit_signed(self, ir):
        with pytest.raises(OverflowError):
            ir.to_twos_complement(200, bits=8)  # max signed 8-bit is 127

    def test_overflow_8bit_unsigned(self, ir):
        with pytest.raises(OverflowError):
            ir.to_unsigned(256, bits=8)  # max unsigned 8-bit is 255

    def test_invalid_bit_width_not(self, bw):
        with pytest.raises(ValueError):
            bw.NOT(0b1010, bits=3)

    def test_empty_ascii_string_raises(self, at):
        with pytest.raises(ValueError):
            at.string_to_binary("")

    def test_invalid_octal(self, bc):
        with pytest.raises(ValueError):
            bc.octal_to_binary("89")  # 8 and 9 are not octal digits

    def test_hex_divide_by_zero(self, hc):
        with pytest.raises(ZeroDivisionError):
            hc.hex_divide("FF", "00")

    def test_binary_add_invalid_input(self, bc):
        with pytest.raises(ValueError):
            bc.binary_add("1010", "1290")  # 9 not valid binary

    def test_extract_bits_high_less_than_low(self, bw):
        with pytest.raises(ValueError):
            bw.extract_bits(0b1111, high=2, low=5)  # high < low

    def test_invalid_memory_byte_value(self, mc):
        with pytest.raises(ValueError):
            mc.format_memory_dump([300])  # > 255

    def test_endianness_invalid_bits(self, mc):
        with pytest.raises(ValueError):
            mc.explain_endianness(42, bits=10)  # must be 16, 32, or 64


# ===========================================================================
# 14. TOOL REGISTRY INTEGRATION
# ===========================================================================

class TestToolRegistry:

    def test_build_computer_tool_returns_definition(self):
        from axen.tools.computer.computer_controller import build_computer_tool
        tool = build_computer_tool()
        assert tool is not None
        assert tool.name == "computer.query"
        assert tool.risk == "low"
        assert tool.enabled is True

    def test_computer_tool_handler(self):
        from axen.tools.computer.computer_controller import build_computer_tool
        tool = build_computer_tool()
        result = tool.handler({"request": "Convert 42 to binary"})
        assert result["success"] is True
        assert "101010" in result["summary"]

    def test_computer_tool_registered_in_registry(self):
        from axen.tools import build_default_registry
        registry = build_default_registry()
        assert "computer.query" in registry.names()

    def test_computer_tool_call_via_registry(self):
        from axen.tools import build_default_registry
        registry = build_default_registry()
        result = registry.call("computer.query", {"request": "Convert 255 to binary"})
        assert result.success is True
        assert "11111111" in result.output["summary"]


# ===========================================================================
# 15. ADDITIONAL VERIFICATION TESTS
# ===========================================================================

class TestAdditionalVerification:
    """Additional tests to ensure correctness of edge cases and specific examples."""

    def test_25_to_binary(self, bc):
        # From spec: 25 → 11001
        assert bc.decimal_to_binary(25) == "11001"

    def test_255_to_binary(self, bc):
        # From spec: 255 → 11111111
        assert bc.decimal_to_binary(255) == "11111111"

    def test_255_to_hex(self, hc):
        # From spec: 255 → FF
        assert hc.decimal_to_hex(255) == "FF"

    def test_hex_FF_to_255(self, hc):
        assert hc.hex_to_decimal("FF") == 255

    def test_shift_left_1_by_3(self, bw):
        # 1 << 3 = 8
        result = bw.left_shift(1, 3)
        assert result["result_decimal"] == 8

    def test_shift_right_8_by_2(self, bw):
        # 8 >> 2 = 2
        result = bw.right_shift(8, 2)
        assert result["result_decimal"] == 2

    def test_mask_11110000_on_byte(self, bw):
        # Apply mask 0xF0 to 0xCC → 0xC0
        result = bw.mask_bits(0xCC, 0xF0)
        assert result["result_decimal"] == 0xC0

    def test_controller_25_to_binary(self, ctrl):
        result = ctrl.handle("Convert 25 to binary")
        assert result["success"] is True
        assert "11001" in result["summary"]

    def test_controller_42_to_binary(self, ctrl):
        result = ctrl.handle("Convert 42 to binary")
        assert result["success"] is True
        assert "101010" in result["summary"]

    def test_controller_255_to_binary(self, ctrl):
        result = ctrl.handle("Convert 255 to binary")
        assert result["success"] is True
        assert "11111111" in result["summary"]

    def test_twos_complement_shows_invert_step(self, ir):
        result = ir.to_twos_complement(-5, bits=8, show_steps=True)
        # Step 3 should mention "Invert all bits"
        steps_text = " ".join(result["steps"])
        assert "Invert" in steps_text or "invert" in steps_text

    def test_twos_complement_add1_step(self, ir):
        result = ir.to_twos_complement(-5, bits=8, show_steps=True)
        steps_text = " ".join(result["steps"])
        assert "Add 1" in steps_text or "+1" in steps_text or "add 1" in steps_text.lower()

    def test_binary_add_step_count_non_trivial(self, bc):
        result = bc.binary_add("1010", "0110", show_steps=True)
        assert len(result["steps"]) >= 4

    def test_positional_breakdown_correctness(self, ns):
        # "1010" in binary: 1×8 + 0×4 + 1×2 + 0×1 = 10
        bd = ns.positional_breakdown("1010", "binary")
        total = sum(d["contribution"] for d in bd)
        assert total == 10

    def test_endianness_big_msb_first(self, mc):
        # 0x12345678: big-endian MSB first = 12
        result = mc.explain_endianness(0x12345678, bits=32)
        assert result["big_endian"]["bytes"][0]["byte"] == "12"

    def test_endianness_little_lsb_first(self, mc):
        # 0x12345678: little-endian LSB first = 78
        result = mc.explain_endianness(0x12345678, bits=32)
        assert result["little_endian"]["bytes"][0]["byte"] == "78"


# ===========================================================================
# AUDIT REGRESSION: TWO'S COMPLEMENT, BIT WIDTH, OVERFLOW, ROUND-TRIP
# ===========================================================================

class TestAuditRegression:

    def test_negative_twos_complement_8bit(self, ir):
        """Regression: -5 in 8 bits must be 11111011 (never 11111111011)."""
        expected = {
            -5: "11111011",
            -1: "11111111",
            -2: "11111110",
            -127: "10000001",
            -128: "10000000",
            0: "00000000",
            1: "00000001",
            2: "00000010",
            5: "00000101",
            127: "01111111",
        }
        for val, exp_bin in expected.items():
            res = ir.to_twos_complement(val, bits=8)
            assert res["binary"] == exp_bin
            assert len(res["binary"]) == 8
            assert res["binary"] != "11111111011"

    def test_bit_widths_4_8_16_32_64(self, ir):
        """Regression: bit widths must return exactly requested bit count."""
        widths = [4, 8, 16, 32, 64]
        for w in widths:
            res = ir.to_twos_complement(-5, bits=w)
            assert len(res["binary"]) == w
            dec = ir.from_twos_complement(res["binary"], bits=w)
            assert dec["decimal"] == -5

    def test_positive_numbers_width(self, ir):
        """Positive numbers must remain normal binary with requested width."""
        res5 = ir.to_twos_complement(5, bits=8)
        assert res5["binary"] == "00000101"
        assert len(res5["binary"]) == 8

        res127 = ir.to_twos_complement(127, bits=8)
        assert res127["binary"] == "01111111"
        assert len(res127["binary"]) == 8

    def test_overflow_protection(self, ir):
        """Values exceeding range must raise OverflowError."""
        with pytest.raises(OverflowError):
            ir.to_twos_complement(128, bits=8)
        with pytest.raises(OverflowError):
            ir.to_twos_complement(-129, bits=8)

    def test_all_8bit_round_trip(self, ir):
        """Regression: decode(encode(x, 8), 8) == x for every -128 <= x <= 127."""
        for x in range(-128, 128):
            enc = ir.to_twos_complement(x, bits=8)["binary"]
            assert len(enc) == 8
            dec = ir.from_twos_complement(enc, bits=8)["decimal"]
            assert dec == x

    def test_controller_twos_complement_various_phrasings(self, ctrl):
        """Test user queries with various natural phrasings."""
        r1 = ctrl.handle("Two's complement of -5 in 8 bits")
        assert r1["success"] is True
        assert r1["result"]["binary"] == "11111011"
        assert "11111011" in r1["summary"]
        assert "11111111011" not in r1["summary"]

        r2 = ctrl.handle("Two's complement -5 using 8 bits")
        assert r2["success"] is True
        assert r2["result"]["binary"] == "11111011"

        r3 = ctrl.handle("Convert 101010 to decimal")
        assert r3["success"] is True
        assert r3["result"]["decimal"] == 42

        r4 = ctrl.handle("Convert FF to decimal")
        assert r4["success"] is True
        assert r4["result"]["decimal"] == 255

        r5 = ctrl.handle("Explain 0x12345678 endianness")
        assert r5["success"] is True
        assert r5["sub_domain"] == "memory_concepts"

