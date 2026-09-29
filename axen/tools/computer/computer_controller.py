"""
axen/tools/computer/computer_controller.py
===========================================
Central orchestrator for AXEN's Computer Systems domain.

Architecture:
    USER REQUEST
         ↓
    ComputerController.handle(request)
         ↓
    Intent sub-classification (binary/hex/bitwise/ascii/memory/assembly/...)
         ↓
    Deterministic Tool (binary.py, bitwise.py, ascii_tool.py, ...)
         ↓
    Structured Result
         ↓
    Natural language summary (formatted for chat)

The controller also provides build_computer_tool() which registers
the domain as a single ToolDefinition in AXEN's ToolRegistry.

INTEGRATION NOTE:
  Callers should call:
      result = controller.handle(user_request)
  which returns a dict with keys:
      "success", "result", "summary", "verified", "domain", "sub_domain"
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .binary import BinaryConverter
from .hexadecimal import HexConverter
from .bitwise import BitwiseOps
from .ascii_tool import AsciiTool
from .number_systems import NumberSystemConverter
from .integer_representation import IntegerRepresentation
from .memory import MemoryConcepts
from .assembly import AssemblyEducation

# Import ToolDefinition for registry integration
try:
    from axen.tools.tool_registry import ToolDefinition
except ImportError:
    # Fallback for direct/test usage
    ToolDefinition = None  # type: ignore


# ---------------------------------------------------------------------------
# ComputerController
# ---------------------------------------------------------------------------

class ComputerController:
    """
    Central router and orchestrator for all Computer Systems operations.

    Sub-domains handled:
      - binary_conversion: dec↔bin↔hex↔oct conversions
      - binary_arithmetic: binary add/sub/mul/div
      - bitwise_ops: AND/OR/XOR/NOT/shifts/set/clear/toggle/check
      - ascii_encoding: char/string ↔ binary/hex/decimal
      - hex_ops: hex arithmetic and bitwise
      - integer_repr: two's complement, signed/unsigned
      - memory_concepts: bytes, endianness, stack/heap, registers
      - assembly_edu: instruction explanations, FDE cycle, architectures
      - general_computer: multi-topic educational question
    """

    def __init__(self) -> None:
        self.binary = BinaryConverter()
        self.hex_conv = HexConverter()
        self.bitwise = BitwiseOps()
        self.ascii = AsciiTool()
        self.number_sys = NumberSystemConverter()
        self.int_repr = IntegerRepresentation()
        self.memory = MemoryConcepts()
        self.assembly = AssemblyEducation()

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def handle(self, request: str) -> dict:
        """
        Route a natural language computer-systems request to the right sub-tool.

        Returns:
            {
                "success": bool,
                "sub_domain": str,
                "result": dict | str,
                "summary": str,
                "verified": bool,
                "label": "Computer Systems Verified" | "Computer Systems"
            }
        """
        req = request.strip()
        lower = req.lower()

        try:
            sub_domain, result = self._dispatch(req, lower)
            summary = self._format_summary(sub_domain, result, req)
            return {
                "success": True,
                "sub_domain": sub_domain,
                "result": result,
                "summary": summary,
                "verified": True,
                "label": "Computer Systems Verified",
            }
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return {
                "success": False,
                "sub_domain": "error",
                "result": {},
                "summary": f"Computer Systems Error: {exc}",
                "verified": False,
                "label": "Computer Systems",
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "sub_domain": "error",
                "result": {},
                "summary": f"Computer Systems: unexpected error — {exc}",
                "verified": False,
                "label": "Computer Systems",
                "error": str(exc),
            }

    # -----------------------------------------------------------------------
    # Intent dispatch (deterministic pattern matching)
    # -----------------------------------------------------------------------

    def _dispatch(self, req: str, lower: str) -> tuple[str, Any]:
        """Route request to the appropriate sub-tool. Returns (sub_domain, result)."""

        # ------------------------------------------------------------------
        # 0. Two's complement / Signed Integer Representation & Decode
        # ------------------------------------------------------------------
        if "complement" in lower:
            # Decode: "decode [two's complement] 11111011 [in 8 bits]"
            decode_match = re.search(
                r"\bdecode\s+(?:two['\u2019s]*s?\s+complement\s+)?([01]+)(?:\s+(?:in|using)\s+(\d+)\s+bits?)?\b",
                lower,
            )
            if decode_match:
                b_str = decode_match.group(1)
                bits = int(decode_match.group(2)) if decode_match.group(2) else len(b_str)
                result = self.int_repr.from_twos_complement(b_str, bits=bits)
                return "integer_repr", result

            # "represent -5 using 8-bit two's complement", "what is -5 in 8-bit two's complement"
            twos_match1 = re.search(
                r"(?:represent|show|what\s+is)?\s*(-?\d+)\s+(?:using\s+)?(\d+)[-\s]bit\s+two['\u2019s]*s?\s+complement",
                lower,
            )
            if twos_match1:
                value = int(twos_match1.group(1))
                bits = int(twos_match1.group(2))
                result = self.int_repr.to_twos_complement(value, bits=bits)
                return "integer_repr", result

            # "two's complement of -5 in 8 bits", "two's complement -5 using 8 bits", "two's complement -5"
            twos_match2 = re.search(
                r"two['\u2019s]*s?\s+complement\s+(?:of\s+)?(-?\d+)(?:\s+(?:in|using)\s+(\d+)\s+bits?)?",
                lower,
            )
            if twos_match2:
                value = int(twos_match2.group(1))
                bits = int(twos_match2.group(2)) if twos_match2.group(2) else 8
                result = self.int_repr.to_twos_complement(value, bits=bits)
                return "integer_repr", result

            # "-5 in [8-bit] two's complement"
            twos_match3 = re.search(
                r"(-?\d+)\s+(?:in\s+)?(?:(\d+)[-\s]bit\s+)?two['\u2019s]*s?\s+complement",
                lower,
            )
            if twos_match3:
                value = int(twos_match3.group(1))
                bits = int(twos_match3.group(2)) if twos_match3.group(2) else 8
                result = self.int_repr.to_twos_complement(value, bits=bits)
                return "integer_repr", result

        # Decode binary string: "decode 11111011 in 8 bits"
        decode_gen_match = re.search(
            r"\bdecode\s+([01]{4,64})(?:\s+(?:in|using)\s+(\d+)[-\s]?bits?)?\b",
            lower,
        )
        if decode_gen_match:
            b_str = decode_gen_match.group(1)
            bits = int(decode_gen_match.group(2)) if decode_gen_match.group(2) else len(b_str)
            if bits in (4, 8, 16, 32, 64):
                result = self.int_repr.from_twos_complement(b_str, bits=bits)
                return "integer_repr", result

        # ------------------------------------------------------------------
        # 1. Explicit conversions: "convert X to binary/hex/octal/decimal"
        # ------------------------------------------------------------------
        convert_match = re.search(
            r"\bconvert\s+([0-9a-fA-F]+)\s+(?:from\s+\w+\s+)?to\s+(binary|decimal|hex(?:adecimal)?|octal)\b",
            lower,
        )
        if convert_match:
            raw_val = convert_match.group(1)
            target = convert_match.group(2)
            # If raw_val is a single letter (like 'A') and target is 'binary', without explicit hex keyword,
            # route to ASCII encoding (ASCII 65 = 01000001)
            if len(raw_val) == 1 and raw_val.isalpha() and target == "binary" and not re.search(r"\b(?:0x|hex|hexadecimal)\b", lower):
                pass
            else:
                source = self._infer_source_base(raw_val, req, lower, target=target)
                result = self.number_sys.convert(raw_val, from_base=source)
                result["requested_target"] = target
                return "binary_conversion", result

        # ------------------------------------------------------------------
        # 2. "What is X in decimal/binary/hex/octal?"
        # ------------------------------------------------------------------
        what_is_match = re.search(
            r"\bwhat\s+is\s+([0-9a-fA-F]+)\s+in\s+(decimal|binary|hex(?:adecimal)?|octal)\b",
            lower,
        )
        if what_is_match:
            raw_val = what_is_match.group(1)
            target = what_is_match.group(2)
            # Determine source base from the TARGET (opposite of target)
            # E.g. "What is 101010 in decimal?" → source = binary, target = decimal
            if target == "decimal":
                # The value must be in some other base; detect it
                detected = self.number_sys.detect_base(raw_val)
                if detected in ("binary", "binary_or_decimal") and all(c in "01" for c in raw_val):
                    source = "binary"
                elif detected == "hexadecimal" or not raw_val.isdigit():
                    source = "hexadecimal"
                else:
                    source = "binary"  # for pure 0/1 strings, assume binary
            elif target in ("binary", "hex", "hexadecimal", "octal"):
                # Source is likely decimal if it's just digits
                source = self._infer_source_base(raw_val, req, lower)
            else:
                source = self._infer_source_base(raw_val, req, lower)
            result = self.number_sys.convert(raw_val, from_base=source)
            result["requested_target"] = target
            return "binary_conversion", result

        # ------------------------------------------------------------------
        # 3. "X binary / X decimal / X hex / X octal"  (inline format)
        # ------------------------------------------------------------------
        # "101010 binary", "101010 in decimal", "FF in binary"
        base_suffix_match = re.search(
            r"\b([0-9a-fA-F]+)\s+(?:in\s+)?(binary|decimal|hex(?:adecimal)?|octal)\b",
            lower,
        )
        if base_suffix_match and not re.search(r"\b(and|or|xor|not|shift|set|clear|toggle)\b", lower):
            raw_val = base_suffix_match.group(1)
            # The second group tells us the TARGET format here
            claimed_target = base_suffix_match.group(2)
            if claimed_target.startswith("hex"):
                claimed_target = "hexadecimal"
            # Infer SOURCE from the value itself (opposite direction)
            if claimed_target == "decimal":
                # "101010 in decimal" — source is binary/hex/octal
                if all(c in "01" for c in raw_val):
                    source = "binary"
                elif any(c in "abcdefABCDEF" for c in raw_val):
                    source = "hexadecimal"
                else:
                    source = "binary"  # fallback
            elif claimed_target == "binary":
                # "FF in binary" — source is hex or decimal
                if any(c in "abcdefABCDEF" for c in raw_val):
                    source = "hexadecimal"
                elif raw_val.isdigit():
                    source = "decimal"
                else:
                    source = "hexadecimal"
            else:
                source = self._infer_source_base(raw_val, req, lower)

            try:
                result = self.number_sys.convert(raw_val, from_base=source)
                result["requested_target"] = claimed_target
                return "binary_conversion", result
            except ValueError:
                pass  # Fall through to next dispatch

        # ------------------------------------------------------------------
        # 4. Hex prefix: "0xFF", "0x1A"  — convert to all bases
        # ------------------------------------------------------------------
        hex_prefix_match = re.search(r"\b0[xX]([0-9a-fA-F]+)\b", req)
        if hex_prefix_match and not re.search(r"\b(and|or|xor|not|endian|endianness|memory|register|shift)\b", lower):
            raw_hex = hex_prefix_match.group(1)
            result = self.number_sys.convert(raw_hex, from_base="hexadecimal")
            return "binary_conversion", result

        # 4b. Bare hex string: "What is FF in decimal?", "FF in binary"
        # Only trigger if string has hex chars (A-F) and no 0-9 only ambiguity
        bare_hex_match = re.search(
            r"\b([0-9a-fA-F]{2,})\b",
            req
        )
        if bare_hex_match and re.search(r"\b(in\s+decimal|to\s+decimal|in\s+binary|to\s+binary)\b", lower):
            raw_val = bare_hex_match.group(1)
            # Only trigger if it has at least one hex letter
            if any(c.lower() in "abcdef" for c in raw_val):
                source = "hexadecimal"
                target_match = re.search(r"in\s+(decimal|binary|hex(?:adecimal)?|octal)|to\s+(decimal|binary|hex(?:adecimal)?|octal)", lower)
                target = (target_match.group(1) or target_match.group(2)) if target_match else "decimal"
                result = self.number_sys.convert(raw_val, from_base=source)
                result["requested_target"] = target
                return "binary_conversion", result

        # ------------------------------------------------------------------
        # 5. Bitwise operations: "1010 AND 1100", "1010 OR 1100", etc.
        # ------------------------------------------------------------------
        bitwise_binary_match = re.search(
            r"\b(?:0b)?([01]+)\s+(and|or|xor)\s+(?:0b)?([01]+)\b", lower
        )
        if bitwise_binary_match:
            a_str = bitwise_binary_match.group(1)
            op = bitwise_binary_match.group(2).upper()
            b_str = bitwise_binary_match.group(3)
            result = self.bitwise.from_binary_strings(op, a_str, b_str)
            return "bitwise_ops", result

        # NOT with bit width: "NOT 1010 using 4 bits"
        not_match = re.search(r"\bnot\s+([01]+)(?:\s+using\s+(\d+)\s+bits?)?\b", lower)
        if not_match:
            a_str = not_match.group(1)
            bits = int(not_match.group(2)) if not_match.group(2) else 8
            result = self.bitwise.from_binary_strings("NOT", a_str, bits=bits)
            return "bitwise_ops", result

        # Hex bitwise: "0xFF AND 0x0F"
        hex_bitwise_match = re.search(
            r"0[xX]([0-9a-fA-F]+)\s+(and|or|xor)\s+0[xX]([0-9a-fA-F]+)", req, re.I
        )
        if hex_bitwise_match:
            h_a = hex_bitwise_match.group(1)
            op = hex_bitwise_match.group(2).upper()
            h_b = hex_bitwise_match.group(3)
            if op == "AND":
                result = self.hex_conv.hex_and(h_a, h_b)
            elif op == "OR":
                result = self.hex_conv.hex_or(h_a, h_b)
            else:
                result = self.hex_conv.hex_xor(h_a, h_b)
            return "hex_ops", result

        # Left/right shift: "shift 1010 left 2", "1 << 3"
        shift_match = re.search(
            r"\b([01]+)\s*(?:<<|left\s+shift(?:ed)?\s*(?:by)?)\s*(\d+)\b", lower
        )
        if shift_match:
            val = int(shift_match.group(1), 2)
            n = int(shift_match.group(2))
            result = self.bitwise.left_shift(val, n)
            return "bitwise_ops", result

        rshift_match = re.search(
            r"\b([01]+)\s*(?:>>|right\s+shift(?:ed)?\s*(?:by)?)\s*(\d+)\b", lower
        )
        if rshift_match:
            val = int(rshift_match.group(1), 2)
            n = int(rshift_match.group(2))
            result = self.bitwise.right_shift(val, n)
            return "bitwise_ops", result

        # ------------------------------------------------------------------
        # 6. Bit manipulation: set/clear/toggle/check bit
        # ------------------------------------------------------------------
        set_bit_match = re.search(r"\bset\s+bit\s+(\d+)\s+(?:of\s+)?([01]+)\b", lower)
        if set_bit_match:
            bit_pos = int(set_bit_match.group(1))
            val = int(set_bit_match.group(2), 2)
            result = self.bitwise.set_bit(val, bit_pos)
            return "bit_manipulation", result

        clear_bit_match = re.search(r"\bclear\s+bit\s+(\d+)\s+(?:of\s+)?([01]+)?\b", lower)
        if clear_bit_match and clear_bit_match.group(2):
            bit_pos = int(clear_bit_match.group(1))
            val = int(clear_bit_match.group(2), 2)
            result = self.bitwise.clear_bit(val, bit_pos)
            return "bit_manipulation", result

        toggle_bit_match = re.search(r"\btoggle\s+bit\s+(\d+)\s+(?:of\s+)?([01]+)\b", lower)
        if toggle_bit_match:
            bit_pos = int(toggle_bit_match.group(1))
            val = int(toggle_bit_match.group(2), 2)
            result = self.bitwise.toggle_bit(val, bit_pos)
            return "bit_manipulation", result

        check_bit_match = re.search(r"\bcheck\s+(?:whether\s+)?bit\s+(\d+)\s+(?:of\s+|is\s+)?([01]+)?\b", lower)
        if check_bit_match and check_bit_match.group(2):
            bit_pos = int(check_bit_match.group(1))
            val = int(check_bit_match.group(2), 2)
            result = self.bitwise.check_bit(val, bit_pos)
            return "bit_manipulation", result

        # ------------------------------------------------------------------
        # 7. Binary arithmetic: "1010 + 0110", "add 1010 and 0110"
        # ------------------------------------------------------------------
        bin_arith_match = re.search(
            r"\b([01]+)\s*([\+\-\*\/])\s*([01]+)\b", req
        )
        if bin_arith_match:
            a_str = bin_arith_match.group(1)
            op = bin_arith_match.group(2)
            b_str = bin_arith_match.group(3)
            # Confirm both look like binary (not decimal)
            if all(c in "01" for c in a_str) and all(c in "01" for c in b_str):
                if op == "+":
                    result = self.binary.binary_add(a_str, b_str, show_steps=True)
                elif op == "-":
                    result = self.binary.binary_subtract(a_str, b_str)
                elif op == "*":
                    result = self.binary.binary_multiply(a_str, b_str)
                elif op == "/":
                    result = self.binary.binary_divide(a_str, b_str)
                return "binary_arithmetic", result

        # Keyword binary arithmetic: "add 101010 and 111"
        kw_arith_match = re.search(
            r"\b(?:add|plus)\s+([01]+)\s+(?:and|to)\s+([01]+)\b|\b(?:subtract|minus)\s+([01]+)\s+(?:from)\s+([01]+)\b|\b(?:multiply)\s+([01]+)\s+(?:by|times)\s+([01]+)\b",
            lower,
        )
        if kw_arith_match:
            if kw_arith_match.group(1):
                a_str, b_str = kw_arith_match.group(1), kw_arith_match.group(2)
                result = self.binary.binary_add(a_str, b_str, show_steps=True)
            elif kw_arith_match.group(3):
                b_str, a_str = kw_arith_match.group(3), kw_arith_match.group(4)
                result = self.binary.binary_subtract(a_str, b_str)
            else:
                a_str, b_str = kw_arith_match.group(5), kw_arith_match.group(6)
                result = self.binary.binary_multiply(a_str, b_str)
            return "binary_arithmetic", result

        # ------------------------------------------------------------------
        # 8. ASCII conversions: "convert 'A' to binary", "ASCII of 65", "ASCII of A"
        # ------------------------------------------------------------------
        # "ASCII of X" or "ASCII value of X"
        ascii_of_match = re.search(r"\bascii\s+(?:of|value\s+of|code\s+of)\s+([^\s]+)\b", lower)
        if ascii_of_match:
            token = ascii_of_match.group(1).strip("'\".,")
            if token.isdigit():
                result = self.ascii.decimal_to_char(int(token))
            elif len(token) == 1:
                result = self.ascii.char_to_info(token)
            else:
                # Multi-char: treat as string
                result = {
                    "text": token,
                    "characters": self.ascii.string_to_binary(token),
                    "compact_binary": self.ascii.string_to_binary_compact(token),
                }
            return "ascii_encoding", result

        # Convert string to binary: "convert Hello to binary"
        str_to_bin_match = re.search(
            r'\bconvert\s+["\']?([A-Za-z ]+)["\']?\s+to\s+binary\b', req, re.I
        )
        if str_to_bin_match:
            text = str_to_bin_match.group(1).strip()
            if len(text) > 1:
                result = {
                    "text": text,
                    "characters": self.ascii.string_to_binary(text),
                    "compact_binary": self.ascii.string_to_binary_compact(text),
                    "hex_encoding": self.ascii.string_to_hex(text),
                }
            else:
                result = self.ascii.char_to_info(text)
            return "ascii_encoding", result

        # Single char to binary: convert "A" to binary
        char_match = re.search(
            r'(?:convert\s+)?["\']([^"\']+)["\'](?:\s+(?:char|character))?\s+(?:to\s+)?(binary|hex(?:adecimal)?|decimal|ascii)\b',
            req, re.I
        )
        if char_match:
            char = char_match.group(1)
            result = self.ascii.char_to_info(char) if len(char) == 1 else {
                "text": char,
                "characters": self.ascii.string_to_binary(char),
                "compact_binary": self.ascii.string_to_binary_compact(char),
            }
            return "ascii_encoding", result

        # ASCII code point to char: "ASCII 65", "what character is 65"
        ascii_dec_match = re.search(
            r"\b(?:ascii|character\s+for|char\s+of)\s+(\d+)\b", lower
        )
        if ascii_dec_match:
            code = int(ascii_dec_match.group(1))
            result = self.ascii.decimal_to_char(code)
            return "ascii_encoding", result

        # ------------------------------------------------------------------
        # 9. Two's complement: "represent -5 using 8-bit two's complement"
        # ------------------------------------------------------------------
        twos_match = re.search(
            r"(?:represent|show|what\s+is)\s+(-?\d+)\s+(?:using\s+)?(\d+)[-\s]bit\s+two['\u2019s]*s?\s+complement",
            lower,
        )
        if twos_match:
            value = int(twos_match.group(1))
            bits = int(twos_match.group(2))
            result = self.int_repr.to_twos_complement(value, bits=bits)
            return "integer_repr", result

        # Simpler: "two's complement of -5 in 8 bits"
        twos_simple_match = re.search(
            r"two['\u2019s]*s?\s+complement\s+of\s+(-?\d+)(?:\s+in\s+(\d+)\s+bits?)?",
            lower,
        )
        if twos_simple_match:
            value = int(twos_simple_match.group(1))
            bits = int(twos_simple_match.group(2)) if twos_simple_match.group(2) else 8
            result = self.int_repr.to_twos_complement(value, bits=bits)
            return "integer_repr", result

        # "Represent 13 in 8 bits" / "show 13 in 8 bits"
        repr_bits_match = re.search(
            r"\b(?:represent|show|store|display)\s+(\d+)\s+in\s+(\d+)\s+bits?\b", lower
        )
        if repr_bits_match:
            value = int(repr_bits_match.group(1))
            bits = int(repr_bits_match.group(2))
            result = self.int_repr.to_unsigned(value, bits=bits)
            return "integer_repr", result

        # ------------------------------------------------------------------
        # 10. Memory concept explanations
        # ------------------------------------------------------------------
        if re.search(r"\b(endian(?:ness)?|big.endian|little.endian)\b", lower):
            # Try to extract a value if given
            val_match = re.search(r"\b(0[xX][0-9a-fA-F]+|\d+)\b", req)
            bits_match = re.search(r"(\d+)[-\s]?bit", lower)
            if val_match:
                val = int(val_match.group(1), 16) if val_match.group(1).startswith("0") and len(val_match.group(1)) > 1 and val_match.group(1)[1] in "xX" else int(val_match.group(1))
                b = int(bits_match.group(1)) if bits_match else 32
                result = self.memory.explain_endianness(val, bits=b)
            else:
                result = self.memory.explain_endianness(0x12345678, bits=32)
            return "memory_concepts", result

        if re.search(r"\b(stack|heap)\b", lower) and re.search(r"\b(what|explain|how|describe)\b", lower):
            if "heap" in lower and "stack" not in lower:
                result = self.memory.explain_heap()
            elif "stack" in lower and "heap" not in lower:
                result = self.memory.explain_stack()
            else:
                result = {"stack": self.memory.explain_stack(), "heap": self.memory.explain_heap()}
            return "memory_concepts", result

        if re.search(r"\b(register|registers|cpu\s+register)\b", lower) and re.search(r"\b(what|explain|how|describe)\b", lower):
            arch_match = re.search(r"\b(x86.64|x64|arm64|aarch64|risc.v)\b", lower)
            arch = arch_match.group(1) if arch_match else "x86-64"
            result = self.memory.explain_registers(arch=arch)
            return "memory_concepts", result

        if re.search(r"\b(byte|bit|nibble)\b", lower) and re.search(r"\b(what|explain|define|how)\b", lower):
            result = self.memory.bit_byte_info()
            return "memory_concepts", result

        if re.search(r"\b(how\s+is|how\s+does)\b.*\b(stored|represented)\b", lower) or re.search(r"\bstored?\s+in\s+(?:a\s+)?byte\b", lower):
            val_match = re.search(r"\b(\d+)\b", req)
            if val_match:
                result = self.memory.explain_byte_storage(int(val_match.group(1)))
                return "memory_concepts", result

        if re.search(r"\b(pointer|pointers)\b", lower):
            result = self.memory.explain_pointer()
            return "memory_concepts", result

        if re.search(r"\b(address\s+space|memory\s+address(?:es)?|addressable\s+memory)\b", lower):
            bits_match = re.search(r"(\d+)[-\s]?bit", lower)
            b = int(bits_match.group(1)) if bits_match else 32
            result = self.memory.explain_address_space(bits=b)
            return "memory_concepts", result

        # ------------------------------------------------------------------
        # 11. Assembly / architecture
        # ------------------------------------------------------------------
        if re.search(r"\b(assembly|assembler|asm)\b", lower) and re.search(r"\b(what|explain|how|describe|language)\b", lower):
            if re.search(r"\b(fetch|decode|execute|fde|instruction\s+cycle)\b", lower):
                result = self.assembly.fetch_decode_execute_cycle()
            elif re.search(r"\b(opcode|machine\s+code)\b", lower):
                result = self.assembly.explain_opcode()
            else:
                result = self.assembly.educational_cpu_model()
            return "assembly_edu", result

        # Specific instruction: "what does MOV do", "explain ADD instruction"
        instr_match = re.search(
            r"\b(?:what\s+does\s+|explain\s+|what\s+is\s+)?(MOV|ADD|SUB|MUL|DIV|CMP|JMP|CALL|RET|PUSH|POP|LOAD|STORE)\b",
            req, re.I
        )
        if instr_match:
            mnem = instr_match.group(1).upper()
            arch_match = re.search(r"\b(x86.64|x64|arm64|aarch64|risc.v)\b", lower)
            arch = arch_match.group(1) if arch_match else None
            result = self.assembly.explain_instruction(mnem, arch=arch)
            return "assembly_edu", result

        if re.search(r"\b(x86.64|arm64|aarch64|risc.v)\b", lower) and re.search(r"\b(arch|architecture|register|explain|describe|how)\b", lower):
            arch_match = re.search(r"\b(x86.64|x64|arm64|aarch64|risc.v)\b", lower)
            result = self.assembly.explain_architecture(arch_match.group(1))
            return "assembly_edu", result

        if re.search(r"\b(compilation\s+pipeline|how.*compiled|compile.*to|high.level.*assembly|assembly.*machine\s+code)\b", lower):
            lang_match = re.search(r"\b(C|C\+\+|Python|Java|JavaScript|Rust)\b", req, re.I)
            lang = lang_match.group(1) if lang_match else "C"
            result = self.assembly.explain_compilation_pipeline(lang)
            return "assembly_edu", result

        if re.search(r"\b(fetch.decode.execute|fde\s+cycle|instruction\s+cycle|cpu\s+execute|how.*cpu.*execute|cpu.*fetch)\b", lower):
            result = self.assembly.fetch_decode_execute_cycle()
            return "assembly_edu", result

        if re.search(r"\b(opcode|opcodes)\b", lower):
            result = self.assembly.explain_opcode()
            return "assembly_edu", result

        if re.search(r"\b(cpu\s+(?:architecture|model|simulation|emulator|abstraction))\b", lower):
            result = self.assembly.educational_cpu_model()
            return "assembly_edu", result

        # ------------------------------------------------------------------
        # 12. Number system info: "what is binary", "explain hexadecimal"
        # ------------------------------------------------------------------
        sys_match = re.search(
            r"\b(?:what\s+is|explain|describe|how\s+does)\s+(binary|hexadecimal|hex|octal|decimal)\b",
            lower,
        )
        if sys_match:
            sys_name = sys_match.group(1)
            result = self.number_sys.explain_system(sys_name)
            return "number_system_info", result

        # ------------------------------------------------------------------
        # 13. Signed/unsigned ranges
        # ------------------------------------------------------------------
        range_match = re.search(
            r"\b(?:range|min|max|limits?)\s+of\s+(\d+)[-\s]?bit\s+(signed|unsigned)\b", lower
        )
        if range_match:
            bits = int(range_match.group(1))
            signed = range_match.group(2) == "signed"
            if signed:
                result = self.int_repr.signed_range(bits)
            else:
                result = self.int_repr.unsigned_range(bits)
            return "integer_repr", result

        # ------------------------------------------------------------------
        # 14. Memory size conversions
        # ------------------------------------------------------------------
        mem_size_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(bit|byte|kb|mb|gb|tb|kilobyte|megabyte|gigabyte|terabyte)s?\s+(?:in|to|into|as)\s+(bit|byte|kb|mb|gb|tb|kilobyte|megabyte|gigabyte|terabyte)s?\b",
            lower,
        )
        if mem_size_match:
            val = float(mem_size_match.group(1))
            from_u = mem_size_match.group(2)
            to_u = mem_size_match.group(3)
            result = self.memory.convert_memory_size(val, from_u, to_u)
            return "memory_concepts", result

        # ------------------------------------------------------------------
        # 15. Direct decimal → binary / hex requests (most natural form)
        #     "Convert 42 to binary"  "42 decimal to binary"
        # ------------------------------------------------------------------
        dec_to_base = re.search(
            r"\b(\d+)\s*(?:decimal)?\s*(?:to|in|into|as)\s+(binary|hex(?:adecimal)?|octal)\b",
            lower,
        )
        if dec_to_base:
            val = int(dec_to_base.group(1))
            target = dec_to_base.group(2)
            result = self.number_sys.convert(val, from_base="decimal")
            result["requested_target"] = target
            return "binary_conversion", result

        # ------------------------------------------------------------------
        # 16. Fallback: look for bare number + likely binary context
        # ------------------------------------------------------------------
        bare_num_match = re.search(r"\b(\d+)\b", req)
        if bare_num_match and re.search(r"\b(binary|hex|bits?|bytes?|base)\b", lower):
            val = int(bare_num_match.group(1))
            result = self.number_sys.convert(val, from_base="decimal")
            return "binary_conversion", result

        # ------------------------------------------------------------------
        # 17. General computer education fallback
        # ------------------------------------------------------------------
        if re.search(r"\b(ram|random\s+access\s+memory|difference.*ram.*register|register.*ram)\b", lower):
            result = {
                "RAM": {
                    "full_name": "Random Access Memory",
                    "type": "Main memory",
                    "size": "Gigabytes (4 GB – 512 GB typical)",
                    "speed": "~50–100 ns access time",
                    "volatile": True,
                    "location": "External to the CPU, connected via memory bus",
                    "purpose": "Stores programs and data currently in use by the CPU.",
                },
                "Registers": {
                    "type": "On-chip CPU storage",
                    "size": "Bytes (e.g. 64-bit = 8 bytes per register)",
                    "speed": "< 1 ns (sub-clock-cycle access)",
                    "volatile": True,
                    "location": "Inside the CPU die",
                    "purpose": "Immediate working storage for the ALU — the fastest storage in the system.",
                },
                "hierarchy": (
                    "Speed hierarchy (fastest → slowest): "
                    "Registers → L1 Cache → L2 Cache → L3 Cache → RAM → SSD → HDD"
                ),
            }
            return "memory_concepts", result

        # Truly generic computer question — return educational overview
        return "general_computer", {
            "message": (
                "AXEN Computer Systems module can help with:\n"
                "- Binary, hex, octal, decimal conversions\n"
                "- Binary arithmetic (+, -, ×, ÷)\n"
                "- Bitwise operations (AND, OR, XOR, NOT, shifts)\n"
                "- ASCII / Unicode encoding\n"
                "- Two's complement integer representation\n"
                "- Memory concepts (endianness, stack, heap, registers)\n"
                "- Assembly language education (x86-64, ARM64, RISC-V)\n"
                "- CPU fetch-decode-execute cycle\n"
                "- Compilation pipeline"
            ),
            "request": req,
        }

    # -----------------------------------------------------------------------
    # Source base inference
    # -----------------------------------------------------------------------

    def _infer_source_base(self, raw_val: str, req: str, lower: str, target: Optional[str] = None) -> str:
        """Infer the source number base from context clues in the request."""
        if re.search(r"\bfrom\s+binary\b|\bin\s+binary\b|binary\s+number\b", lower):
            return "binary"
        if re.search(r"\bfrom\s+hex(?:adecimal)?\b|\bin\s+hex(?:adecimal)?\b", lower):
            return "hexadecimal"
        if re.search(r"\bfrom\s+octal\b|\bin\s+octal\b", lower):
            return "octal"
        if re.search(r"\bfrom\s+decimal\b|\bin\s+decimal\b", lower):
            return "decimal"

        # If converting TO decimal, infer source from characters
        if target == "decimal" or re.search(r"\b(?:to|in)\s+decimal\b", lower):
            if all(c in "01" for c in raw_val):
                return "binary"
            if any(c in "abcdefABCDEF" for c in raw_val):
                return "hexadecimal"

        # If converting TO binary, infer source from characters
        if target == "binary" or re.search(r"\b(?:to|in)\s+binary\b", lower):
            if any(c in "abcdefABCDEF" for c in raw_val):
                return "hexadecimal"
            if raw_val.isdigit():
                return "decimal"

        # Auto-detect
        detected = self.number_sys.detect_base(raw_val)
        if detected == "binary_or_decimal":
            if re.search(r"\bbinary\b", lower):
                return "binary"
            return "decimal"
        if detected in ("binary", "hexadecimal", "octal", "decimal"):
            return detected
        return "decimal"

    # -----------------------------------------------------------------------
    # Human-readable summary formatter
    # -----------------------------------------------------------------------

    def _format_summary(self, sub_domain: str, result: Any, req: str) -> str:
        """Format a result dict into a clean natural language summary."""

        if sub_domain == "binary_conversion":
            d = result.get("decimal", "?")
            b = result.get("binary", "?")
            h = result.get("hexadecimal", "?")
            o = result.get("octal", "?")
            target = result.get("requested_target", "")
            if target in ("binary",):
                return (
                    f"**{d}** decimal = **{b}** binary\n\n"
                    f"Also: 0x{h} (hex) | {o} (octal)"
                )
            if target in ("hex", "hexadecimal"):
                return (
                    f"**{d}** decimal = **0x{h}** hexadecimal\n\n"
                    f"Also: {b} (binary) | {o} (octal)"
                )
            if target == "decimal":
                src = result.get("source_value", "?")
                src_base = result.get("source_base", "")
                return (
                    f"**{src}** {src_base} = **{d}** decimal\n\n"
                    f"Also: {b} (binary) | 0x{h} (hex)"
                )
            return (
                f"**Decimal:** {d}\n"
                f"**Binary:** {b}\n"
                f"**Hexadecimal:** 0x{h}\n"
                f"**Octal:** {o}"
            )

        if sub_domain == "binary_arithmetic":
            op = result.get("operation", "")
            a = result.get("a", "")
            b = result.get("b", "")
            if "quotient" in result:
                q = result.get("quotient", "")
                r = result.get("remainder", "")
                dq = result.get("decimal_quotient", "")
                dr = result.get("decimal_remainder", "")
                return (
                    f"**Binary Division:**\n"
                    f"{a} ÷ {b}\n"
                    f"= **{q}** (quotient), **{r}** (remainder)\n"
                    f"= {dq} ÷ {result.get('decimal_b','?')} = {dq} remainder {dr} (decimal)"
                )
            res = result.get("result", "")
            dr = result.get("decimal_result", "")
            da = result.get("decimal_a", "")
            db = result.get("decimal_b", "")
            neg = result.get("is_negative", False)
            lines = [f"**Binary {op.title()}:**", f"  {a}", f"{'+ ' if op == 'addition' else '- '}{b}", "  " + "─" * max(len(a), len(b), len(res)), f"  {res}"]
            if result.get("steps"):
                lines += result["steps"]
            lines.append(f"\nDecimal check: {da} {'+' if op == 'addition' else '-'} {db} = {dr}")
            return "\n".join(lines)

        if sub_domain == "bitwise_ops":
            op = result.get("operation", "")
            res = result.get("result_bin", result.get("result_decimal", "?"))
            note = result.get("note", "")
            if "NOT" in op:
                a_bin = result.get("a_binary", "")
                return (
                    f"**{op}:**\n"
                    f"Input:  {a_bin}\n"
                    f"Result: {res}\n\n"
                    f"{note}"
                )
            a_bin = result.get("a_binary", "")
            b_bin = result.get("b_binary", "")
            return (
                f"**{op}:**\n"
                f"{a_bin}\n"
                f"{b_bin}\n"
                f"{'─' * max(len(a_bin), len(b_bin))}\n"
                f"{res}\n\n"
                f"{note}"
            )

        if sub_domain == "bit_manipulation":
            op = result.get("operation", "")
            orig = result.get("value_binary", "?")
            res = result.get("result_bin", "?")
            note = result.get("note", "")
            if "CHECK" in op:
                is_set = result.get("is_set", False)
                return f"**{op}:** {'SET (1)' if is_set else 'CLEAR (0)'}\n\nValue: {orig}\n{note}"
            return f"**{op}:**\nBefore: {orig}\nAfter:  {res}\n\n{note}"

        if sub_domain == "ascii_encoding":
            if "characters" in result:
                text = result.get("text", "")
                compact = result.get("compact_binary", "")
                chars = result.get("characters", [])
                lines = [f"**ASCII / Binary encoding of '{text}':**"]
                for c in chars:
                    lines.append(
                        f"  '{c['char']}' → decimal {c['decimal']} | binary {c['binary']} | hex {c['hexadecimal']}"
                    )
                lines.append(f"\n**Compact binary:** {compact}")
                return "\n".join(lines)
            char = result.get("char", "?")
            dec = result.get("decimal", "?")
            bin_r = result.get("binary", "?")
            hex_r = result.get("hexadecimal", "?")
            name = result.get("unicode_name", "")
            return (
                f"**Character:** '{char}'\n"
                f"**ASCII decimal:** {dec}\n"
                f"**Binary:** {bin_r}\n"
                f"**Hex:** {hex_r}\n"
                f"**Unicode name:** {name}"
            )

        if sub_domain == "integer_repr":
            if "twos_complement" in str(result) or "binary" in result:
                val = result.get("value", result.get("decimal", "?"))
                bits = result.get("bits", "?")
                bin_r = result.get("binary", "?")
                hex_r = result.get("hexadecimal")
                steps = result.get("steps", [])
                lines = [f"**{bits}-bit Two's Complement Representation of {val}:**"]
                if hex_r:
                    lines.append(f"Binary: {bin_r}  |  Hex: 0x{hex_r}")
                else:
                    lines.append(f"Binary: {bin_r}  |  Decimal: {val}")
                if steps:
                    lines.append("\n**Steps:**")
                    lines.extend(f"  {s}" for s in steps)
                return "\n".join(lines)
            # Range result
            t = result.get("type", "")
            b = result.get("bits", "?")
            mn = result.get("min", "?")
            mx = result.get("max", "?")
            return f"**{b}-bit {t} range:** {mn} to {mx}"

        if sub_domain in ("memory_concepts", "hex_ops"):
            # Structured dict — format key fields
            if isinstance(result, dict):
                if "name" in result:
                    lines = [f"**{result['name']}**"]
                    if "description" in result:
                        lines.append(result["description"])
                    for k, v in result.items():
                        if k not in ("name", "description") and isinstance(v, (str, list)):
                            if isinstance(v, list):
                                lines.append(f"\n**{k.replace('_', ' ').title()}:**")
                                for item in v:
                                    lines.append(f"  - {item}")
                            else:
                                lines.append(f"\n**{k.replace('_', ' ').title()}:** {v}")
                    return "\n".join(lines)
                elif "operation" in result:
                    op = result.get("operation", "")
                    res_val = result.get("result", "")
                    return f"**{op}:** {res_val}"
                else:
                    # Format as key-value summary
                    lines = []
                    for k, v in result.items():
                        lines.append(f"**{k.replace('_',' ').title()}:** {v}")
                    return "\n".join(lines)

        if sub_domain == "assembly_edu":
            if isinstance(result, dict):
                name = result.get("name", "")
                desc = result.get("description", "")
                lines = []
                if name:
                    lines.append(f"**{name}**")
                if desc:
                    lines.append(desc)
                # Add key points if present
                if "stages" in result:
                    lines.append("\n**Stages:**")
                    for s in result["stages"]:
                        stage_name = s.get("stage", s.get("name", ""))
                        stage_desc = s.get("description", "")
                        lines.append(f"  **{stage_name}:** {stage_desc}")
                if "examples" in result:
                    lines.append("\n**Architecture Examples:**")
                    for arch, ex in result["examples"].items():
                        lines.append(f"  {arch}: `{ex}`")
                if "key_points" in result:
                    lines.append("\n**Key Points:**")
                    for pt in result["key_points"]:
                        lines.append(f"  - {pt}")
                return "\n".join(lines)

        if sub_domain == "number_system_info":
            if isinstance(result, dict):
                name = result.get("name", "")
                digits = result.get("digits", "")
                use = result.get("use", "")
                ex = result.get("example", "")
                return (
                    f"**{name}**\n"
                    f"Digits: {digits}\n"
                    f"Use: {use}\n"
                    f"Example: {ex}"
                )

        if sub_domain == "general_computer":
            return result.get("message", str(result))

        # Default: pretty-print the dict
        if isinstance(result, dict):
            lines = []
            for k, v in result.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
            return "\n".join(lines) if lines else str(result)

        return str(result)


# ---------------------------------------------------------------------------
# Tool registration helper
# ---------------------------------------------------------------------------

def build_computer_tool():
    """Build and return a ToolDefinition for the Computer Systems domain."""
    controller = ComputerController()

    def handler(args: dict) -> dict:
        request = args.get("request", args.get("query", ""))
        if not request:
            return {
                "success": False,
                "error": "Missing 'request' argument.",
                "summary": "Please provide a computer systems query.",
            }
        return controller.handle(request)

    # Import here to allow standalone testing without full AXEN stack
    try:
        from axen.tools.tool_registry import ToolDefinition
        return ToolDefinition(
            name="computer.query",
            description=(
                "AXEN Computer Systems domain. Handles binary/hex/octal/decimal conversions, "
                "binary arithmetic, bitwise operations (AND/OR/XOR/NOT/shifts), ASCII encoding, "
                "two's complement integer representation, memory concepts (endianness, stack/heap, "
                "registers), assembly language education (x86-64, ARM64, RISC-V), and CPU concepts."
            ),
            handler=handler,
            input_schema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "Natural language computer systems query.",
                    }
                },
                "required": ["request"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "sub_domain": {"type": "string"},
                    "summary": {"type": "string"},
                    "verified": {"type": "boolean"},
                    "label": {"type": "string"},
                },
            },
            risk="low",
        )
    except ImportError:
        return None
