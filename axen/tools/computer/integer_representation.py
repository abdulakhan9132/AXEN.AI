"""
axen/tools/computer/integer_representation.py
===============================================
Signed/unsigned integer representation for AXEN.

Supports:
  - Unsigned integers: 8, 16, 32, 64-bit
  - Signed integers (two's complement): 8, 16, 32, 64-bit
  - Two's complement encoding (positive and negative)
  - Two's complement step-by-step explanation
  - Range checks and overflow detection
  - Sign extension

Two's Complement Rule:
  To represent -N in k bits:
    1. Start with |N| in binary
    2. Invert all bits (NOT)
    3. Add 1
    4. The result is the two's complement representation

  To decode a two's complement number:
    If MSB = 0 → positive → value is just the binary number
    If MSB = 1 → negative → flip bits, add 1, result is the magnitude, value is negative
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Supported widths
# ---------------------------------------------------------------------------

_WIDTHS = (4, 8, 16, 32, 64)


def _validate_width(bits: int) -> None:
    if bits not in _WIDTHS:
        raise ValueError(f"Bit width {bits} not supported. Choose from {_WIDTHS}.")


# ---------------------------------------------------------------------------
# IntegerRepresentation
# ---------------------------------------------------------------------------

class IntegerRepresentation:
    """
    Deterministic integer representation engine.

    Usage::
        ir = IntegerRepresentation()
        ir.to_twos_complement(-5, bits=8)
        # → {"value": -5, "bits": 8, "twos_complement": "11111011", "steps": [...]}

        ir.from_twos_complement("11111011", bits=8)
        # → {"binary": "11111011", "decimal": -5, "is_negative": True}
    """

    # -----------------------------------------------------------------------
    # Unsigned integers
    # -----------------------------------------------------------------------

    def unsigned_range(self, bits: int) -> dict:
        """Return the min/max range for an unsigned integer of 'bits' width."""
        _validate_width(bits)
        return {
            "type": "unsigned",
            "bits": bits,
            "min": 0,
            "max": (1 << bits) - 1,
            "total_values": 1 << bits,
        }

    def to_unsigned(self, value: int, bits: int) -> dict:
        """Represent a non-negative integer as an unsigned integer of 'bits' width."""
        _validate_width(bits)
        if value < 0:
            raise ValueError(f"Unsigned integers cannot represent negative values. Use two's complement.")
        max_val = (1 << bits) - 1
        if value > max_val:
            raise OverflowError(
                f"Value {value} overflows {bits}-bit unsigned integer (max: {max_val})."
            )
        return {
            "type": "unsigned",
            "decimal": value,
            "bits": bits,
            "binary": bin(value)[2:].zfill(bits),
            "hexadecimal": hex(value)[2:].upper().zfill(bits // 4),
            "range": {"min": 0, "max": max_val},
        }

    # -----------------------------------------------------------------------
    # Signed integers (two's complement)
    # -----------------------------------------------------------------------

    def signed_range(self, bits: int) -> dict:
        """Return the min/max range for a signed two's complement integer."""
        _validate_width(bits)
        return {
            "type": "signed (two's complement)",
            "bits": bits,
            "min": -(1 << (bits - 1)),
            "max": (1 << (bits - 1)) - 1,
            "total_values": 1 << bits,
        }

    def to_twos_complement(self, value: int, bits: int = 8, show_steps: bool = True) -> dict:
        """
        Represent a signed integer using two's complement notation.

        Works for both positive and negative values:
          - Positive: just zfill to 'bits', MSB = 0
          - Negative: invert bits of |value| then add 1

        Returns a dict with binary, hex, steps.
        """
        _validate_width(bits)
        min_val = -(1 << (bits - 1))
        max_val = (1 << (bits - 1)) - 1

        if not (min_val <= value <= max_val):
            raise OverflowError(
                f"Value {value} overflows {bits}-bit signed range [{min_val}, {max_val}]."
            )

        if value >= 0:
            binary = bin(value)[2:].zfill(bits)
            steps = [
                f"Value {value} is positive.",
                f"Binary of {value} = {bin(value)[2:]}",
                f"Zero-padded to {bits} bits: {binary}",
                f"MSB = 0 (positive sign bit)",
            ]
        else:
            magnitude = abs(value)
            mag_bin = bin(magnitude)[2:].zfill(bits)
            # Invert bits
            inverted = "".join("1" if b == "0" else "0" for b in mag_bin)
            # Add 1
            inverted_int = int(inverted, 2)
            result_int = (inverted_int + 1) & ((1 << bits) - 1)
            binary = bin(result_int)[2:].zfill(bits)
            steps = [
                f"Value {value} is negative -> use two's complement.",
                f"Step 1: Take magnitude |{value}| = {magnitude}",
                f"Step 2: Write {magnitude} in {bits}-bit binary: {mag_bin}",
                f"Step 3: Invert all bits (NOT): {inverted}",
                f"Step 4: Add 1: {inverted} + 1 = {binary}",
                f"Result: {binary}",
                f"MSB = 1 (negative sign bit)",
            ]

        # Mask to ensure correct width
        int_value = int(binary, 2)
        hex_value = hex(int_value)[2:].upper().zfill(bits // 4)

        result: dict = {
            "value": value,
            "bits": bits,
            "type": "signed (two's complement)",
            "binary": binary,
            "hexadecimal": hex_value,
        }
        if show_steps:
            result["steps"] = steps
        return result

    def from_twos_complement(self, binary_str: str, bits: Optional[int] = None) -> dict:
        """
        Decode a two's complement binary string to its signed decimal value.

        If MSB = 0 → positive
        If MSB = 1 → negative: flip bits, add 1, negate
        """
        clean = binary_str.strip()
        if bits is None:
            bits = len(clean) if len(clean) in _WIDTHS else 8
        _validate_width(bits)
        b = clean.zfill(bits)
        if not all(c in "01" for c in b):
            raise ValueError(f"Invalid binary string: '{binary_str}'")
        if len(b) > bits:
            b = b[-bits:]  # keep LSB 'bits' worth

        msb = int(b[0])
        raw_int = int(b, 2)

        if msb == 0:
            # Positive
            decimal = raw_int
            steps = [
                f"MSB = 0 -> positive number.",
                f"Decimal value = {decimal}",
            ]
        else:
            # Negative two's complement
            # Method: value = raw_int - 2^bits
            decimal = raw_int - (1 << bits)
            # Step-by-step verification
            inverted = "".join("1" if c == "0" else "0" for c in b)
            magnitude = int(inverted, 2) + 1
            steps = [
                f"MSB = 1 -> negative number (two's complement).",
                f"Step 1: Invert bits of {b}: {inverted}",
                f"Step 2: Add 1: {int(inverted, 2)} + 1 = {magnitude}",
                f"Step 3: Value = -{magnitude} = {decimal}",
            ]

        return {
            "binary": b,
            "bits": bits,
            "is_negative": msb == 1,
            "decimal": decimal,
            "steps": steps,
        }

    # -----------------------------------------------------------------------
    # Sign extension
    # -----------------------------------------------------------------------

    def sign_extend(self, binary_str: str, from_bits: int, to_bits: int) -> dict:
        """
        Extend a signed two's complement value from from_bits to to_bits.

        Copies the MSB (sign bit) into the new high-order bits.
        from_bits can be any positive integer.
        to_bits must be a standard width: 8, 16, 32, 64.
        """
        if from_bits <= 0:
            raise ValueError(f"from_bits must be > 0, got {from_bits}")
        _validate_width(to_bits)
        if to_bits <= from_bits:
            raise ValueError(f"to_bits ({to_bits}) must be larger than from_bits ({from_bits}).")

        b = binary_str.strip().zfill(from_bits)
        msb = b[0]  # sign bit
        extended = msb * (to_bits - from_bits) + b

        # Decode to verify
        result = self.from_twos_complement(extended, bits=to_bits)

        return {
            "original_binary": b,
            "original_bits": from_bits,
            "extended_binary": extended,
            "target_bits": to_bits,
            "sign_bit": msb,
            "decimal_value": result["decimal"],
            "note": f"Sign bit ({msb}) replicated to fill {to_bits - from_bits} new high-order bits.",
        }

    # -----------------------------------------------------------------------
    # Overflow detection
    # -----------------------------------------------------------------------

    def check_overflow(self, value: int, bits: int, signed: bool = True) -> dict:
        """Check if value can be represented within bits-width integer."""
        _validate_width(bits)
        if signed:
            min_val = -(1 << (bits - 1))
            max_val = (1 << (bits - 1)) - 1
        else:
            min_val = 0
            max_val = (1 << bits) - 1

        overflows = not (min_val <= value <= max_val)
        return {
            "value": value,
            "bits": bits,
            "type": "signed" if signed else "unsigned",
            "min_representable": min_val,
            "max_representable": max_val,
            "overflows": overflows,
            "note": (
                f"Value {value} {'OVERFLOWS' if overflows else 'fits'} in "
                f"{bits}-bit {'signed' if signed else 'unsigned'} integer."
            ),
        }
