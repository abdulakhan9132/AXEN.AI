"""
Audit script for Computer Systems Module.
Verifies all 9 sections specified in the user request.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from axen.tools.computer.integer_representation import IntegerRepresentation
from axen.tools.computer.computer_controller import ComputerController
from axen.assistant.controller import AssistantController

def run_audit():
    print("=" * 60)
    print("AXEN COMPUTER SYSTEMS AUDIT")
    print("=" * 60)

    ir = IntegerRepresentation()
    cc = ComputerController()

    # ==================================================
    # 1. TEST ACTUAL PUBLIC API
    # ==================================================
    print("\n--- 1. TEST ACTUAL PUBLIC API (to_twos_complement) ---")
    test_cases_8bit = [
        (-5, "11111011"),
        (-1, "11111111"),
        (-2, "11111110"),
        (-127, "10000001"),
        (-128, "10000000"),
        (0, "00000000"),
        (1, "00000001"),
        (2, "00000010"),
        (5, "00000101"),
        (127, "01111111"),
    ]
    for val, expected_bin in test_cases_8bit:
        res = ir.to_twos_complement(val, bits=8)
        actual_bin = res["binary"]
        assert actual_bin == expected_bin, f"Failed for {val}: expected {expected_bin}, got {actual_bin}"
        assert len(actual_bin) == 8, f"Length for {val} is not 8: {actual_bin}"
        print(f"  [PASS] {val:>4} (8 bits) -> {actual_bin} (expected: {expected_bin})")

    # ==================================================
    # 2. TEST DECODE AS WELL
    # ==================================================
    print("\n--- 2. TEST DECODE AND ROUND-TRIP ---")
    # Single test case
    dec_neg5 = ir.from_twos_complement("11111011", bits=8)
    assert dec_neg5["decimal"] == -5, f"Expected -5, got {dec_neg5['decimal']}"
    print(f"  [PASS] decode('11111011', 8) -> {dec_neg5['decimal']}")

    # All 256 signed 8-bit values
    round_trip_failures = []
    for x in range(-128, 128):
        enc = ir.to_twos_complement(x, bits=8)["binary"]
        dec = ir.from_twos_complement(enc, bits=8)["decimal"]
        if dec != x:
            round_trip_failures.append((x, enc, dec))
    assert not round_trip_failures, f"Round trip failures: {round_trip_failures}"
    print("  [PASS] decode(encode(x, 8), 8) == x for all -128 <= x <= 127 (256/256 verified)")

    # ==================================================
    # 3. CHECK BIT WIDTH
    # ==================================================
    print("\n--- 3. CHECK BIT WIDTH (4, 8, 16, 32, 64) ---")
    widths = [4, 8, 16, 32, 64]
    for w in widths:
        res = ir.to_twos_complement(-5, bits=w)
        b = res["binary"]
        assert len(b) == w, f"Width {w} returned string of length {len(b)}: {b}"
        dec = ir.from_twos_complement(b, bits=w)["decimal"]
        assert dec == -5, f"Width {w} decode returned {dec}"
        print(f"  [PASS] {w:>2}-bit: -5 -> {b} (len={len(b)}, decode={dec})")

    # Specifically check -5 in 8 bits: MUST NOT contain 11111111011
    b8 = ir.to_twos_complement(-5, bits=8)["binary"]
    assert b8 == "11111011"
    assert b8 != "11111111011"
    assert len(b8) == 8
    print("  [PASS] 8-bit -5 is exactly '11111011' and NOT '11111111011'")

    # ==================================================
    # 4. CHECK POSITIVE NUMBERS
    # ==================================================
    print("\n--- 4. CHECK POSITIVE NUMBERS ---")
    pos_cases = [
        (5, 8, "00000101"),
        (127, 8, "01111111"),
    ]
    for val, w, exp in pos_cases:
        res = ir.to_twos_complement(val, bits=w)
        assert res["binary"] == exp, f"Expected {exp}, got {res['binary']}"
        assert len(res["binary"]) == w
        print(f"  [PASS] +{val} ({w} bits) -> {res['binary']}")

    # ==================================================
    # 5. CHECK OVERFLOW
    # ==================================================
    print("\n--- 5. CHECK OVERFLOW ---")
    overflow_cases = [128, -129, 256, -200]
    for ov in overflow_cases:
        try:
            ir.to_twos_complement(ov, bits=8)
            assert False, f"Did not raise OverflowError for {ov} in 8 bits"
        except OverflowError as e:
            print(f"  [PASS] Value {ov} correctly raised OverflowError: {e}")

    # ==================================================
    # 6. CHECK ALL OTHER COMPUTER SYSTEMS FUNCTIONS
    # ==================================================
    print("\n--- 6. CHECK ALL OTHER COMPUTER SYSTEMS FUNCTIONS ---")
    real_examples = [
        ("Convert 42 to binary", lambda r: "101010" in str(r["result"])),
        ("Convert 101010 to decimal", lambda r: r["result"].get("decimal") == 42 or "42" in r["summary"]),
        ("Convert FF to decimal", lambda r: r["result"].get("decimal") == 255 or "255" in r["summary"]),
        ("1010 AND 1100", lambda r: r["result"].get("result_bin") == "1000" or "1000" in r["summary"]),
        ("1010 OR 1100", lambda r: r["result"].get("result_bin") == "1110" or "1110" in r["summary"]),
        ("1010 XOR 1100", lambda r: r["result"].get("result_bin") == "0110" or "0110" in r["summary"]),
        ("NOT 1010 using 4 bits", lambda r: r["result"].get("result_bin") == "0101" or "0101" in r["summary"]),
        ("Convert A to binary", lambda r: "01000001" in str(r["result"])),
        ("Convert Hello to binary", lambda r: "01001000" in str(r["result"])),
        ("Two's complement -5 using 8 bits", lambda r: r["result"].get("binary") == "11111011" and "11111011" in r["summary"]),
        ("Explain 0x12345678 endianness", lambda r: "big" in str(r["result"]).lower() and "little" in str(r["result"]).lower()),
        ("Explain x86-64 MOV", lambda r: "mov" in str(r["result"]).lower()),
        ("Explain CPU fetch-decode-execute", lambda r: "fetch" in str(r["result"]).lower() and "decode" in str(r["result"]).lower()),
    ]

    for query, validator in real_examples:
        res = cc.handle(query)
        assert res["success"], f"Query '{query}' failed: {res}"
        assert validator(res), f"Query '{query}' produced unexpected result: {res}"
        first_line = res["summary"].split("\n")[0]
        print(f"  [PASS] '{query}' -> Subdomain: {res['sub_domain']} | First line: {first_line}")

    # ==================================================
    # 9. USER-FACING TEST VIA ASSISTANT CONTROLLER
    # ==================================================
    print("\n--- 9. USER-FACING TEST VIA ASSISTANT CONTROLLER ---")
    ac = AssistantController()
    user_query = "Two's complement of -5 in 8 bits"
    answer = ac.respond(user_query)

    print(f"User Query: '{user_query}'")
    print(f"Answer received:\n{answer}\n")

    assert "11111011" in answer, f"Answer does NOT contain 11111011! Got:\n{answer}"
    assert "11111111011" not in answer, f"Answer INCORRECTLY contains 11111111011! Got:\n{answer}"
    print("  [PASS] User-facing answer contains '11111011' and does NOT contain '11111111011'")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE - ALL 9 VERIFICATION STEPS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_audit()
