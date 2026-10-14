# Operator-authored offline lesson: Python range boundaries

This is a local educational fixture written for the bounded debugging demo.
It is not a retrieved website or transcript.

Reference: https://docs.python.org/3/library/stdtypes.html#ranges

Python range has an exclusive stop boundary. For a positive step, values are
strictly less than stop. To include a nonnegative upper bound n, use range(n + 1).
Check zero, one, and other unseen bounds before adopting a repair.

Practice: repair an inclusive summation function whose loop excludes the upper
bound. A correct function returns the sum of the integers from zero through n.
