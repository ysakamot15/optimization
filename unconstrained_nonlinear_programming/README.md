# introduction
Python implementation of solving following linear programming problem with two-phase simplex method.

```
max c.T x
s.t Ax <= b
    x >= 0
```

# description
- `simplex_method.py`:Implementation file of two-phase simplex method.
- `simplex_test.py`:Verifing script with standard input.
- `test_input.txt`:Sample Test case.

## excecution
```
python3 simplex_test.py < test_input.txt
```

## `test_input.txt` format
The first line is in the format below

```
T
```
`T` means number of test cases.

Then,  `T` test cases in the following format are written.
```
n m
c_1 c_2 ... c_n
A_11 A_12 ... A_1n
A_21 A_22 ... A_2n
...
A_m1 A_m2 ... A_mn
b_1 b_2 ... b_m
```

`n` means number of variable, `m` means number of constraints.
`c_i`, `A_ij`, `b_j` are optimization problem coefficients (i = 1, ..., n j = 1, ..., m).