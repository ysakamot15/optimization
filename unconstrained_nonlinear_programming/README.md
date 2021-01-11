# introduction
Python implementation of solving following nonlinear
programming problem with quasi newton method.

```
min f(x)
```

# description
- `quasi_newton_method.py`:Implementation file of quasi newton method.
- `quasi_newton_test.py`:Verifing script.

## quasi_newton arguments
- `oebjective`:Class of obujective function. The class must have following menber.
    - `dim`: Variable dimensions.
    - `func`: Method returning f(x)
    - `grad`: Method returning \nabla f(x)
- `eps`: Convergence criterion
- `tau1`: Parameter of armijo condition
- `beta`: Dumping factor of backtracking line search

## excecution
```
python quasi_newton_test.py
```
