# introduction
Python implementation of solving following nonlinear
programming problem with primal dual interior point method.

```
min f(x)
s.t. gi(x) <= 0 (i = 1, ..., l)
     hi(x) = 0 (i = 1, ..., m)
```

# description
- `interior_point_method.py`:Implementation file of primal dual interior point method.
- `constrained_nonlinear_programming_test.py`:Verifing script.
- `linear_svm_with_interior_point_method.py`:Linear SVM implementation using interior point method.
- `svm_with_interior_point_method.py`:Kernel SVM implementation using interior point method.

## interior_point arguments
- `Problem`:Class of nonlinear programming problem function.
     - `objective`: Objective function f(x) class.
     - `inequality_constraints_list`:List of inequality constraint g_i(x) class.
     - `equality_constraints_list`:List of equality constraint h_i(x) class.
          - And each f(x), g_i(x), h_i(x) classes must have following member.
               - `dim`: Variable dimensions.
               - `func`: Method returning f(x)
               - `grad`: Method returning \nabla f(x)
               - `hessian`: Method returning \nabla^2 f(x)
- `eps`: Convergence criterion
- `eta`: Parameter of merit function
- `beta`: Dumping factor of backtracking line search
- `t`: Parameter of rho update

## excecution
- Kernel SVM sample excecution.
```
python svm_with_interior_point_method.py
```
