import numpy as np
import interior_point_method as ipm

class function_nonnegative_constraints:
    def __init__(self, dim, idx):
        self.dim = dim
        self.idx = idx
    
    def func(self, x):
        return -x[self.idx]
    
    def grad(self, x):
        res = np.zeros(self.dim)
        res[self.idx] = -1
        return res
    
    def hessian(self, x):
        return np.zeros((self.dim, self.dim))

class test1_objective:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2

    def grad(self, x):
        return np.array([4 * ((x[0] - 2) ** 3) + 2 * (x[0] - 2 * x[1]),
                         -4 * (x[0] - 2 * x[1])]) 

    def hessian(self, x):
        return np.array([[12 * ((x[0] - 2) ** 2) + 2, -4],
                          [-4, 8]])

class test1_inequality_constraint:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 - x[1]

    def grad(self, x):
        return np.array([2 * x[0], -1]) 

    def hessian(self, x):
        return np.array([[2, 0],
                         [0, 0]])

class test2_objective:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 - x[1] ** 2

    def grad(self, x):
        return np.array([2 * x[0], - 2 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, -2.0]])

class test2_equality_constraint:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 + 4 * (x[1] ** 2) - 1

    def grad(self, x):
        return np.array([2 * x[0], 8 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, 8.0]])

class test3_objective:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return 4 * x[0] * x[0] - 4 * x[0] * x[1] + 3 * x[1] * x[1] - 8 * x[0]

    def grad(self, x):
        return np.array([8 * x[0] - 4 * x[1] - 8, 
                         -4 * x[0] + 6 * x[1]]) 

    def hessian(self, x):
        return np.array([[8.0, -4.0],[-4.0, 6.0]])

class test3_inequality_constraint:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] + x[1] - 4

    def grad(self, x):
        return np.array([1.0, 1.0]) 

    def hessian(self, x):
        return np.array([[0.0, 0.0],[0.0, 0.0]])

class linear_test1_objective:
    def __init__(self):
        self.dim = 3

    def func(self, x):
        return -(4 * x[0] + 8 * x[1] + 10 * x[2])

    def grad(self, x):
        return np.array([-4.0, -8.0, -10.0]) 

    def hessian(self, x):
        return np.zeros((self.dim, self.dim))

class linear_test1_inequality_constraint:
    def __init__(self, idx):
        self.dim = 3
        self.idx = idx
        self.A = np.array([[1.0, 1.0, 1.0],
                           [3.0, 4.0, 6.0],
                           [4.0, 5.0, 3.0]])
        self.b = np.array([20, 100, 100])

    def func(self, x):
        return self.A[self.idx, :] @ x - self.b[self.idx]

    def grad(self, x):
        return self.A[self.idx, :]

    def hessian(self, x):
        return np.zeros((self.dim, self.dim))

class linear_test2_objective:
    def __init__(self):
        self.dim = 3

    def func(self, x):
        return 2 * x[0] + 3 * x[1] + x[2]

    def grad(self, x):
        return np.array([2.0, 3.0, 1.0]) 

    def hessian(self, x):
        return np.zeros((self.dim, self.dim))

class linear_test2_inequality_constraint:
    def __init__(self, idx):
        self.dim = 3
        self.idx = idx
        self.A = np.array([[-1.0, -4.0, -2.0],
                           [-3.0, -2.0, 0.0]])
        self.b = np.array([-8.0, -6.0])

    def func(self, x):
        return self.A[self.idx, :] @ x - self.b[self.idx]

    def grad(self, x):
        return self.A[self.idx, :]

    def hessian(self, x):
        return np.zeros((self.dim, self.dim))

class quadratic_test_objective:
    def __init__(self, dim, seed = 2222):
        np.random.seed(seed)
        A = np.random.randn(dim, dim)
        self.Q = A.T @ A
        self.c = np.random.randn(dim)
        self.dim = dim

    def func(self, x):
        return 0.5 * x @ self.Q @ x + self.c @ x

    def grad(self, x):
        return self.Q @ x + self.c

    def hessian(self, x):
        return self.Q

class linear_constraint_creator:
    def __init__(self, dim, constraint_num, seed = 2222):
        self.dim = dim
        np.random.seed(seed)
        self.A = np.random.randn(constraint_num, dim)
        self.b = np.random.randn(constraint_num)

    class function_linear_constraint:
        def __init__(self, dim, A, b, idx):
            self.idx = idx
            self.A = A
            self.b = b
            self.dim = dim
        
        def func(self, x):
            return self.A[self.idx, :] @ x - self.b[self.idx]

        def grad(self, x):
            return self.A[self.idx, :]

        def hessian(self, x):
            return np.zeros((self.dim, self.dim))

    def create(self, idx):
        return self.function_linear_constraint(self.dim, self.A, self.b, idx)

def main():
    # 不等式制約テスト
    Pro = ipm.problem(test1_objective(), [test1_inequality_constraint()], [])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
    print(x_opt, Pro.objective.func(x_opt))

    # 等式制約と非負制約テスト
    dim = 2
    Pro = ipm.problem(test2_objective(), 
            [function_nonnegative_constraints(dim, i) for i in range(dim)], 
            [test2_equality_constraint()])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
    print(x_opt, Pro.objective.func(x_opt))

    # 不等式制約テスト
    Pro = ipm.problem(test3_objective(), [test3_inequality_constraint()], [])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
    print(x_opt, Pro.objective.func(x_opt))

    # 線形計画テスト1
    dim = 3
    Pro = ipm.problem(linear_test1_objective(), 
            [linear_test1_inequality_constraint(i) for i in range(3)] +
            [function_nonnegative_constraints(dim, i) for i in range(3)],
            [])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
    print(x_opt, Pro.objective.func(x_opt))
    
    # 線形計画テスト2
    dim = 3
    Pro = ipm.problem(linear_test2_objective(), 
            [linear_test2_inequality_constraint(i) for i in range(2)] +
            [function_nonnegative_constraints(dim, i) for i in range(2)],
            [])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
    print(x_opt, Pro.objective.func(x_opt))

    # 2次計画問題テスト
    dim = 50
    num = 20
    lcf = linear_constraint_creator(dim, num)
    P = ipm.problem(quadratic_test_objective(dim), 
                    [lcf.create(i) for i in range(num)],
                    [])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(P, eps = 1e-10)
    print(x_opt, P.objective.func(x_opt))

    # 非負制約つき2次計画問題テスト
    dim = 50
    num = 20
    lcf = linear_constraint_creator(dim, num)
    Pro = ipm.problem(quadratic_test_objective(dim), 
            [function_nonnegative_constraints(dim, i) for i in range(dim)],
            [lcf.create(i) for i in range(num)])
    x_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-10)
    print(x_opt, Pro.objective.func(x_opt))

if __name__ == '__main__':
    main()