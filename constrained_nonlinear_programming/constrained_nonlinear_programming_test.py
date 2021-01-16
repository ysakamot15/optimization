import numpy as np
import interior_point_method as ipm


class function1:
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

class function2:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 - x[1]

    def grad(self, x):
        return np.array([2 * x[0], -1]) 

    def hessian(self, x):
        return np.array([[2, 0],
                         [0, 0]])

class function3:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 - x[1] ** 2


    def grad(self, x):
        return np.array([2 * x[0], - 2 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, -2.0]])

class function_positive_constraint:
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


class function_quadratic_objective:
    def __init__(self, dim):
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

class linear_constraint_factory:
    def __init__(self, dim, constraint_num):
        self.dim = dim
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


class function4:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 + 4 * (x[1] ** 2) - 1

    def grad(self, x):
        return np.array([2 * x[0], 8 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, 8.0]])


class function5:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 - x[1] ** 2

    def grad(self, x):
        return np.array([2 * x[0], - 2 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, -2.0]])

class function6:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] ** 2 + 4 * (x[1] ** 2) - 1

    def grad(self, x):
        return np.array([2 * x[0], 8 * x[1]]) 

    def hessian(self, x):
        return np.array([[2.0, 0.0],[0.0, 8.0]])

def main():
    dim = 2
    P = ipm.problem(function3(), [function_positive_constraint(dim, i) for i in range(dim)], [function4()])
    x_opt, s_opt, u_opt, v_opt = ipm.inter_point(P, eps = 1e-10)
    print(x_opt, P.objective.func(x_opt))

    # dim = 50
    # num = 20
    # lcf = linear_constraint_factory(dim, num)
    # P = ipm.problem(function_quadratic_objective(dim), 
    #                 [function_positive_constraint(dim, i) for i in range(dim)],
    #                 [lcf.create(i) for i in range(num)])
    # x_opt, s_opt, u_opt, v_opt = ipm.inter_point(P, eps = 1e-10)
    # print(x_opt, s_opt, u_opt, v_opt, P.objective.func(x_opt))

    # dim = 50
    # num = 20
    # lcf = linear_constraint_factory(dim, num)
    # P = ipm.problem(function_quadratic_objective(dim), 
    #                 [lcf.create(i) for i in range(num)],
    #                 [])
    # x_opt, s_opt, u_opt, v_opt = ipm.inter_point(P, eps = 1e-10)
    # print(x_opt, s_opt, u_opt, v_opt, P.objective.func(x_opt))


if __name__ == '__main__':
    main()