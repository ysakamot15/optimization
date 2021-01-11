import numpy as np
import quasi_newton_method as qnm

class objective_function_test1:
    def __init__(self):
        self.dim = 2
        self.A = np.array([[2, 0.5],
                            [0.5, 1]])
        self.b = np.array([-5, -3])
        self.c = 4

    def func(self, x):
        return x @ self.A @ x + self.b @ x + self.c

    def grad(self, x):
        return 2 * self.A @ x + self.b

class objective_function_test2:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return x[0] * x[0] + 3 * (x[1] ** 4)

    def grad(self, x):
        return np.array([2 * x[0], 12 * (x[1] ** 3)])

class objective_function_test3:
    def __init__(self):
        self.dim = 2

    def func(self, x):
        return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2

    def grad(self, x):
        return np.array([4 * (x[0] - 2) ** 3 + 2 * (x[0] - 2 * x[1]),
                        -4 *  (x[0] - 2 * x[1])])

class objective_function_test4:
    def __init__(self, dim):
        self.dim = dim
        self.A = np.random.randn(dim, dim)
        self.A = self.A.T @ self.A
        self.b = np.random.randn(dim)
        self.c = np.random.randn(1)
        
    def func(self, x):
        return x @ self.A @ x + self.b @ x + self.c

    def grad(self, x):
        return 2 * self.A @ x + self.b

    def analytic_solution(self):
        x_opt = -0.5 * np.linalg.inv(self.A) @ self.b
        return x_opt

def main():
    objective = objective_function_test1()
    print(qnm.quasi_newton(objective, eps=1e-6))
    
    objective = objective_function_test2()
    print(qnm.quasi_newton(objective, eps=1e-6))

    objective = objective_function_test3()
    print(qnm.quasi_newton(objective, eps=1e-6))
    
    objective = objective_function_test4(200)
    print(qnm.quasi_newton(objective, eps=1e-3))

if __name__ == '__main__':
    main()