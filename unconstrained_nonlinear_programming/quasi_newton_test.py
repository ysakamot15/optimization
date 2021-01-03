import numpy as np
import quasi_newton_method as qnm

class objective_function1:
    def __init__(self, dim):
        self.dim = dim
        self.A = np.array([[2, 0.5],
                            [0.5, 1]])
        self.b = np.array([-5, -3])
        self.c = 4

    def f(self, x):
        return x @ self.A @ x + self.b @ x + self.c

    def nabla_f(self, x):
        return 2 * self.A @ x + self.b

class objective_function3:
    def __init__(self, dim):
        self.dim = dim
        self.A = np.random.randn(dim, dim)
        self.A = self.A.T @ self.A
        self.b = np.random.randn(dim)
        self.c = np.random.randn(1)
        
    def f(self, x):
        return x @ self.A @ x + self.b @ x + self.c

    def nabla_f(self, x):
        return 2 * self.A @ x + self.b

    def analytic_solution(self):
        x_opt = -0.5 * np.linalg.inv(self.A) @ self.b
        return x_opt

class objective_function2:
    def __init__(self, dim):
        self.dim = dim

    def f(self, x):
        return x[0] * x[0] + 3 * (x[1] ** 4)

    def nabla_f(self, x):
        return np.array([2 * x[0], 12 * (x[1] ** 3)])

def main():
    func = objective_function3(200)
    print(qnm.quasi_newton(func, eps=1e-3))
    x_a = func.analytic_solution()
    print("correct:", x_a, func.f(x_a))
    

if __name__ == '__main__':
    main()