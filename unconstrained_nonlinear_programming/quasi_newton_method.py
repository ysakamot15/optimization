import numpy as np
import copy



def backtracking_line_search(func, x, d, tau1, tau2, beta):
    alpha = 1
    g = lambda alpha : func.f(x + alpha * d)
    g_dash = lambda alpha : func.nabla_f(x + alpha * d) @ d    
    while(1):
        if ((g(alpha) <= g(0) + tau1 * g_dash(0) * alpha)):
                break
        alpha *= beta
    return alpha

def quasi_newton(func, eps = 1e-9, tau1 = 0.02, tau2 = 0.9, beta = 0.9):
    H = np.eye(func.dim) # = inv(B)
    x = np.random.randn(func.dim)

    while(1):
        grad = func.nabla_f(x)
        if(np.linalg.norm(grad) <= eps):
            break
        d = - H @ grad
        alpha = backtracking_line_search(func, x, d, tau1, tau2, beta)
        x_new = x + alpha * d
        s = (x_new - x)[:, None]
        y = (func.nabla_f(x_new) - grad)[:, None]
        I = np.eye(H.shape[0])
        sy = (s.T @ y)
        H = (I - (s @ y.T) /(sy))  @ H @ (I - (y @ s.T) /(sy)) + (s @ s.T /(sy))
        x = x_new
    return x, func.f(x)