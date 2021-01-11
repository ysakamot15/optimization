import numpy as np
import copy


# アルミホ条件を満たすαを線形探索
def backtracking_line_search(objective, x, d, tau1, beta):
    alpha = 1
    g = lambda alpha : objective.func(x + alpha * d)
    g_dash = lambda alpha : objective.grad(x + alpha * d) @ d    
    while(1):
        if ((g(alpha) <= g(0) + tau1 * g_dash(0) * alpha)):
                break
        alpha *= beta
    return alpha

# 目的関数のクラスobjectiveは以下のメンバを持っている必要あり
# objective.func(x) : return f(x)
# objective.grad(x) : return ∇f(x)
# objective.dim : 変数の次元
def quasi_newton(objective, eps = 1e-9, tau1 = 0.5, beta = 0.9):
    H = np.eye(objective.dim) # = inv(B)
    x = np.random.randn(objective.dim)
    
    while(1):
        grad = objective.grad(x)
        if(np.linalg.norm(grad) <= eps):
            break
        d = - H @ grad
        alpha = backtracking_line_search(objective, x, d, tau1, beta)
        x_new = x + alpha * d
        s = (x_new - x)[:, None]
        y = (objective.grad(x_new) - grad)[:, None]
        I = np.eye(H.shape[0])
        sy = (s.T @ y)
        H = (I - (s @ y.T) / sy)  @ H @ (I - (y @ s.T) / sy) + (s @ s.T / sy)
        x = x_new
    return x, objective.func(x)