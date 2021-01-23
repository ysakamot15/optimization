import numpy as np
import copy

# 以下の最適化問題を定義するクラス
# min f(x) : objective
# s.t. g_i(x) <= 0 (i = 1..l) : inequality_constraints_list
#      h_i(x) = 0  (i = 1..m) : equality_constraints_list
# objective, inequality_constraints_list[i], equality_constraints_list[i]は、
# それぞれ以下のメンバを持ってる必要あり
#      func(x)   : return f(x)
#      grad(x)   : return ∇f(x)
#      hessian(x): return ∇^2 f(x)
#      dim       : 変数の次元
class problem():
    def __init__(self, objective, 
                inequality_constraints_list,
                equality_constraints_list):

        for constraint in inequality_constraints_list:
            if objective.dim != constraint.dim:
                raise Exception('dimension mismatch')
        
        for constraint in equality_constraints_list:
            if objective.dim != constraint.dim:
                raise Exception('dimension mismatch')
        
        self.dim = objective.dim
        self.objective = objective
        self.inequality_constraints_list = inequality_constraints_list
        self.equality_constraints_list = equality_constraints_list

def merit_function(problem, rho, eta, x, s):
    res = problem.objective.func(x)
    res -= rho * np.sum(np.log(s))
    g_sum = 0
    for i, constraint in enumerate(problem.inequality_constraints_list):
        g_sum += np.abs(constraint.func(x) + s[i])
    res += eta * g_sum
    h_sum = 0
    for i, constraint in enumerate(problem.equality_constraints_list):
        h_sum += np.abs(constraint.func(x))
    res += eta * h_sum 
    return res 

def backtracking_line_search(problem, x, s, u, dx, ds, du, rho, eta, beta):
    phi = lambda x, s:merit_function(problem, rho, eta, x, s)

    alpha1 = 1.0
    alpha2 = 1.0

    # αの初期値はs + α ds > 0 and u + α du > 0を満たす最大値
    # ただし、α <= 1
    if(np.min(ds) < 0):
        alpha1 = np.min(-s[ds < 0]/ds[ds < 0])
    if(np.min(du) < 0):
        alpha2 = np.min(-u[du < 0]/du[du < 0])
    alpha = min(1.0, alpha1 * beta, alpha2 * beta)

    min_merit = np.inf
    res_alpha = alpha
    init_alpha = alpha

    # 探索範囲はαの初期値の1/10
    while(alpha > init_alpha * 0.1):
        new_x = x + alpha * dx
        new_s = s + alpha * ds

        m = phi(new_x, new_s)
        if (m < min_merit):
            min_merit = m
            res_alpha = alpha
        alpha *= beta
    return res_alpha

# problem : 最適化問題
# eps     : 収束判定
# eta     : メリット関数パラメータ
# beta    : 直線探索の減衰パラメータ
# t       : ρ更新時のパラメータ
def interior_point(problem, eps = 1e-9, eta =0.1, beta = 0.9, t = 0.5):
    
    l = len(problem.inequality_constraints_list) #不等式制約数
    m = len(problem.equality_constraints_list) #等式制約数
    n = problem.dim #変数の次元

    rho = 1
    # 初期値xはとりあえず制約を満たさなくてもよい
    x = np.ones(n)
    # s の初期値は s > 0
    s = np.ones(l)
    # s_i u_i = ρを満たすようにuを設定
    u = rho / s

    v = np.zeros(m)
    
    while(1):
        if rho < eps:
            break
        H = problem.objective.hessian(x)
        d = problem.objective.grad(x)
        Jg = np.zeros((l, n))
        g = np.zeros(l)
        # 不等式制約の関数がらみの計算
        for i, constraint in enumerate(problem.inequality_constraints_list):
            H += constraint.hessian(x) * u[i]
            Jg[i, :] = constraint.grad(x)
            d += Jg[i, :] * u[i]
            g[i] = constraint.func(x)
        
        Jh = np.zeros((m, n))
        h = np.zeros(m)
        # 等式制約の関数がらみの計算
        for i, constraint in enumerate(problem.equality_constraints_list):
            H += constraint.hessian(x) * v[i]
            Jh[i, :] = constraint.grad(x)
            d += Jh[i, :] * v[i]
            h[i] = constraint.func(x)

        Du = np.diag(u)
        Ds = np.diag(s)

        # make array P       
        # |H   O   Jg.T  Jh.T|
        # |O   Du  Ds    O   |
        # |Jg  I   O     O   |
        # |Jh  O   O     O   | 
        P = np.zeros((n + 2 * l + m, n + 2 * l + m))
        P[:n, :n] = H
        P[:n, n + l: n + 2 * l] = Jg.T
        P[:n, n + 2 * l:] = Jh.T
        P[n:n + l, n:n + l] = Du
        P[n:n + l, n + l:n + 2 * l] = Ds
        P[n + l:n + 2 * l, :n] = Jg
        P[n + l:n + 2 * l, n:n + l] = np.eye(l)
        P[n + 2 * l:, :n] = Jh

        # make vector r
        r = np.concatenate([-d, rho - s * u, -(g + s), -h])

        delta = np.linalg.solve(P, r)

        dx = delta[:n]
        ds = delta[n:n + l]
        du = delta[n + l:n + 2 * l]
        dv = delta[n + 2 * l:]

        alpha = backtracking_line_search(problem, x, s, u,
                                            dx, ds, du, rho, eta, beta)

        # x,s,u,vの更新
        x += alpha * dx
        s += alpha * ds
        u += alpha * du
        v += alpha * dv

        # ρの更新
        rho =  t * u @ s / l

    return x, s, u, v