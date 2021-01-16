import numpy as np
import copy


# min f(x) : objective
# s.t. g_i(x) <= 0 (i = 1..M) : constraints_list
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
    alpha = 1.0
    phi = lambda x, s:merit_function(problem, rho, eta, x, s)
    alpha1 = 1.0
    alpha2 = 1.0
    if(np.min(ds) < 0):
        alpha1 = np.min(-s[ds < 0]/ds[ds < 0])
    if(np.min(du) < 0):
        alpha2 = np.min(-u[du < 0]/du[du < 0])
    # min_ds = np.min(ds)
    # min_ds_idx = np.argmin(ds)
    # min_du = np.min(du)
    # min_du_idx = np.argmin(du)
    # alpha1 = 1.0
    # alpha2 = 1.0
    # if(min_ds < 0):
    #     alpha1 = -s[min_ds_idx]/min_ds
    # if(min_du < 0):
    #     alpha2 = -u[min_du_idx]/min_du
    alpha = min(alpha, alpha1 * beta, alpha2 * beta)
    #return alpha
    min_merit = np.inf
    res_alpha = alpha
    init_alpha = alpha
    #return res_alpha
    while(alpha > init_alpha * 0.1):
        new_x = x + alpha * dx
        new_s = s + alpha * ds
        new_u = u + alpha * du
        # if((new_s <= 0).any() == True or
        #    (new_u <= 0).any() == True):
        #     alpha *= beta
        #     continue

        m = phi(new_x, new_s)
        if (m < min_merit):
            # 1度でも更新が発生
            # if(is_update == False):
            #     is_update = True
            #     res_alpha_first_update = alpha
            min_merit = m
            res_alpha = alpha
        alpha *= beta
    # if(res_alpha < 1e-7):
    #     return res_alpha_first_update
    return res_alpha



def inter_point(problem, eps = 1e-9, eta =0.1, beta = 0.9, t = 0.05):
    
    l = len(problem.inequality_constraints_list) #不等式制約数
    m = len(problem.equality_constraints_list) #等式制約数
    n = problem.dim #変数の次元
    
    rho = 1
    # 初期値xはとりあえず制約を満たさなくてもよい
    x = np.random.rand(n)#p.ones(n) * 1#np.random.rand(n)# np.ones(n) * 1# 

    #s = np.zeros(l)
    s = np.ones(l) * 1.0 #np.random.rand(l) + 1.005 #np.ones(l) * 1
    # si + gi(x) = 0より
    # for i, constraint in enumerate(problem.inequality_constraints_list):
    #     s[i] = -constraint.func(x) 
    # ui si = rhoより
    u = rho / s

    v = np.zeros(m)
    
    while(1):
        if rho < eps:
            break
        H = problem.objective.hessian(x)
        d = problem.objective.grad(x)
        Ng = np.zeros((n, l))
        g = np.zeros(l)
        h = np.zeros(m)
        for i, constraint in enumerate(problem.inequality_constraints_list):
            H += constraint.hessian(x) * u[i]
            Ng[:, i] = constraint.grad(x)
            d += Ng[:, i] * u[i]
            g[i] = constraint.func(x)
        
        Nh = np.zeros((n, m))
        for i, constraint in enumerate(problem.equality_constraints_list):
            H += constraint.hessian(x) * v[i]
            Nh[:, i] = constraint.grad(x)
            d += Nh[:, i] * v[i]
            h[i] = constraint.func(x)

        Du = np.diag(u)
        Ds = np.diag(s)

        # make array P       
        # |H     O   Ng  Nh|
        # |O     Du  Ds  O |
        # |Ng.T  I   O   O |
        # |Nh.T  O   O   O | 
        P = np.zeros((n + 2 * l + m, n + 2 * l + m))
        P[:n, :n] = H
        P[:n, n + l: n + 2 * l] = Ng
        P[:n, n + 2 * l:] = Nh
        P[n:n + l, n:n + l] = Du
        P[n:n + l, n + l:n + 2 * l] = Ds
        P[n + l:n + 2 * l, :n] = Ng.T
        P[n + l:n + 2 * l, n:n + l] = np.eye(l)
        P[n + 2 * l:, :n] = Nh.T

        # make vector r
        r = np.concatenate([-d, rho - s * u, -(g + s), -h])

        #delta = np.linalg.inv(P + 0.1 * np.eye(P.shape[0], P.shape[1])) @ r
        #delta = np.linalg.pinv(P, rcond=1e-6) @ r
        delta = np.linalg.solve(P, r)
        dx = delta[:n]
        ds = delta[n:n + l]
        du = delta[n + l:n + 2 * l]
        dv = delta[n + 2 * l:]

        alpha = backtracking_line_search(problem, x, s, u, dx, ds, du, rho, eta, beta)

        x += alpha * dx
        s += alpha * ds
        u += alpha * du
        v += alpha * dv

        rho =  t * u @ s / l

        print(rho)

    return x, s, u, v


