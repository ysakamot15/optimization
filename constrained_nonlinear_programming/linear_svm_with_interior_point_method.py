import numpy as np
import interior_point_method as ipm
import matplotlib.pyplot as plt
class objective_function:
    def __init__(self, C, var_dim, constraint_num):
        self.C = C
        self.var_dim = var_dim
        self.constraint_num = constraint_num
        self.dim = var_dim + constraint_num + 1 # w, xi, b

    def func(self, w_s_b):
        return 0.5 * w_s_b[:self.var_dim] @ w_s_b[:self.var_dim] + \
        self.C * np.sum(w_s_b[self.var_dim:self.var_dim + self.constraint_num])
           
    def grad(self, w_s_b):
        return np.concatenate([w_s_b[:self.var_dim],
                               self.C * np.ones(self.constraint_num),
                               [0]])

    def hessian(self, w_s):
        H = np.zeros((self.dim, self.dim))
        H[:self.var_dim, :self.var_dim] = np.eye(self.var_dim)
        return H


class constraint_factory:
    def __init__(self, X, y):
        self.constraint_num = y.shape[0]
        self.dim = X.shape[1] 
        self.X = X
        self.y = y

    class function_constraint:
        def __init__(self, X, y, idx):
            self.idx = idx
            self.X = X
            self.y = y
            self.dim = X.shape[1] + y.shape[0] + 1
        
        def func(self, w_s_b):
            return -w_s_b[self.X.shape[1] + self.idx] + 1 - \
                    self.y[self.idx] * \
                    (w_s_b[:self.X.shape[1]] @ self.X[self.idx, :] + w_s_b[-1])
             

        def grad(self, w_s_b):
            return np.concatenate([-self.X[self.idx, :] * self.y[self.idx],
                                    np.ones(self.y.shape[0]) * -1,
                                    [-self.y[self.idx]]])

        def hessian(self, w_s_b):
            return np.zeros((self.dim, self.dim))

    def create(self, idx):
        return self.function_constraint(self.X, self.y, idx)
    
class positive_constraint_factory:
    def __init__(self, X, y):
        self.constraint_num = y.shape[0]
        self.var_dim = X.shape[1]
        self.constraint_num = y.shape[0]
        self.dim = self.var_dim + self.constraint_num + 1

    class function_positive_constraint:
        def __init__(self, var_dim, constraint_num, idx):
            self.idx = idx
            self.var_dim = var_dim
            self.constraint_num = constraint_num
            self.dim = self.var_dim + self.constraint_num + 1
        
        def func(self, w_s_b):
            return -w_s_b[self.var_dim + self.idx]
        
        def grad(self, w_s_b):
            res = np.zeros(self.dim)
            res[self.var_dim + self.idx] = -1
            return res

        def hessian(self, w_s_b):
            return np.zeros((self.dim, self.dim))
             
    def create(self, idx):
        return self.function_positive_constraint(self.var_dim, self.constraint_num, idx)


class svm:
    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        cf = constraint_factory(X, y)
        pcf = positive_constraint_factory(X, y)
        P = ipm.problem(objective_function(self.C, X.shape[1], y.shape[0]), 
                    [cf.create(i) for i in range(y.shape[0])] + 
                    [pcf.create(i) for i in range(y.shape[0])],
                    [])
        w_s_b_opt, s_opt, u_opt, v_opt = ipm.inter_point(P, eps = 1e-10)
        self.w = w_s_b_opt[:X.shape[1]]
        self.b = w_s_b_opt[-1]
        self.u = u_opt
        self.support_vectors = np.where(self.u[:X.shape[0]] > 1e-4)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

def main():
    N_p = 300
    mu_p = [3, 20]
    Sig_p = [[8, 4],[4, 7]]
    X_p = np.random.multivariate_normal(mu_p, Sig_p, N_p)
    

    N_m = 300
    mu_m = [7, 10]
    Sig_m = [[8, -4],[-4, 7]]
    X_m = np.random.multivariate_normal(mu_m, Sig_m, N_m)
    


    X_train = np.concatenate([X_p[:N_p//2, :], X_m[:N_m//2, :]], axis = 0)
    y_train = np.ones(N_m//2 + N_p//2)
    y_train[N_p//2:] = -1
    sv = svm(500) 
    X_test = np.concatenate([X_p[N_p//2:, :], X_m[N_m//2:, :]], axis = 0)
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    print(y_pred)

    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color = 'r')
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color = 'b')

    x = np.array([-100 , 0, 100])
    y = -(sv.w[0] * x + sv.b)/sv.w[1]
    plt.plot(x, y, color = 'b')
    plt.xlim(-10, 25)
    plt.ylim(-10, 25)
    plt.scatter(X_train[sv.support_vectors, 0], X_train[sv.support_vectors, 1], color = 'g', marker='^')

    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], color = 'r', marker='x')
    plt.scatter(X_test[y_pred == -1, 0], X_test[y_pred == -1, 1], color = 'b', marker='x')   
    plt.show()


if __name__ == '__main__':
    main()