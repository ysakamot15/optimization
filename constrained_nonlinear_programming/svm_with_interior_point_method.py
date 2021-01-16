import numpy as np
import interior_point_method as ipm
import matplotlib.pyplot as plt

class objective_function:
    def __init__(self, K, C, var_dim, constraint_num):
        self.K = K
        self.C = C
        self.var_dim = var_dim
        self.constraint_num = constraint_num
        self.dim = var_dim + constraint_num + 1

    def func(self, a_e_b):
        return 0.5 * a_e_b[:self.var_dim] @ self.K @ a_e_b[:self.var_dim] + \
                self.C * np.sum(a_e_b[self.var_dim:self.var_dim + self.constraint_num])
           
    def grad(self, a_e_b):
        return np.concatenate([self.K @ a_e_b[:self.var_dim],
                               self.C * np.ones(self.constraint_num),
                               [0]])

    def hessian(self, w_s):
        H = np.zeros((self.dim, self.dim))
        H[:self.var_dim, :self.var_dim] = self.K
        return H


class constraint_factory:
    def __init__(self, K, y):
        self.constraint_num = y.shape[0]
        self.dim = K.shape[0] 
        self.K = K
        self.y = y

    class function_constraint:
        def __init__(self, K, y, idx):
            self.idx = idx
            self.K = K
            self.y = y
            self.dim = K.shape[0] + y.shape[0] + 1
        
        def func(self, a_e_b):
            return -a_e_b[self.K.shape[0] + self.idx] + 1 - \
                    self.y[self.idx] * \
                    (a_e_b[:self.K.shape[0]] @ self.K[self.idx, :] + a_e_b[-1])
             

        def grad(self, a_e_b):
            return np.concatenate([-self.K[self.idx, :] * self.y[self.idx],
                                    np.ones(self.y.shape[0]) * -1,
                                    [-self.y[self.idx]]])

        def hessian(self, a_e_b):
            return np.zeros((self.dim, self.dim))

    def create(self, idx):
        return self.function_constraint(self.K, self.y, idx)
    
class positive_constraint_factory:
    def __init__(self, K, y):
        self.constraint_num = y.shape[0]
        self.var_dim = K.shape[0]
        self.constraint_num = y.shape[0]
        self.dim = self.var_dim + self.constraint_num + 1

    class function_positive_constraint:
        def __init__(self, var_dim, constraint_num, idx):
            self.idx = idx
            self.var_dim = var_dim
            self.constraint_num = constraint_num
            self.dim = self.var_dim + self.constraint_num + 1
        
        def func(self, a_e_b):
            return -a_e_b[self.var_dim + self.idx]
        
        def grad(self, a_e_b):
            res = np.zeros(self.dim)
            res[self.var_dim + self.idx] = -1
            return res

        def hessian(self, a_e_b):
            return np.zeros((self.dim, self.dim))
             
    def create(self, idx):
        return self.function_positive_constraint(self.var_dim, self.constraint_num, idx)


class svm:
    def __init__(self, C, kernel_func):
        self.C = C
        self.K = None
        self.X = None
        self.kernel_func = kernel_func

    def fit(self, X, y):
        self.X = X
        self.K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                self.K[i, j] = self.kernel_func(X[i, :], X[j, :])
                self.K[j, i] = self.K[i, j]
        cf = constraint_factory(self.K, y)
        pcf = positive_constraint_factory(self.K, y)
        P = ipm.problem(objective_function(self.K, self.C, self.K.shape[1], y.shape[0]), 
                    [cf.create(i) for i in range(y.shape[0])] + 
                    [pcf.create(i) for i in range(y.shape[0])],
                    [])
        a_e_b_opt, s_opt, u_opt, v_opt = ipm.inter_point(P, eps = 1e-9)
        self.a = a_e_b_opt[:self.K.shape[0]]
        self.b = a_e_b_opt[-1]
        self.u = u_opt
        self.support_vectors = np.where(self.u[:X.shape[0]] > 1e-4)

    def predict(self, X_):
        K_ = np.zeros((self.K.shape[0], X_.shape[0]))
        for i in range(self.K.shape[0]):
            for j in range(X_.shape[0]):
                K_[i, j] = self.kernel_func(self.X[i], X_[j])
        return np.sign(self.a @ K_ + self.b)


def make_data(xmin, xmax, n):
    np.random.seed(2222)
    cnt = 0
    t = 0
    N = n * 9
    X_train = np.zeros((2, N))
    y_train = np.zeros(N)
    center1 =  np.linspace(-(-xmin[0] + xmax[0])/3.5, (-xmin[0] + xmax[0])/3.5, 3)
    center2 =  np.linspace(-(-xmin[1] + xmax[1])/3.5, (-xmin[1] + xmax[1])/3.5, 3)
    for i in range(3):
        for j in range(3):
            y_train[t:t + n] = 1 if cnt % 2 == 0 else -1
            X_train[:, t:t + n] = 1.5 * np.random.randn(2, n) \
                                    + np.array([center1[i], center2[j]])[:, None]
            t += n
            cnt += 1


    #テスト入力点生成
    T1 = np.linspace(xmin[0], xmax[0], 60)
    T2 = np.linspace(xmin[1], xmax[1], 60)
    X_test = np.zeros((2, T1.shape[0] * T2.shape[0]))
    i = 0
    for t1 in T1:
        for t2 in T2:
            X_test[:, i] = np.array([t1, t2])
            i += 1
    return X_train.T, y_train, X_test.T

def main():
    xmin = np.array([-9, -9])
    xmax = np.array([9, 9])
    
    #学習データとテストデータの生成
    n = 20
    X_train, y_train, X_test = make_data(xmin, xmax, n)

    gamma = 0.05
    kernel_func = lambda x_i, x_j : np.exp(-gamma * (x_i - x_j) @ (x_i - x_j))
    sv = svm(1000, kernel_func)
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    y_pred[y_pred == -1] = 0
    size = 100
    sc = plt.scatter(X_test[:, 0], X_test[:, 1], size, y_pred, marker = ',', cmap='bwr')
    plt.colorbar(sc)

    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1] ,
                edgecolors="black", marker="o" , color='r')
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1] ,
                edgecolors="black", marker="o" , color='b')

    plt.scatter(X_train[sv.support_vectors, 0], 
                X_train[sv.support_vectors, 1], color = 'g', marker='^', edgecolors="black")


    plt.axis([xmin[0], xmax[0], xmin[1], xmax[1]])

    plt.show()
    # 2020/2/2 プロットの色が逆だったので修正
    # plt.scatter(X_train[0, y_train == 1], X_train[1, y_train == 1] ,
    #             edgecolors="black", marker="o" , color='b')
    # plt.scatter(X_train[0, y_train == 0], X_train[1, y_train == 0] ,
    #             edgecolors="black", marker="o" , color='r')
    # plt.scatter(X_train[0, y_train == 1], X_train[1, y_train == 1] ,
    #             edgecolors="black", marker="o" , color='r')
    # plt.scatter(X_train[0, y_train == 0], X_train[1, y_train == 0] ,
    #             edgecolors="black", marker="o" , color='b')

    # plt.axis([xmin[0], xmax[0], xmin[1], xmax[1]])

    # plt.show()
    # N_p = 40
    # mu_p = [10, 20]
    # Sig_p = [[8, 4],[4, 7]]
    # X_p = np.random.multivariate_normal(mu_p, Sig_p, N_p)
    

    # N_m = 40
    # mu_m = [11, 17]
    # Sig_m = [[8, -4],[-4, 7]]
    # X_m = np.random.multivariate_normal(mu_m, Sig_m, N_m)


    # X_train = np.concatenate([X_p[:N_p//2, :], X_m[:N_m//2, :]], axis = 0)
    # y_train = np.ones(N_m//2 + N_p//2)
    # y_train[N_p//2:] = -1
    # gamma = 0.001
    # kernel_func = lambda x_i, x_j : np.exp(-gamma * (x_i - x_j) @ (x_i - x_j))
    # #kernel_func = lambda x_i, x_j : (1 + gamma * x_i @ x_j) ** 4
    # #kernel_func = lambda x_i, x_j : gamma * x_i @ x_j
    # sv = svm(200000, kernel_func)
    # X_test = np.concatenate([X_p[N_p//2:, :], X_m[N_m//2:, :]], axis = 0)
    # sv.fit(X_train, y_train)
    # y_pred = sv.predict(X_test)
    # print(y_pred)

    # test_range = np.arange(0, 36, 1)
    # X_new = np.zeros((2, len(test_range)**2))
    # S_new = np.zeros((3, len(test_range)**2))
    # count = 0
    # S_res = np.zeros((len(test_range), len(test_range), 3))
    # for x1 in test_range:
    #     for x2 in test_range:
    #         #各座標位置に対する予測平均を算出
    #         X_new[:, count] = np.array([float(x1), float(x2)])
    #         y_pred = sv.predict(X_new[:, count][None, :])
    #         if  y_pred == 1:
    #             S_new[0, count] = 1
    #         else:
    #             S_new[2, count] = 1
    #         S_res[int(x2 - test_range[0]), int(x1 - test_range[0]), :] = S_new[:, count]
    #         count += 1

    # plt.imshow(S_res)

    # plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color = 'r',edgecolors="black")
    # plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color = 'b', edgecolors="black")

    # # x = np.array([-100 , 0, 100])
    # # y = -(sv.w[0] * x + sv.b)/sv.w[1]
    # # plt.plot(x, y, color = 'b')
    # plt.xlim(0, 25)
    # plt.ylim(5, 30)
    # #plt.scatter(X_train[sv.support_vectors, 0], X_train[sv.support_vectors, 1], color = 'g', marker='^', edgecolors="black")

    # #plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], color = 'r', marker='x')
    # #plt.scatter(X_test[y_pred == -1, 0], X_test[y_pred == -1, 1], color = 'b', marker='x')   
    # plt.show()


if __name__ == '__main__':
    main()