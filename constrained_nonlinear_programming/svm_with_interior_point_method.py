import numpy as np
import interior_point_method as ipm
import matplotlib.pyplot as plt

# 目的関数を表すクラス
class objective_function:
    def __init__(self, K, C, var_dim, constraint_num):
        self.K = K
        self.C = C
        self.a_dim = var_dim
        self.z_dim = constraint_num # ξはzで表現
        self.dim = var_dim + constraint_num + 1 # aの次元 + ξの次元 + bの次元

    # a_z_bはaとξとbを1つに並べたベクトル（array）
    def func(self, a_z_b):
        return 0.5 * a_z_b[:self.a_dim] @ self.K @ a_z_b[:self.a_dim] + \
                self.C * np.sum(a_z_b[self.a_dim:self.a_dim + self.z_dim])
           
    def grad(self, a_z_b):
        return np.concatenate([self.K @ a_z_b[:self.a_dim],
                               self.C * np.ones(self.z_dim),
                               [0]])

    def hessian(self, a_z_b):
        H = np.zeros((self.dim, self.dim))
        H[:self.a_dim, :self.a_dim] = self.K
        return H

# 教師データによる不等式制約クラスを生成するクラス
class train_data_constraint_creator:
    def __init__(self, K, y):
        self.K = K
        self.y = y

    # 教師データによる不等式制約を表すクラス
    class train_data_constraint:
        def __init__(self, K, y, idx):
            self.idx = idx
            self.K = K
            self.y = y
            self.a_dim = K.shape[0]
            self.z_dim = y.shape[0]
            self.dim = self.a_dim + self.z_dim + 1
        
        def func(self, a_z_b):
            return -a_z_b[self.a_dim + self.idx] + 1 - \
                    self.y[self.idx] * \
                    (a_z_b[:self.a_dim] @ self.K[self.idx, :] + a_z_b[-1])
             
        def grad(self, a_z_b):
            return np.concatenate([-self.K[self.idx, :] * self.y[self.idx],
                                    np.ones(self.z_dim) * -1,
                                    [-self.y[self.idx]]])

        def hessian(self, a_z_b):
            return np.zeros((self.dim, self.dim))

    def create(self, idx):
        return self.train_data_constraint(self.K, self.y, idx)

# 非負制約クラスを生成するクラス
class nonnegative_constraint_creator:
    def __init__(self, K, y):
        self.constraint_num = y.shape[0]
        self.var_dim = K.shape[0]
        self.constraint_num = y.shape[0]
        self.dim = self.var_dim + self.constraint_num + 1

    # 非負制約を表すクラス
    class function_nonnegative_constraint:
        def __init__(self, var_dim, constraint_num, idx):
            self.idx = idx
            self.a_dim = var_dim
            self.z_dim = constraint_num
            self.dim = self.a_dim + self.z_dim + 1
        
        def func(self, a_z_b):
            return -a_z_b[self.a_dim + self.idx]
        
        def grad(self, a_z_b):
            res = np.zeros(self.dim)
            res[self.a_dim + self.idx] = -1
            return res

        def hessian(self, a_z_b):
            return np.zeros((self.dim, self.dim))
             
    def create(self, idx):
        return self.function_nonnegative_constraint(self.var_dim,
                                                self.constraint_num, idx)

class svm:
    def __init__(self, C, kernel_func):
        self.C = C
        self.K = None
        self.X = None
        self.kernel_func = kernel_func

    def fit(self, X, y):
        self.X = X
        self.K = np.zeros((X.shape[0], X.shape[0]))

        # グラム行列の計算
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                self.K[i, j] = self.kernel_func(X[i, :], X[j, :])
                self.K[j, i] = self.K[i, j]

        # 各種不等式制約インスタンス生成用
        tdcc = train_data_constraint_creator(self.K, y)
        ncc = nonnegative_constraint_creator(self.K, y)
        # 最適化問題のインスタンスを生成
        Pro = ipm.problem(objective_function(self.K, self.C,
                                            self.K.shape[1], y.shape[0]), 
                        [tdcc.create(i) for i in range(y.shape[0])] + 
                        [ncc.create(i) for i in range(y.shape[0])],
                        [])
        
        a_z_b_opt, s_opt, u_opt, v_opt = ipm.interior_point(Pro, eps = 1e-15)
        self.a = a_z_b_opt[:self.K.shape[0]]
        self.b = a_z_b_opt[-1]
        self.u = u_opt
        self.support_vectors = np.where(self.u[:X.shape[0]] > 1e-8)

    def predict(self, X_):
        if X_ is None:
             raise Exception('not learned')
            
        K_ = np.zeros((self.K.shape[0], X_.shape[0]))
        for i in range(self.K.shape[0]):
            for j in range(X_.shape[0]):
                K_[i, j] = self.kernel_func(self.X[i], X_[j])
        return np.sign(self.a @ K_ + self.b)


# テストデータ生成用
def make_data(xmin, xmax, n):
    np.random.seed(1111)
    cnt = 0
    t = 0
    N = n * 9
    X_train = np.zeros((N, 2))
    y_train = np.zeros(N)
    center1 =  np.linspace(-(-xmin[0] + xmax[0])/3.5, 
                            (-xmin[0] + xmax[0])/3.5, 3)
    center2 =  np.linspace(-(-xmin[1] + xmax[1])/3.5, 
                            (-xmin[1] + xmax[1])/3.5, 3)
    for i in range(3):
        for j in range(3):
            y_train[t:t + n] = 1 if cnt % 2 == 0 else -1
            X_train[t:t + n, :] = 1.5 * np.random.randn(n, 2) + \
                                np.array([center1[i], center2[j]])
            t += n
            cnt += 1

    #テスト入力点生成
    T1 = np.linspace(xmin[0], xmax[0], 60)
    T2 = np.linspace(xmin[1], xmax[1], 60)
    X_test = np.zeros((T1.shape[0] * T2.shape[0], 2))
    i = 0
    for t1 in T1:
        for t2 in T2:
            X_test[i, :] = np.array([t1, t2])
            i += 1
    return X_train, y_train, X_test

def main():
    #教師データとテストデータの生成
    xmin = np.array([-9, -9])
    xmax = np.array([9, 9])
    n = 20
    X_train, y_train, X_test = make_data(xmin, xmax, n)

    # カーネル関数（ガウスカーネル）
    gamma = 0.5
    kernel_func = lambda x_i, x_j : np.exp(-gamma * (x_i - x_j) @ (x_i - x_j))

    sv = svm(0.1, kernel_func)
    # SVM学習
    sv.fit(X_train, y_train)
    # テストデータの予測
    y_pred = sv.predict(X_test)
    y_pred[y_pred == -1] = 0

    # 以下結果のプロット
    size = 100
    sc = plt.scatter(X_test[:, 0], X_test[:, 1],
                        size, y_pred, marker = ',', cmap='bwr')

    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1] ,
                edgecolors="black", marker="o" , color='r',
                label= "train data (y = 1)")
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1] ,
                edgecolors="black", marker="o" , color='b',
                label= "train data (y = -1)")
    plt.axis([xmin[0], xmax[0], xmin[1], xmax[1]])
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()