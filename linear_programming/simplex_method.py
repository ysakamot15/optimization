import numpy as np
import copy

def swapNB(cB, cN, B, N, Bidx, Nidx, swap_pos_B, swap_pos_N):
    # idx swap
    tmp = Nidx[swap_pos_N]
    Nidx[swap_pos_N] = Bidx[swap_pos_B]
    Bidx[swap_pos_B] = tmp
    
    # N, B swap
    tmp = np.copy(N[:, swap_pos_N])
    N[:, swap_pos_N] = B[:, swap_pos_B]
    B[:, swap_pos_B] = tmp

    #cN, cB, swap
    tmp = cN[swap_pos_N]
    cN[swap_pos_N] = cB[swap_pos_B]
    cB[swap_pos_B] = tmp   

# 単体法本体
def Simplex(cB_init, cN_init, B_init, N_init, Bidx_init, Nidx_init, b):
    Bidx = copy.copy(Bidx_init)
    Nidx = copy.copy(Nidx_init)
    cB = copy.copy(cB_init)
    cN = copy.copy(cN_init)
    B = copy.copy(B_init)
    N = copy.copy(N_init)
    while(1):
        invB = np.linalg.inv(B)
        b_ = invB @ b
        y = invB.T @ cB
        cN_ = cN - N.T @ y
        # 全て0以下なら停止
        if (cN_ <= 0).all() == True:
            xB = b_
            return cB @ xB, xB, cB, cN, B, N, Bidx, Nidx
        # cN_の正の要素の中で一番最初の要素に
        # 対応する変数交換する非基底変数として選択
        k = np.where(cN_ > 0)[0][0]
        
        ak_ = invB @ N[:, k]
        # 全て0以下なら非有界
        if((ak_ <= 0).all() == True):
            raise Exception('unbounded error')
        # ak_の中で、b_i/a_k_iが最小になる要素iに
        # 対応する変数を交換する基底変数として選択
        tmp = np.ones(b_.shape) * np.inf
        tmp[ak_ > 0] = b_[ak_ > 0] / ak_[ak_ > 0]
        l = np.argmin(tmp)
        swapNB(cB, cN, B, N, Bidx, Nidx, l, k)

# 初期値を探す（2段階単体法の1段目）
def SearchFeasibleDicstionary(cB_init, cN_init, B_init, N_init, 
                              Bidx_init, Nidx_init, b):
    # bに負が含まれる場合、実行可能な辞書を探す
    bmin = np.min(b)
    if bmin < 0:
        Bidx_aux = copy.copy(Bidx_init)
        Nidx = copy.copy(Nidx_init)
        cB_aux = copy.copy(cB_init)
        cN = copy.copy(cN_init)
        B_aux = copy.copy(B_init)
        N = copy.copy(N_init)
        
        # 人工変数のindex
        artificial_var_idx = cB_init.shape[0] + cN_init.shape[0]

        # 人工変数を非基底変数にする
        Nidx_aux = copy.copy(Nidx)
        Nidx_aux.append(artificial_var_idx)

        # 人工変数に対応する要素と列を追加
        cN_aux = np.zeros(cN.shape[0] + 1)
        # min x_a -> max -x_a　という最適化問題を解くため係数は-1
        cN_aux[cN.shape[0]] = -1
        N_aux = np.concatenate([N, np.ones((N.shape[0], 1)) * -1], axis = 1)

        # bの最小値に対応する基底変数と人工変数を入れ替え
        k = np.argmin(b)
        swapNB(cB_aux, cN_aux, B_aux, N_aux, 
                Bidx_aux, Nidx_aux, k, N_aux.shape[1] - 1)

        # 補助問題を解いて実行可能解を見つける
        z, _, res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx = \
         Simplex(cB_aux, cN_aux, B_aux, N_aux, Bidx_aux, Nidx_aux, b)
        
        # 最適解が負なら実行不能
        if z < 0:
            raise Exception('infeasible error')  
        # 得られた辞書のidxから人工変数を削除
        if artificial_var_idx in res_Nidx:
            # 人工変数が非基底変数にあるならそのまま削除
            res_Nidx.remove(artificial_var_idx)
        else:
            #退化して最適解の基底変数に人工変数が含まれる場合は
            # 非基底変数1つと交換する
            r = res_Bidx.index(artificial_var_idx)
            for i in range(len(res_Nidx)):
                swapNB(res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx, r, i)
                # 基底行列が正則か確かめる
                if(np.linalg.matrix_rank(res_B) == res_B.shape[0]):
                    res_Nidx.remove(artificial_var_idx)
                    break
                # 正則でなければ元に戻して、交換するindexを変えてもう一度
                swapNB(res_cB, res_cN, res_B, res_N, res_Bidx, res_Nidx, i, r)

        # 得られた基底、非基底変数の割り当てに基づいて
        # 基底、非基底行列、係数を再構成
        res_cNB = np.concatenate([cN_init, cB_init])[res_Nidx + res_Bidx]
        res_cN = res_cNB[:cN_init.shape[0]]
        res_cB = res_cNB[cN_init.shape[0]:]
        res_NB = np.concatenate([N_init, B_init],
                                        axis = 1)[:, res_Nidx + res_Bidx]
        res_N = res_NB[:, :cN_init.shape[0]]
        res_B = res_NB[:, cN_init.shape[0]:]
        return res_Bidx, res_Nidx, res_cB, res_cN, res_B, res_N
        
    # 負がなければそのままreturn
    return Bidx_init, Nidx_init, cB_init, cN_init, B_init, N_init

# max c.T @ x
# s.t. A @ x <= b
#      x >= 0
def SolveLP(c, A, b):
    cB = np.zeros(A.shape[0])
    cN = np.copy(c)
    N = np.copy(A)
    B = np.eye(A.shape[0])
    # 実際の変数のインデックス(0~n-1)
    actual_variable_idx = list(range(c.shape[0]))
    # スラック変数のインデックス(n~n+m-1)
    slack_variable_idx = list(range(c.shape[0], c.shape[0] + A.shape[0]))
    Nidx = copy.copy(actual_variable_idx)
    Bidx = copy.copy(slack_variable_idx)
   
    # 2段階単体法
    new_Bidx, new_Nidx, new_cB, new_cN, new_B, new_N = \
        SearchFeasibleDicstionary(cB, cN, B, N, Bidx, Nidx, b)
    z, xB_opt, cB_opt, cN_opt, B_opt, N_opt, Bidx_opt, Nidx_opt = \
        Simplex(new_cB, new_cN, new_B, new_N, new_Bidx, new_Nidx, b)
    
    # 得られた解から実際の変数（スラック変数以外の変数）を並び替えて出力
    x_opt = np.concatenate([xB_opt, np.zeros(cN_opt.shape[0])])
    x_actual_opt = x_opt[np.argsort(Bidx_opt + Nidx_opt)][actual_variable_idx]
    return z, x_actual_opt
