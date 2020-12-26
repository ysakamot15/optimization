import numpy as np
import simplex_method as LP

def main():
    T = int(input())
    for t in range(T):
        n, m = input().strip().split(' ')
        n, m = int(n), int(m) 
        c = np.array(list(map(float, input().strip().split(' '))))
        A = []
        for i in range(m):
            row = list(map(float, input().strip().split(' ')))
            A.append(row)
        A = np.array(A)
        b = np.array(list(map(float, input().strip().split(' '))))
        try:
            print(LP.SolveLP(c, A, b))
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()