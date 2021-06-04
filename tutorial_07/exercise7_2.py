import numpy as np
from compMBP.tutorial_07.exercise7_1 import MatrixProductStates
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import functools

class AKLT(MatrixProductStates):
    def __init__(self, L):
        super(AKLT, self).__init__(L=L)
        self.L = L

    def compress_aklt(self, psi, L, chimax):
        psi_aR = np.reshape(psi, (1, 2 ** L))  # psi_aR[(i1,...in)]
        Ms = []
        for n in range(1, L + 1):
            chi_n, dim_R = psi_aR.shape
            print(n, dim_R)
            assert dim_R == 2 ** (L - (n - 1))
            psi_LR = np.reshape(psi_aR,
                                (chi_n * 2, dim_R // 2))  # psi_aR[i1;(i2,...,iL)] # psi_aR[(vn,i_n);(in+1,...,iL)]
            # M_n[i1;w1]*lambda_n[w1]*psi_tilde[w1;(i2,...,in)]
            # M_n[(vn,in),wn+1]*lambda_n[wn+1]*psi_tilde[wn+1;(in+1,...,iL)]
            M_n, lambda_n, psi_tilde = scipy.linalg.svd(psi_LR, full_matrices=False, lapack_driver='gesvd')
            # Truncation step

            if len(lambda_n) > chimax:
                keep = np.argsort(lambda_n)[::-1][:chimax]
                M_n = M_n[:, keep]  # M_n[(vn,in),wn+1]-->M_n[(vn,in),vn+1]
                lambda_n = lambda_n[keep]
                psi_tilde = psi_tilde[keep, :]
            chi_np1 = len(lambda_n)
            M_n = np.reshape(M_n, (chi_n, 2, chi_np1))
            # M_n[vn;in;vn+1]
            Ms.append(M_n)
            # psi_aR = lambda_n[:, np.newaxis] * psi_tilde[:,:]
            # psi_aR[vn+1,(in+1,...iL)]=lambda_n[vn+1,vn+1]*psi_tilde[vn+1;(in+1,...,iL)]
            psi_aR = np.tensordot(np.diag(lambda_n), psi_tilde, (1, 0))
        return Ms

    def task2a(self):
        singlet = 1/np.sqrt(2)*np.array([0,1,-1,0])

        singlet_product = singlet
        for i in range(1,self.L//2):
            singlet_product = np.kron(singlet_product, singlet)
        mps = self.compress_aklt(singlet_product, self.L, chimax=2)

        [print(mp.shape) for mp in mps]
        [print(mp[:,0,:]) for mp in mps]

        # print([mp.shape for mp in mps])







def task_decorator(func):
    def inner(*args, **kwargs):
        print('-----')
        print(func.__name__)
        func(*args, **kwargs)
        print('------')

    return inner




if __name__ == '__main__':
    aklt = AKLT(L=4)
    aklt.task2a()