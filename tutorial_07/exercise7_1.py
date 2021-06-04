import scipy
import numpy as np
import matplotlib.pyplot as plt
from compMBP.tutorial_05 import exact_diagonalization as ed
from matplotlib import cm
import functools


def task_decorator(func):
    def inner(*args, **kwargs):
        print('-----')
        print(func.__name__)
        func(*args, **kwargs)
        print('------')

    return inner


class MatrixProductStates():
    def __init__(self, L=10., J=1., g=0.1):
        self.L = L
        self.J = J
        self.g = g

        sx_list = ed.gen_sx_list(L)
        sz_list = ed.gen_sz_list(L)
        self.H = ed.gen_hamiltonian(sx_list, sz_list, g=g, J=J, bc='open')

    def product(self, Mns):

        product = Mns[0][:, 0, :]
        for Mn_idx in range(1, len(Mns)):
            product = product @ Mns[Mn_idx][:, 0, :]
        print(product)
        product = Mns[0][:, 1, :]
        for Mn_idx in range(1, len(Mns)):
            product = product @ Mns[Mn_idx][:, 0, :]
        print(product)

    def get_product(self, Mns):
        psi = 0
        for j in range(2):
            for i in range(2):
                product = Mns[0][:, j, :]
                for Mn_idx in range(1, len(Mns)):
                    print(product)
                    product = np.matmul(product, Mns[Mn_idx][:, i, :])
                psi += product
        return psi

    def print_number_of_floats(self, list_of_arrays):
        number_floats = sum([array.size for array in list_of_arrays])
        print('Number of floats:', number_floats)

    @task_decorator
    def task1a(self):
        E, psi0 = self.get_ground_state()

    @task_decorator
    def task1b(self):
        E, psi0 = self.get_ground_state()
        Mns = self.compress(psi0, self.L, 2 ** (self.L // 2))
        print([Mn.shape for Mn in Mns])
        self.print_number_of_floats(Mns)

    @task_decorator
    def task1c(self):
        E, psi0 = self.get_ground_state()
        print(psi0.shape)
        Mns = self.compress(psi0, self.L, 10)
        print([Mn.shape for Mn in Mns])
        self.print_number_of_floats(Mns)

    def overlap(self, product_state_1, product_state_2=None):

        if product_state_2 is None:
            product_state_2 = product_state_1

        left = np.tensordot(product_state_1[0], product_state_2[0].conj(), ((0, 1), (0, 1)))
        for Mn_1, Mn_2 in zip(product_state_1[1:], product_state_2[1:]):
            right = np.tensordot(Mn_1, Mn_2.conj(), (1, 1))
            left = np.tensordot(left, right, ((0, 1), (0, 2)))

        return left[0, 0]

    @task_decorator
    def task1d(self):
        E, psi0 = self.get_ground_state()
        Mns_ex = self.compress(psi0, self.L, 256)

        Mns_compr = self.compress(psi0, self.L, 10)

        overlap_exact_exact = self.overlap(Mns_ex)
        overlap_exact_compr = self.overlap(Mns_ex, Mns_compr)
        print('overlap of psi_exact with itself: ', overlap_exact_exact)
        print('overlap of psi_exact with psi_compr: ', overlap_exact_compr)

    @task_decorator
    def task1f(self):
        psi = np.zeros(int(2 ** self.L))
        psi[0] = 1
        E, psi0 = self.get_ground_state()
        mps = self.compress2(psi, self.L, 2)
        mps0 = self.compress2(psi0, self.L, 2 ** (self.L // 2))

        overlap_up = self.overlap(mps, mps0)
        print(overlap_up)

    def compress(self, psi, L, chimax):
        Mns = []
        for n in range(L):
            psi = psi.reshape(-1, 2 ** (L - n) // 2)
            M_n, lambda_n, psitilde = scipy.linalg.svd(psi, full_matrices=False, lapack_driver='gesvd')

            keep = np.argsort(lambda_n)[:: -1][: chimax]
            M_n = M_n[:, keep]
            lambda_ = lambda_n[keep]
            psitilde = psitilde[keep, :]

            M_n = M_n.reshape(-1, 2, M_n.shape[1])

            Mns.append(M_n)

            psi = lambda_[:, np.newaxis] * psitilde[:, :]
        return Mns

    def compress2(self, psi, L, chimax):
        psi_aR = np.reshape(psi, (1, 2 ** L))  # psi_aR[(i1,...in)]
        Ms = []
        for n in range(1, L + 1):
            chi_n, dim_R = psi_aR.shape
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

    def get_ground_state(self):
        E, vecs = scipy.sparse.linalg.eigsh(self.H, which='SA')
        psi0 = vecs[:, 0]
        return E, psi0 / np.linalg.norm(psi0)


if __name__ == '__main__':
    part1 = MatrixProductStates(L=14, J=1, g=1.5)
    part1.task1a()
    part1.task1b()
    part1.task1c()
    part1.task1d()
    part1.task1f()
