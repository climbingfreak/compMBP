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
        Mns = self.compress(psi0, self.L, 10)
        self.print_number_of_floats(Mns)

    @task_decorator
    def task1d(self):
        E, psi0 = self.get_ground_state()
        Mns_ex = self.compress(psi0, self.L, 256)
        Mns_compr = self.compress(psi0, self.L, 10)

        # psi_ex = self.get_product(Mns_ex)
        # psi_compr = self.get_product(Mns_compr)
        # print(psi_ex, psi_compr)

        a = self.product(Mns_ex)

        # a = functools.partial(functools.reduce(np.tensordot, Mns), axes=([1,2], [1, 0]))

    def compress(self, psi, L, chimax):
        Mns = []
        for n in range(L):
            psi = psi.reshape(-1, 2 ** (L - n) // 2)
            M_n, lambda_n, psitilde = scipy.linalg.svd(psi, full_matrices=False)

            keep = np.argsort(lambda_n)[:: -1][: chimax]
            M_n = M_n[:, keep]
            lambda_ = lambda_n[keep]
            psitilde = psitilde[keep, :]

            M_n = M_n.reshape(-1, 2, M_n.shape[1])

            Mns.append(M_n)

            psi = lambda_[:, np.newaxis] * psitilde[:, :]
        return Mns

    def get_ground_state(self):
        E, vecs = scipy.sparse.linalg.eigsh(self.H, which='SA')
        psi0 = vecs[:, 0]
        return E, psi0 / np.linalg.norm(psi0)


if __name__ == '__main__':
    part1 = MatrixProductStates(L=14, J=1, g=1.5)
    part1.task1a()
    # part1.task1c()
    part1.task1d()
