import scipy
import numpy as np
import matplotlib.pyplot as plt
from compMBP.tutorial_05 import ed
from matplotlib import cm
import functools


def task_decorator(func):
    def inner(*args, **kwargs):
        print('--'*30)
        print(func.__name__)
        func(*args, **kwargs)

    return inner


class MatrixProductStates():
    def __init__(self, L=14, J=1, g=1.5):
        self.L = L
        self.J = J
        self.g = g

    def initialize(self):
        sx_list = ed.gen_sx_list(L)
        sz_list = ed.gen_sz_list(L)
        self.H = ed.gen_hamiltonian(sx_list, sz_list, g=g, J=J)
        E0, self.psi0 = self.get_ground_state()
        self.chimax_exact = 2 ** (self.L // 2)

    def get_ground_state(self):
        E, vecs = scipy.sparse.linalg.eigsh(self.H, which='SA')
        psi0 = vecs[:, 0]
        return E, psi0

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
        """
        1) Contract left (bra and ket)
        2) Contract right (bra and ket)
        3) Cotract left with right
        :param product_state_1:
        :param product_state_2:
        :return:
        """

        if product_state_2 is None:
            product_state_2 = product_state_1

        left = np.tensordot(product_state_1[0], product_state_2[0].conj(), ((0, 1), (0, 1)))
        for Mn_1, Mn_2 in zip(product_state_1[1:], product_state_2[1:]):
            right = np.tensordot(Mn_1, Mn_2.conj(), (1, 1))
            left = np.tensordot(left, right, ((0, 1), (0, 2)))

        return left[0, 0]

    @staticmethod
    def overlap2(bra, ket=None):
        """
        More efficient:
        1) ket to left
        2) (ket and left) to bra
        :param bra:
        :param ket:
        :return:
        """

        if ket is None:
            ket = bra

        braket = np.ones((1, 1))
        for bra_i, ket_i in zip(bra, ket):
            ket_i = np.tensordot(braket, ket_i, (1, 0))
            braket = np.tensordot(bra_i, ket_i.conj(), ((0, 1), (0, 1)))

        return braket.item()

    @task_decorator
    def task1d(self):
        E, psi0 = self.get_ground_state()
        Mns_ex = self.compress(psi0, self.L, 256)

        Mns_compr = self.compress(psi0, self.L, 10)

        overlap_exact_exact = self.overlap2(Mns_ex)
        print('overlap of psi_exact with itself: ', overlap_exact_exact)

        overlap_exact_compr = self.overlap2(Mns_ex, Mns_compr)
        print('overlap of psi_exact with psi_compr: ', overlap_exact_compr)

    @task_decorator
    def task1f(self):
        psi = np.zeros(int(2 ** self.L))
        psi[0] = 1
        mps = self.compress(psi, self.L, self.chimax_exact)

        E, psi0 = self.get_ground_state()
        mps0 = self.compress(psi0, self.L, self.chimax_exact)

        overlap_up = self.overlap2(mps0, mps)
        print('overlap of all up and groundstate:',overlap_up)

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



if __name__ == '__main__':
    part1 = MatrixProductStates(L=14, J=1, g=1.5)
    part1.initialize()
    part1.task1a()
    part1.task1b()
    part1.task1c()
    part1.task1d()
    part1.task1f()
