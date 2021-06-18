from compMBP.tutorial_08 import a_mps
from compMBP.tutorial_08 import b_model
from compMBP.tutorial_08 import c_tebd
from compMBP.tutorial_08 import c_tebd_real_time
import numpy as np
import matplotlib.pyplot as plt

sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])


def task_decorator(func):
    def inner(*args, **kwargs):
        print('--' * 30)
        print(func.__name__)
        func(*args, **kwargs)

    return inner


class TEBD:
    def __init__(self, L):
        self.L = L

    @task_decorator
    def task1a(self):
        all_up = a_mps.init_spinup_MPS(self.L)

        sz_exp = all_up.site_expectation_value(sz)
        sx_exp = all_up.site_expectation_value(sx)
        print('<sz>:', sz_exp)
        print('<sx>:', sx_exp)

    @task_decorator
    def task1b(self):
        all_right = self.init_spinright_MPS()
        sz_exp = all_right.site_expectation_value(sz)
        sx_exp = all_right.site_expectation_value(sx)
        print('<sz>:', sz_exp)
        print('<sx>:', sx_exp)

    @task_decorator
    def task1c(self):
        all_up = a_mps.init_spinup_MPS(self.L)
        all_right = self.init_spinright_MPS()

        for g in [0.5,1, 1.5]:
            print('g:', g)
            tfi = b_model.TFIModel(self.L, J=1, g=g)
            energy_right = tfi.energy(all_right)
            energy_up = tfi.energy(all_up)
            print('energy all right:',energy_right)
            print('energy all up:',energy_up)


    @task_decorator
    def task1d(self):
        c_tebd.example_TEBD_gs_finite(L=14, J=1., g=1.5)


    @task_decorator
    def task1e(self):
        E, psi, model, magnetizations, entropies = c_tebd_real_time.example_TEBD_gs_finite(self.L, J=1, g= 1, t=10)

        time = np.arange(len(magnetizations))/len(magnetizations) * 10

        plt.figure()
        plt.plot(time, magnetizations, 'o', c='b')
        plt.figure()
        plt.plot(time, entropies, 'o', c='b')


    def init_spinright_MPS(self):
        """Return a product state with all spins up as an MPS"""
        B = np.zeros([1, 2, 1], np.float64)
        B[0, 0, 0] = 1 / np.sqrt(2)
        B[0, 1, 0] = 1 / np.sqrt(2)
        S = np.ones([1], np.float64)
        Bs = [B.copy() for i in range(self.L)]
        Ss = [S.copy() for i in range(self.L)]
        return a_mps.MPS(Bs, Ss)


if __name__ == '__main__':
    tebd = TEBD(14)
    tebd.task1a()
    tebd.task1b()
    tebd.task1c()
    tebd.task1d()
    tebd.task1e()
    plt.show()