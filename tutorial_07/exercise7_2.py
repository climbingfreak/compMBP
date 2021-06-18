import numpy as np
from compMBP.tutorial_07.exercise7_1 import MatrixProductStates, task_decorator
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import functools
from compMBP.tutorial_05 import ed
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

sz = np.array([[1, 0], [0, -1]])


class AKLT(MatrixProductStates):
    def __init__(self, L):
        super(AKLT, self).__init__(L=L)
        self.L = L

        sqrt2 = 1 / np.sqrt(2)


        self.M_odd = np.zeros((1,2,2))
        self.M_odd[:,0,0] = sqrt2
        self.M_odd[:,1,1] = -sqrt2

        self.M_even = np.zeros((2,2,1))
        self.M_even[1,0,:] = 1
        self.M_even[0,1,:] = 1


    def get_singlet_mps(self):
        single_spin_basis = np.zeros((1,2,1))
        single_spin_basis[:,0,:] = 1

        singlet_mps = [single_spin_basis] + [self.M_odd, self.M_even] * ((self.L-2)//2) + [single_spin_basis]
        return singlet_mps

    @task_decorator
    def task2a(self):


        singlet_mps = self.get_singlet_mps()


        i, j = 1,2
        corr = self.correlate_sz(singlet_mps, sz, i, j)
        print("correlation between site {i:d} and {j:d}: {corr:0.5f}".format(i=i, j=j, corr=corr))

        correlation = self.get_correlation_matrix(singlet_mps, sz)
        correlation[abs(correlation)<1e-4] =0
        print(correlation)

    def get_correlation_matrix(self, mps, op):
        correlation = np.zeros((len(mps), len(mps)))
        for i in range(len(mps)):
            for j in range(len(mps)):

                correlation[i,j] = self.correlate_sz(mps,op, i,j)
        correlation[abs(correlation)<1e-4] =0
        return correlation

    @task_decorator
    def task2c(self):
        mps_ground = self.spin_1_project()

    @task_decorator
    def task2d(self):
        mps_ground = self.spin_1_project()
        norm_ground_state = self.overlap2(mps_ground)

        print('norm of ground state:', norm_ground_state)
        norm_ground_state = self.overlap2(mps_ground)

        mps_ground[1] /= np.sqrt(norm_ground_state) #TODO fix

        print(self.overlap2(mps_ground))
        Sz = np.array([[1,0,0],[0,0,0],[0,0,-1],])
        correlation = self.get_correlation_matrix(mps_ground, Sz)

        print(np.abs(correlation))
        plt.plot(range(len(mps_ground)), np.abs(correlation[0,:]))
        plt.yscale('log')

    def spin_1_project(self):
        mps = self.get_singlet_mps()
        P = np.zeros((2,3,2))
        P[:,0,:] = [[1,0],[0,0]]
        P[:,1,:] = [[0,1],[1,0]]/np.sqrt(2)
        P[:,2,:] = [[0,0],[0,1]]

        mps0 = []

        for i in range(0, self.L, 2):
            Meven_odd = np.tensordot(mps[i],mps[(i+1)], (2,0))
            Mn = np.tensordot(Meven_odd, P, ((1,2),(0,2))).transpose((0,2,1))
            mps0.append(Mn)

        self.print_shapes(mps0)

        return mps0

    @staticmethod
    def print_shapes(list_):
        print([m.shape for m in list_])

    def correlate_sz(self, mps, op, i, j):
        mps_i = self.apply_operator_on_site(mps, op, i)
        si = self.overlap2(mps, mps_i)

        mps_j = self.apply_operator_on_site(mps, op, j)
        sj = self.overlap2(mps, mps_j)
        mps_ij = self.apply_operator_on_site(mps_i, op, j)
        sisj = self.overlap2(mps, mps_ij)
        return sisj - si*sj


    def apply_operator_on_site(self, mps, op, site):
        operator_on_site = np.tensordot(op, mps[site], (0,1)).transpose(1,0,2)
        applied_mps = mps.copy()
        applied_mps[site] = operator_on_site
        return applied_mps





    @task_decorator
    def task2b(self):
        mps = self.get_singlet_mps()
        norm_singlet_mps = self.overlap2(mps, mps)
        print('singlet_norm:', norm_singlet_mps)
        pauli_z = np.array([[1, 0], [0, -1]])


if __name__ == '__main__':
    aklt = AKLT(L=30)
    aklt.task2a()
    aklt.task2b()
    aklt.task2c()
    aklt.task2d()
    plt.show()