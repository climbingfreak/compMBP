import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from compMBP.tutorial_08 import a_mps
from compMBP.tutorial_08 import tfi_exact
from compMBP.tutorial_09 import b_model
from compMBP.tutorial_09 import d_dmrg


def task_decorator(func):
    def inner(*args, **kwargs):
        print('--' * 30)
        print(func.__name__)
        func(*args, **kwargs)

    return inner


sz = np.array([[1, 0], [0, -1]])


class DMRG:
    def __init__(self, L):
        self.L = L

    @task_decorator
    def task1a(self):
        model = b_model.TFIModel(self.L, J=1, g=1.5)
        print([w.shape for w in model.H_mpo])

    @task_decorator
    def task1b(self):
        mps_up = a_mps.init_spinup_MPS(14)
        model = b_model.TFIModel(self.L, J=1, g=1.5)
        dmrg_engine = d_dmrg.DMRGEngine(mps_up, model, chi_max=5)

        e = 1
        condition = 1e-6
        E = model.energy(dmrg_engine.psi)
        E_exact = tfi_exact.finite_gs_energy(self.L, J=1, g=1.5)

        Es = [E]
        epsilons = []
        iteration = 0
        while e > condition:
            dmrg_engine.sweep()
            E = model.energy(dmrg_engine.psi)
            e = abs(Es[-1] - E)
            Es.append(E)
            epsilons.append(e)
            iteration += 1
            print('convergence:', e)

        plt.title('Energy difference between energies of subsequent iterations')
        plt.plot(range(1, iteration + 1), epsilons, 'o-')
        plt.yscale('log')

        plt.figure()
        plt.title('Difference in energy with exact value')
        plt.plot(range(iteration + 1), np.abs(np.array(Es) - np.array([E_exact] * len(Es))), 'o-')
        plt.yscale('log')

    @staticmethod
    def full_dmrg_simulation(L, convergence_condition, J=1, g=1.5, chi_max=10):
        mps_up = a_mps.init_spinup_MPS(L)
        model = b_model.TFIModel(L, J=J, g=g)
        dmrg_engine = d_dmrg.DMRGEngine(mps_up, model, chi_max=chi_max)

        e = 1
        E = model.energy(dmrg_engine.psi)

        Es = [E]
        epsilons = []
        while e > convergence_condition:
            dmrg_engine.sweep()
            E = model.energy(dmrg_engine.psi)
            e = abs(Es[-1] - E)
            Es.append(E)
            epsilons.append(e)

        return model, dmrg_engine.psi

    @task_decorator
    def task1c(self):
        print(self.full_dmrg_simulation(self.L, 1e-8))

    @task_decorator
    def task1d(self):

        plt.figure()

        Ls = [8, 16, 32, 64, 96, 128]
        gs = [0.5, 1, 1.5]
        for g in gs:
            Ss = []
            for L in tqdm(Ls):
                model, pis0 = self.full_dmrg_simulation(L, 1e-8, g=g)
                S = pis0.entanglement_entropy()[L // 2]
                Ss.append(S)

            plt.plot(Ls, Ss, 'o-', label='g={}'.format(g))

            if g == 1:
                p = np.polyfit(np.log(Ls), Ss, 1)
                c = p[1] * 6

        print('c:', c)
        plt.xscale('log')
        plt.title('half-chain entanglement entropy')
        plt.ylabel('S(L/2)')
        plt.xlabel('L')
        plt.legend()

    @staticmethod
    def apply_operator_on_site(psi, op, site):
        operator_on_site = np.tensordot(op, psi.Bs[site], (0, 1)).transpose(1, 0, 2)
        applied_mps = psi.copy()
        applied_mps.Bs[site] = operator_on_site
        return applied_mps

    def overlap(self, bra, ket, i):
        if ket is None:
            ket = bra

        braket = np.tensordot(np.diag(bra.Ss[i]), np.diag(ket.Ss[i]).conj())
        braket = braket * np.ones((1, 1))
        for bra_i, ket_i in zip(bra.Bs, ket.Bs):
            ket_i = np.tensordot(braket, ket_i, (1, 0))
            braket = np.tensordot(bra_i, ket_i.conj(), ((0, 1), (0, 1)))

        return braket.item()

    def get_correlation(self, psi, op1, op2, i):
        L = len(psi.Bs)
        mps_op1 = self.apply_operator_on_site(psi, op1, i)

        correlations = []
        for j in list(range(i, L)):
            mps_op12 = self.apply_operator_on_site(mps_op1, op2, j)
            correlation = self.overlap(psi, mps_op12, i)
            correlations.append(correlation)

        return correlations

    def get_site_expectation(self, psi, op, i):
        psi_op = self.apply_operator_on_site(psi, op, i)
        expectation = self.overlap(psi, psi_op, i)
        return expectation

    @task_decorator
    def task1e(self):
        model, mps = self.full_dmrg_simulation(self.L, 1e-8)
        self.get_correlation(mps, sz, sz, 0)

    @task_decorator
    def task1f(self, L):
        plt.figure()
        gs = [0.5, 1., 1.1, 1.2, 1.5]

        sx = np.array([[0, 1], [1, 0]])

        for g in gs:
            model, psi0 = self.full_dmrg_simulation(L, g=g, convergence_condition=1e-8)
            correlations = self.get_correlation(psi0, sx, sx, L // 4)
            plt.plot(range(L - L // 4), correlations, label=g)

        plt.xlabel('|i-j|')
        plt.ylabel(r'correlation i, j for sigma_x')
        plt.legend()

    @task_decorator
    def task1g(self, L):
        plt.figure()
        gs = [0.5, 1, 1.1, 1.2, 1.5]

        sx = np.array([[0, 1], [1, 0]])
        for g in gs:
            model, psi0 = self.full_dmrg_simulation(L, g=g, convergence_condition=1e-8, chi_max=10)
            correlations = self.get_correlation(psi0, sx, sx, L // 4)
            expectation_sigma_x_j = [self.get_site_expectation(psi0, sx, j) for j in range(L // 4, L)]
            connected_correlations = np.array(correlations) - np.array(
                expectation_sigma_x_j) * self.get_site_expectation(psi0, sx, L // 4)
            plt.plot(range(L - L // 4), connected_correlations, label=g)

            if g != 1:
                p = np.polyfit(np.arange(L - L // 4), np.log(np.abs(connected_correlations)), 1)
                corr_length = -1 / p[0]
                print('g: {}, xi: {}'.format(g, corr_length.round(4)))

        plt.xlabel('|i-j|')
        plt.ylabel(r'Connected correlation i, j for sigma_x')
        plt.yscale('log')
        plt.legend()


if __name__ == '__main__':
    dmrg = DMRG(14)
    # dmrg.task1a()
    # dmrg.task1b()
    # dmrg.task1c()
    # dmrg.task1d()
    # dmrg.task1e()
    # dmrg.task1f(L=50)
    # dmrg.task1f(L=100)
    # dmrg.task1g(L=50)
    # dmrg.task1g(L=100)
    plt.show()
