from scipy.sparse import csr_matrix
from scipy.sparse import kron
from scipy.sparse import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy


# constants
Id = sparse.csr_matrix(np.eye(2))
Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])


# functions
def singlesite_to_full(op, i, L):
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full

def gen_sx_list(L):
    return [singlesite_to_full(Sx, i, L) for i in range(L)]


def gen_sz_list(L):
    return [singlesite_to_full(Sz, i, L) for i in range(L)]


def gen_hamiltonian(sx_list, sz_list, g, J=1., bc="periodic"):
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))

    if bc == "periodic":
        for j in range(L):
            H = H - J *( sx_list[j] * sx_list[(j+1)%L])
            H = H - g * sz_list[j]
    elif bc == "open":
        for j in range(L-1):
            H = H - J *( sx_list[j] * sx_list[(j+1)])
            H = H - g * sz_list[j]

    return H

###################


# task 5.1 a)


L = 6
g = 1.5
sx_list = gen_sx_list(L)
sz_list = gen_sz_list(L)

H = gen_hamiltonian(sx_list, sz_list, g, J=1., bc='open')

def task_a():

    E_0, psi_0 = scipy.sparse.linalg.eigsh(H, k=1, M=None, sigma=None, which='SA')

    print("Eigenvalue ", E_0)
    print("Eigenvector ", psi_0)

    max_prob_idx = np.argmax(np.abs(psi_0))
    state_representation = bin(max_prob_idx)
    print(max_prob_idx, state_representation)




# task 5.1 b)
def task_b():
    k=1
    Es, psis = scipy.sparse.linalg.eigsh(H, k=k, M=None, sigma=None, which='SA')
    for i in range(k):
        psi_ab = np.reshape(psis[:, i], (2**(L//2), 2**(L//2)))
        print(psi_ab)


# task 5.1 c)

def task_c():
    scipy.linalg.svd




#################
#task 5.2

def flip(s, i, N):
    """Flip the bits of the state `s` at positions i and (i+1)%N."""
    return s ^ (1 << i | 1 << ((i+1) % N))


def translate(s, N):
    """Shift the bits of the state `s` one position to the right (cyclically for N bits)."""
    bs = bin(s)[2:].zfill(N)
    return int(bs[-1] + bs[:-1], base=2)


def count_ones(s, N):
    """Count the number of `1` in the binary representation of the state `s`."""
    return bin(s).count('1')


def is_representative(s, k, N):
    """Check if |s> is the representative for the momentum state.

    Returns -1 if s is not a representative.
    If |s> is a representative, return the periodicity R,
    i.e. the smallest integer R > 0 such that T**R |s> = |s>."""
    t = s
    for i in range(N):
        t = translate(t, N)
        if t < s:
            return -1  # there is a smaller representative in the orbit
        elif (t == s):
            if (np.mod(k, N/(i+1)) != 0):
                return -1  # periodicty incompatible with k
            else:
                return i+1


def get_representative(s, N):
    """Find the representative r in the orbit of s and return (r, l) such that |r>= T**l|s>"""
    r = s
    t = s
    l = 0
    for i in range(N):
        t = translate(t, N)
        if (t < r):
            r = t
            l = i + 1
    return r, l


def calc_basis(N):
    """Determine the (representatives of the) basis for each block.

    A block is detemined by the quantum numbers `qn`, here simply `k`.
    `basis` and `ind_in_basis` are dictionaries with `qn` as keys.
    For each block, `basis[qn]` contains all the representative spin configurations `sa`
    and periodicities `Ra` generating the state
    ``|a(k)> = 1/sqrt(Na) sum_l=0^{N-1} exp(i k l) T**l |sa>``

    `ind_in_basis[qn]` is a dictionary mapping from the representative spin configuration `sa`
    to the index within the list `basis[qn]`.
    """
    basis = dict()
    ind_in_basis = dict()
    for sa in range(2**N):
        for k in range(-N//2+1, N//2+1):
            qn = k
            Ra = is_representative(sa, k, N)
            if Ra > 0:
                if qn not in basis:
                    basis[qn] = []
                    ind_in_basis[qn] = dict()
                ind_in_basis[qn][sa] = len(basis[qn])
                basis[qn].append((sa, Ra))
    return basis, ind_in_basis


def calc_H(N, J, g):
    """Determine the blocks of the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonian ... ", end="", flush=True)
    basis, ind_in_basis = calc_basis(N)
    H = {}
    for qn in basis:
        M = len(basis[qn])
        H_block_data = []
        H_block_inds = []
        a = 0
        for sa, Ra in basis[qn]:
            H_block_data.append(-g * (-N + 2*count_ones(sa, N)))
            H_block_inds.append((a, a))
            for i in range(N):
                sb, l = get_representative(flip(sa, i, N), N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    k = qn*2*np.pi/N
                    H_block_data.append(-J*np.exp(-1j*k*l)*np.sqrt(Ra/Rb))
                    H_block_inds.append((b, a))
                # else: flipped state incompatible with the k value, |b(k)> is zero
            a += 1
        H_block_inds = np.array(H_block_inds)
        H_block_data = np.array(H_block_data)
        H_block = scipy.sparse.csr_matrix((H_block_data, (H_block_inds[:, 0], H_block_inds[:, 1])),
                                          shape=(M,M),dtype=np.complex)
        H[qn] = H_block
    print("done", flush=True)
    return H

def task_2_a():
    H = calc_H(N=10, J = 1, g = 0.1)
    E0_list = []
    for trans_eigenvalue, h in H.items():
        E0, vec = scipy.sparse.linalg.eigsh(h, k=1, M=None, sigma=None, which='SA')
        E0_list.append(E0[0])
    print(E0_list)

def translate_2(s, N):
    shifted_bit = 1 & s
    s = s >> 1
    return s | 2**(N-1) * shifted_bit



def task2_b():
    for i in range(10):
        N = np.random.randint(1,10)
        x = np.random.randint(0,2**(N-1))
        assert translate(x, N ) == translate_2(x,N)

def task2_c():
    H = calc_H(N=14, J = 1, g = 1)

    E_list = []
    # for trans_eigenvalue, h in H.items():
    #     E, vec = scipy.sparse.linalg.eigsh(h, k=5, M=None, sigma=None, which='SA')
    #     E_list.append(E)
    #     plt.plot(trans_eigenvalue*np.ones(len(E)), np.real(E), 'o', label="EV = {}".format(trans_eigenvalue))


    for qn in H:
        print(qn)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    task2_c()

    # Es, vec = scipy.sparse.linalg.eigsh(H, k=1, M=None, sigma=None, which='SA')
    # print(Es[0])
# Ls = [6, 8, 10, 12]
# gs = np.linspace(0., 2., 21)

# plt.figure()
# for L in Ls:
#     sx_list = gen_sx_list(L)
#     sz_list = gen_sz_list(L)
#     sxsx = sx_list[0]*sx_list[L//2]
#     corrs = []
#     for g in gs:
#         H = gen_hamiltonian(sx_list, sz_list, g, J=1.)
#         E, v = sparse.linalg.eigsh(H, k=3, which='SA')
#         v0 = v[:, 0]  # first column of v is the ground state
#         corr = np.inner(v0.conj(), sxsx*v0)
#         corrs.append(corr)
#     corrs = np.array(corrs)
#     plt.plot(gs, corrs, label="L={L:d}".format(L=L))
# plt.xlabel("g")
# plt.ylabel("C")
# plt.legend()
# plt.show()ci