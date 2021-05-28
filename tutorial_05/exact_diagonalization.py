# provides the functions of last weeks tasks


Id = sparse.csr_matrix(np.eye(2))
Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])

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
    elif bc = "open":
        for j in range(L-1):
            H = H - J *( sx_list[j] * sx_list[(j+1)])
            H = H - g * sz_list[j]

    return H
