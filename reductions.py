import numpy as np
import networkx as nx

# helper functions to normalize signatures and lift original signatures to extended variable space
def normalize_dense(a, b):
    a = np.asarray(a, dtype=int).copy()
    b = int(b)
    pool = np.append(np.abs(a[a != 0]), abs(b))
    if pool.size:
        g = np.gcd.reduce(pool.astype(int))
        if g > 1:
            a //= g
            b //= g
    return tuple(a.tolist()), b

def lift_original_signature(sig, orig_var_names, ext_var_names):
    a_orig, b = sig
    coeff_by_name = dict(zip(orig_var_names, a_orig))
    a_ext = tuple(coeff_by_name.get(name, 0) for name in ext_var_names)
    return a_ext, b

def delta_signature(v, g, ext_var_index, n_vars):
    a = np.zeros(n_vars, dtype=int)
    a[ext_var_index[v]] = -1
    a[ext_var_index[g]] = 1
    return a, 0

def is_delta_extension_candidate(a_ext, b_ext, meta, orig_sigs_lifted, ext_data):
    F = set(meta["F_edges"])
    T = nx.Graph()
    T.add_nodes_from(F)
    T.add_edges_from([tuple(e) for e in meta["tree_edges"]])

    reps = {
        tuple(sorted(k)): v
        for k, v in meta["rep_assignment"].items()
    }

    for g in F:
        if T.degree[g] != 1:
            continue

        h = next(T.neighbors(g))
        rep_key = tuple(sorted((g, h)))
        v = reps[rep_key]

        d_a, d_b = delta_signature(
            v=v,
            g=g,
            ext_var_index=ext_data["var_index"],
            n_vars=len(ext_data["var_names"]),
        )

        base_a = np.asarray(a_ext, dtype=int) - d_a
        base_b = int(b_ext) - d_b
        base_sig = normalize_dense(base_a, base_b)

        if base_sig in orig_sigs_lifted:
            return True, {"g": g, "h": h, "v": v}

    return False, None

# helper functions for epsilon-reduction experiment
def lift_signature(sig, small_var_names, full_var_names):
    a_small, b = sig
    coeff_by_name = dict(zip(small_var_names, a_small))
    a_full = tuple(coeff_by_name.get(name, 0) for name in full_var_names)
    return a_full, b

def epsilon_signature(g, full_edge_dict, full_var_index, n_vars):
    a = np.zeros(n_vars, dtype=int)

    for v in full_edge_dict[g]:
        a[full_var_index[v]] += 1

    a[full_var_index[g]] -= 1
    b = len(full_edge_dict[g]) - 1

    return a, b