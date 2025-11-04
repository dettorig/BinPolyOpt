import numpy as np
import networkx as nx
import itertools as it
import inequality_classifier
import cvxpy as cp

edge_dict = None
inc_matrix = None
FIXED_EDGE_ORDER = None
n_vertices = None
n_edges = None


def read_incidence_to_edge_dict(path):
    inc_matrix = np.loadtxt(path, dtype=int)

    edge_dict = {}
    for row in inc_matrix:
        vertex_indices = [i + 1 for i, val in enumerate(row) if val == 1]
        edge_name = 'e' + ''.join(str(i) for i in vertex_indices)
        nodes = {f"v{i}" for i in vertex_indices}
        edge_dict[edge_name] = nodes
    return edge_dict, inc_matrix

def build_intersection_graph_on_N(edge_dict, f, restrict_to_f=False):
    """
    Nodes = hyperedges in N = E \ {f}.
    Edge (e,g) exists if they intersect
    """
    G = nx.Graph()
    N = [h for h in edge_dict if h != f]
    G.add_nodes_from(N)
    fset = edge_dict[f]
    for e, g in it.combinations(N, 2):
        inter = edge_dict[e] & edge_dict[g]
        if restrict_to_f:
            inter = inter & fset
        if inter:
            G.add_edge(e, g, S=set(inter)) 
    return G

def all_spanning_trees(G):
    if not nx.is_connected(G):
        return []
    n = G.number_of_nodes()
    nodes = set(G.nodes())
    trees = []
    all_edges = list(G.edges())
    for subset in it.combinations(all_edges, n - 1):
        T = nx.Graph()
        T.add_nodes_from(nodes)
        T.add_edges_from(subset)
        if nx.is_tree(T):
            for u, v in T.edges():
                T[u][v]["S"] = set(G[u][v]["S"])
            trees.append(T)
    return trees

def describe_tree_vs_graph(G, T):
    norm = lambda e: tuple(sorted(e))
    tree_edges = sorted(norm(e) for e in T.edges())
    removed = []
    for u, v, data in G.edges(data=True):
        if not T.has_edge(u, v):
            removed.append((min(u, v), max(u, v), sorted(data["S"])))
    removed.sort()
    return {"tree_edges": tree_edges, "removed_edges": removed}

def check_tree_duplicates(G):
    trees = all_spanning_trees(G)
    keys = [tuple(sorted(tuple(sorted(e)) for e in T.edges())) for T in trees]
    total = len(keys)
    unique = len(set(keys))
    dups = total - unique
    print(f"Total trees: {total}, Unique keys: {unique}, Duplicates: {dups}")
    if dups:
        from collections import Counter
        for k, count in Counter(keys).items():
            if count > 1:
                print("Duplicate tree key:", k, "count:", count)
    return dups, trees

def build_lp_matrices(filename, n_vertices, n_edges):
    """
    Builds coefficient matrix A and RHS vector c from .poi.ieq file.
    
    Parameters:
    -----------
    filename : str
        Path to the .poi.ieq file
    n_vertices : int
        Number of vertices
    n_edges : int
        Number of edges
    
    Returns:
    --------
    A : numpy.ndarray
        Coefficient matrix where each row represents one inequality
    c : numpy.ndarray
        Vector of right-hand side values
    """
    vertex_range = range(1, n_vertices + 1)
    edge_range = range(n_vertices + 1, n_vertices + n_edges + 1)
    
    # Use existing classifier to parse inequalities
    classified = inequality_classifier.classify_poi_ieq_file(
        filename, vertex_range, edge_range, output_txt=None
    )
    
    n_vars = n_vertices + n_edges
    
    # Initialize matrices
    A = []
    c = []
    
    # Process each inequality
    for _, _, coeffs, rhs, _ in classified:
        # Create row vector for this inequality
        row = np.zeros(n_vars)
        for idx, val in coeffs.items():
            row[idx - 1] = val  # Convert 1-based to 0-based indexing
        
        A.append(row)
        c.append(rhs)
    
    return np.array(A), np.array(c)

def read_fixed_edge_order(path, keys=None, key_prefix="x"):
    import re
    mapping = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = [s.strip() for s in line.split(":", 1)]
            if not left.startswith(key_prefix):
                continue
            # take first token after colon (robust to extra text)
            edge = right.split()[0].rstrip(",")
            mapping[left] = edge

    if keys is None:
        # sort by numeric suffix of the key (x12, x13, ...)
        def idx(k):
            m = re.search(r"(\d+)$", k)
            return int(m.group(1)) if m else float("inf")
        items = sorted(mapping.items(), key=lambda kv: idx(kv[0]))
    else:
        items = [(k, mapping[k]) for k in keys if k in mapping]

    return [edge for _, edge in items]

FIXED_EDGE_ORDER = None

def fixed_variable_order(n_vertices=n_vertices, edge_order=FIXED_EDGE_ORDER):
    verts = [f"v{i}" for i in range(1, n_vertices + 1)]
    names = verts + edge_order
    index = {name: i for i, name in enumerate(names)} 
    return names, index

def check_edge_names_against_fixed(edge_dict, edge_order=FIXED_EDGE_ORDER):
    missing = [e for e in edge_order if e not in edge_dict]
    extra = [e for e in edge_dict if (e not in edge_order and e.startswith("e"))]
    if missing:
        raise ValueError(f"Edge(s) missing from incidence but required by mapping: {missing}")
    if extra:
        print(f"[warn] Extra edges ignored mapping: {extra}")

def compute_Pf_relative_to_N(edge_dict, f):
    fset = set(edge_dict[f])
    covered_by_others = set().union(*(edge_dict[e] for e in edge_dict if e != f))
    return fset - covered_by_others

def canonical_representative_choice(T):
    rep = {}
    for u, v in T.edges():
        S = sorted(T[u][v]["S"], key=lambda w: int(w[1:]))
        rep[tuple(sorted((u, v)))] = S[0]
    return rep

def representative_choices_for_tree(T):
    edges = [tuple(sorted(e)) for e in T.edges()]
    per_edge = []
    for u, v in edges:
        S = sorted(T[u][v]["S"], key=lambda w: int(w[1:]))
        per_edge.append([(u, v, w) for w in S])
    reps = []
    for combo in it.product(*per_edge):
        reps.append({(u, v): w for (u, v, w) in combo})
    return reps

def build_spanningtree_coeffs(edge_dict, f, T, rep_assignment, var_index):
    nvars = len(var_index)
    a = np.zeros(nvars, dtype=int)

    # +1 on all edges in N = E\{f} (only those in FIXED_EDGE_ORDER)
    for e in FIXED_EDGE_ORDER:
        if e != f:
            a[var_index[e]] += 1

    # -1 on z_f
    a[var_index[f]] -= 1

    # -1 on chosen representatives z_{v_{pe}}
    for vrep in rep_assignment.values():
        a[var_index[vrep]] -= 1

    # +1 on uncovered vertices P_f
    Pf = compute_Pf_relative_to_N(edge_dict, f)
    for v in Pf:
        a[var_index[v]] += 1

    b = len(Pf)
    meta = {
        "f": f,
        "tree_edges": sorted(tuple(sorted(e)) for e in T.edges()),
        "rep_assignment": dict(rep_assignment),
        "Pf": sorted(Pf, key=lambda x: int(x[1:])),
    }
    return a, b, meta

def generate_spanningtree_for_f(edge_dict, inc_matrix, f, enumerate_reps=True, dedupe=True):
    Gf = build_intersection_graph_on_N(edge_dict, f, restrict_to_f=False)
    trees = all_spanning_trees(Gf)

    _, var_index = fixed_variable_order(n_vertices=inc_matrix.shape[1], edge_order=FIXED_EDGE_ORDER)

    cuts = []
    for T in trees:
        rep_list = representative_choices_for_tree(T) if enumerate_reps else [canonical_representative_choice(T)]
        for rep_assignment in rep_list:
            a, b, meta = build_spanningtree_coeffs(edge_dict, f, T, rep_assignment, var_index)
            
            cuts.append({
                "a": a,
                "b": b,
                "meta": meta,
                "rep": tuple(sorted(rep_assignment.values())),        
                "rep_map": dict(rep_assignment)                      
            })

    if dedupe:
        seen, uniq = set(), []
        for item in cuts:
            # include the full per-edge representative mapping in the dedupe key
            rep_map_key = tuple(sorted(item["rep_map"].items()))
            key = (tuple(item["a"].tolist()), int(item["b"]), rep_map_key)
            if key not in seen:
                seen.add(key)
                uniq.append(item)
        cuts = uniq

    return cuts, Gf, trees

def collect_all_spanningtree(edge_dict, inc_matrix, enumerate_reps=True, dedupe=True):
    check_edge_names_against_fixed(edge_dict, FIXED_EDGE_ORDER)
    names, var_index = fixed_variable_order(n_vertices=inc_matrix.shape[1], edge_order=FIXED_EDGE_ORDER)

    all_rows, all_rhs, metas = [], [], []
    for f in FIXED_EDGE_ORDER:
        cuts_f, _, _ = generate_spanningtree_for_f(edge_dict, inc_matrix, f, enumerate_reps, dedupe)
        for item in cuts_f:
            all_rows.append(item["a"])
            all_rhs.append(item["b"])
            metas.append({"f": f, **item["meta"]})

    A_cand = np.vstack(all_rows) if all_rows else np.zeros((0, len(names)), dtype=int)
    b_cand = np.array(all_rhs, dtype=int) if all_rhs else np.zeros((0,), dtype=int)
    return A_cand, b_cand, metas, names, var_index

def compare_spanningtree_with_porta(A_cand, b_cand, A_porta, b_porta, var_names):
    """
    Compare spanning-tree candidates to PORTA inequalities assuming both are in a^T x ≤ b form.
    """

    import numpy as np

    def gcd_normalize(a, b):
        a = np.asarray(a, dtype=int).copy()
        b = int(b)
        pool = np.append(np.abs(a[a != 0]), abs(b))
        if pool.size:
            g = np.gcd.reduce(pool.astype(int))
            if g > 1:
                a //= g
                b //= g
        return a, b

    # Normalize by GCD
    cand_norm   = [gcd_normalize(a, b) for a, b in zip(A_cand,  b_cand)]
    porta_norm  = [gcd_normalize(a, b) for a, b in zip(A_porta, b_porta)]

    # Convert to hashable for set ops
    cand_set  = {(tuple(a.tolist()), int(b)) for a, b in cand_norm}
    porta_set = {(tuple(a.tolist()), int(b)) for a, b in porta_norm}

    matches       = cand_set & porta_set
    only_in_cand  = cand_set - porta_set
    only_in_porta = porta_set - cand_set

    print("\nComparison Results:")
    print(f"Total spanning-tree candidates: {len(cand_set)}")
    print(f"Total PORTA inequalities:       {len(porta_set)}")
    print(f"Matching inequalities:          {len(matches)}")
    print(f"Only in candidates:             {len(only_in_cand)}")
    print(f"Only in PORTA:                  {len(only_in_porta)}")

    def _fmt_ineq(a_tuple, b):
        terms = []
        for coef, var in zip(a_tuple, var_names):
            if coef == 0: 
                continue
            terms.append(f"+{var}" if coef == 1 else
                        (f"-{var}" if coef == -1 else f"{coef:+d}{var}"))
        return " ".join(terms) + f" ≤ {b}"

    if only_in_cand:
        print("\nInequalities present only in candidates:")
        for a, b in sorted(only_in_cand):
            print(" ", _fmt_ineq(a, b))

    if only_in_porta:
        print("\nInequalities present only in PORTA:")
        for a, b in sorted(only_in_porta):
            print(" ", _fmt_ineq(a, b))

    return {
        "matches": matches,
        "only_in_candidates": only_in_cand,
        "only_in_porta": only_in_porta,
        "stats": {
            "total_candidates": len(cand_set),
            "total_porta": len(porta_set),
            "num_matches": len(matches),
            "num_only_candidates": len(only_in_cand),
            "num_only_porta": len(only_in_porta),
        },
    }

def check_dominance(A_porta, b_porta, a_candidate, b_candidate, tol=0.001):
    """
    Check if a candidate inequality (a_candidate^T x ≤ b_candidate) is dominated 
    by the PORTA system (A_porta x ≤ b_porta).
    
    Parameters:
    -----------
    A_porta : numpy.ndarray
        Matrix of PORTA inequality coefficients
    b_porta : numpy.ndarray
        RHS vector of PORTA inequalities
    a_candidate : numpy.ndarray
        Coefficient vector of candidate inequality
    b_candidate : float/int
        RHS of candidate inequality
    tol : float
        Numerical tolerance
        
    Returns:
    --------
    dict with:
        status : str ('strictly_dominated', 'zero_dominated', 'violated', 'failed')
        violation : float or None
        x_witness : numpy.ndarray or None
    """
    n = len(a_candidate)  # number of variables
    
    # Solve max{a^T x - b : Ax ≤ B}
    x = cp.Variable(n)
    objective = cp.Maximize(a_candidate @ x - b_candidate)
    constraints = [A_porta @ x <= b_porta]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver="ECOS")
        
        if prob.status == "optimal":
            violation = prob.value
            x_witness = x.value if abs(violation) > tol else None
            
            if violation < -tol:  # Clearly dominated
                status = "strictly_dominated"
            elif abs(violation) <= tol:  # Very close to zero
                status = "zero_dominated"
            else:  # violation > tol
                status = "violated"
                
            return {
                "status": status,
                "violation": float(violation),
                "x_witness": x_witness
            }
    except Exception as e:
        return {
            "status": "failed",
            "violation": None,
            "x_witness": None,
            "error": str(e)
        }
    
    return {
        "status": "failed",
        "violation": None, 
        "x_witness": None
    }


def analyze_dominance(A_porta, b_porta, only_in_cand, var_names, tol=0.001):
    """
    Analyze dominance of candidate inequalities against PORTA system.
    """
    print("\nDominance Analysis of **-inequalities not in PORTA:")
    print("-" * 80)

    strictly_dominated_count = 0
    zero_dominated_count = 0
    violated_count = 0
    failed_count = 0

    results = []
    for idx, (a, b) in enumerate(sorted(only_in_cand)):
        terms = []
        for coef, var in zip(a, var_names):
            if coef != 0:
                if coef == 1:
                    terms.append(f"+{var}")
                elif coef == -1:
                    terms.append(f"-{var}")
                else:
                    terms.append(f"{coef:+d}{var}")
        ineq_str = " ".join(terms) + f" ≤ {b}"
        
        # Check dominance
        result = check_dominance(A_porta, b_porta, np.array(a), b, tol)
        
        if result["status"] == "failed":
            status = "LP FAILED"
            failed_count += 1
        elif result["status"] == "strictly_dominated":
            status = f"STRICTLY DOMINATED (violation: {result['violation']:.6f})"
            strictly_dominated_count += 1
        elif result["status"] == "zero_dominated":
            status = f"ZERO DOMINATED (violation: {result['violation']:.6f})"
            zero_dominated_count += 1
        else:  # violated
            status = f"VIOLATED (violation: {result['violation']:.6f})"
            violated_count += 1
            
        print(f"\nInequality {idx+1}:")
        print(f"  {ineq_str}")
        print(f"  Status: {status}")
        
        if result["x_witness"] is not None:
            print("  Witness point:")
            for var, val in zip(var_names, result["x_witness"]):
                if abs(val) > 1e-5:
                    print(f"    {var}: {val:.6f}")
        
        results.append({
            "inequality": ineq_str,
            "status": result["status"],
            "violation": result["violation"],
            "witness": result["x_witness"]
        })
    
    print("\nSummary:")
    print(f"Total inequalities analyzed: {len(only_in_cand)}")
    print(f"Strictly dominated (opt < -ε): {strictly_dominated_count}")
    print(f"Zero dominated (|opt| ≤ ε): {zero_dominated_count}")
    print(f"Violated (opt > ε): {violated_count}")
    print(f"LP failures: {failed_count}")
    
    return results


def test_dominance_with_porta():
    """
    Test dominance of all spanningtree-inequalities that are not in PORTA system.
    """
    # Load PORTA system
    filename = "MPG11,9.poi.ieq"
    A_porta, b_porta = build_lp_matrices(filename, n_vertices=n_vertices, n_edges=n_edges)
    
    # Generate spanningtree-candidates
    A_cand, b_cand, metas, var_names, var_index = collect_all_spanningtree(
        edge_dict, inc_matrix,
        enumerate_reps=True,
        dedupe=True
    )
    
    
    print("Comparing spanningtree-inequalities with PORTA system...")
    comparison = compare_spanningtree_with_porta(A_cand, b_cand, A_porta, b_porta, var_names)
    only_in_cand = comparison["only_in_candidates"]
    
    if not only_in_cand:
        print("\nNo spanningtree-inequalities found outside PORTA system. Nothing to analyze.")
        return []
    
    # Analyze dominance
    print(f"\nAnalyzing dominance of {len(only_in_cand)} spanningtree-inequalities not in PORTA...")
    results = analyze_dominance(A_porta, b_porta, only_in_cand, var_names)
    return results