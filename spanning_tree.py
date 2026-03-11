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

def compute_Pf_relative_to_F(edge_dict, f, F_edges):
    """
    Pf(F) = f \\ union_{e in F_edges} e
    """
    fset = set(edge_dict[f])
    covered_by_F = set()
    for e in F_edges:
        covered_by_F |= set(edge_dict[e])
    return fset - covered_by_F


def build_spanningtree_coeffs_for_cycle(edge_dict, f, F_edges, T, rep_assignment, var_index):
    """
    Build coefficients for the tree-spanning inequality (Theorem 1) for fixed (f, F, T):
        sum_{e in F} z_e
      - sum_{(f',e') in T} z_{v_{f'e'}}
      - z_f
      + sum_{v in Pf(F)} z_v
      <= |Pf(F)|
    """
    nvars = len(var_index)
    a = np.zeros(nvars, dtype=int)

    # +1 on edges in F only
    for e in F_edges:
        a[var_index[e]] += 1

    # -1 on z_f
    a[var_index[f]] -= 1

    # -1 on chosen representative vertices for tree edges
    for vrep in rep_assignment.values():
        a[var_index[vrep]] -= 1

    # +1 on uncovered vertices Pf(F)
    Pf = compute_Pf_relative_to_F(edge_dict, f, F_edges)
    for v in Pf:
        a[var_index[v]] += 1

    b = len(Pf)
    meta = {
        "f": f,
        "F_edges": list(F_edges),
        "tree_edges": sorted(tuple(sorted(e)) for e in T.edges()),
        "rep_assignment": dict(rep_assignment),
        "Pf": sorted(Pf, key=lambda x: int(x[1:])),
    }
    return a, b, meta


def collect_spanningtree_inequalities_from_gamma_cycles(edge_dict, inc_matrix, per_focus,
                                                        enumerate_reps=True, dedupe=True):
    """
    Build candidate inequalities from gamma-cycle induced spanning trees.

    per_focus: output of generate_trees_for_cycles_per_focus(...), each record must contain:
      - 'focus'
      - 'F_edges'
      - 'tree' (networkx tree with edge attribute S)

    Returns:
      A_cand, b_cand, metas, names, var_index
    """
    names, var_index = fixed_variable_order(
        n_vertices=inc_matrix.shape[1],
        edge_order=FIXED_EDGE_ORDER
    )

    all_rows, all_rhs, metas = [], [], []

    for rec in per_focus:
        f = rec["focus"]
        F_edges = rec["F_edges"]
        T = rec["tree"]

        if enumerate_reps:
            rep_list = representative_choices_for_tree(T)
        else:
            rep_list = [canonical_representative_choice(T)]

        for rep_assignment in rep_list:
            a, b, meta = build_spanningtree_coeffs_for_cycle(
                edge_dict=edge_dict,
                f=f,
                F_edges=F_edges,
                T=T,
                rep_assignment=rep_assignment,
                var_index=var_index
            )

            item = {
                "a": a,
                "b": b,
                "meta": {
                    **meta,
                    "cycle_edges": rec.get("cycle_edges"),
                    "canonical_cycle": rec.get("canonical_cycle"),
                },
                "rep_map": dict(rep_assignment),
            }

            all_rows.append(item["a"])
            all_rhs.append(item["b"])
            metas.append(item)

    if not all_rows:
        A_cand = np.zeros((0, len(names)), dtype=int)
        b_cand = np.zeros((0,), dtype=int)
        return A_cand, b_cand, [], names, var_index

    if dedupe:
        seen = set()
        keep_idx = []
        for i, item in enumerate(metas):
            rep_map_key = tuple(sorted(item["rep_map"].items()))
            key = (tuple(item["a"].tolist()), int(item["b"]), rep_map_key)
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append(i)

        A_cand = np.vstack([all_rows[i] for i in keep_idx])
        b_cand = np.array([all_rhs[i] for i in keep_idx], dtype=int)
        metas_out = [metas[i]["meta"] for i in keep_idx]
    else:
        A_cand = np.vstack(all_rows)
        b_cand = np.array(all_rhs, dtype=int)
        metas_out = [m["meta"] for m in metas]

    return A_cand, b_cand, metas_out, names, var_index



def compare_spanningtree_with_porta(A_cand, b_cand, A_porta, b_porta, var_names):
    """
    Compare spanning-tree candidates to PORTA inequalities assuming both are in a^T x ≤ b form.
    """

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
    print("\nDominance Analysis of spanning-tree inequalities not in PORTA:")
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
    Test dominance of spanning-tree inequalities generated from gamma-cycle induced (f, F, T).
    """
    # Load PORTA system
    filename = "MPG11,9.poi.ieq"
    A_porta, b_porta = build_lp_matrices(filename, n_vertices=n_vertices, n_edges=n_edges)

    # Build per-focus tree records from gamma cycles
    all_per_focus = []
    g_by_focus = find_gamma_cycles_per_focus(
        edge_dict, min_length=3, max_length=len(edge_dict), max_cycles=None, debug=False
    )
    for f, cycles_f in g_by_focus.items():
        pf = generate_trees_for_cycles_per_focus(
            edge_dict,
            cycles_f,
            focus_edge=f,
            restrict_to_f=False,
            dedupe=True,
            max_trees_per_focus=None,
            verbose=False,
        )
        all_per_focus.extend(pf)

    # Generate spanning-tree candidates with the new collector
    A_cand, b_cand, metas, var_names, var_index = collect_spanningtree_inequalities_from_gamma_cycles(
        edge_dict=edge_dict,
        inc_matrix=inc_matrix,
        per_focus=all_per_focus,
        enumerate_reps=True,
        dedupe=True,
    )

    print("Comparing spanningtree-inequalities with PORTA system...")
    comparison = compare_spanningtree_with_porta(A_cand, b_cand, A_porta, b_porta, var_names)
    only_in_cand = comparison["only_in_candidates"]

    if not only_in_cand:
        print("\nNo spanningtree-inequalities found outside PORTA system. Nothing to analyze.")
        return []

    print(f"\nAnalyzing dominance of {len(only_in_cand)} spanningtree-inequalities not in PORTA...")
    results = analyze_dominance(A_porta, b_porta, only_in_cand, var_names)
    return results


def vertex_to_edges(edge_dict):
    """Return mapping vertex -> set(edges) for quick membership queries."""
    v2e = {}
    for e, verts in edge_dict.items():
        for v in verts:
            v2e.setdefault(v, set()).add(e)
    return v2e

def find_gamma_cycles_edgegraph(edge_dict, min_length=3, max_length=None, max_cycles=None,
                                focus_edge=None, debug=False):
    """
    Find gamma-cycles in the edge-adjacency graph (hyperedges as nodes).
    Returns cycles as edge sequences only (no vertex output).

    If focus_edge is set, only cycles containing that edge are returned and each
    cycle is oriented to start at focus_edge. Cycles are deduped by edge-set.
    """
    if max_length is None:
        max_length = len(edge_dict)

    v2e = vertex_to_edges(edge_dict)

    # Build edge-adjacency graph: nodes are edge names
    Eg = nx.Graph()
    Eg.add_nodes_from(edge_dict.keys())
    for a, b in it.combinations(edge_dict.keys(), 2):
        S = set(edge_dict[a]) & set(edge_dict[b])
        if S:
            Eg.add_edge(a, b, S=set(S))

    # Enumerate all simple cycles via directed expansion
    D = nx.DiGraph()
    D.add_nodes_from(Eg.nodes())
    for u, v in Eg.edges():
        D.add_edge(u, v)
        D.add_edge(v, u)

    raw_cycles = list(nx.simple_cycles(D))
    if debug:
        print("DEBUG: raw edge-cycles found:", len(raw_cycles))

    def canonical_edge_cycle(cyc):
        m = len(cyc)
        candidates = []
        for shift in range(m):
            cand = tuple(cyc[shift:] + cyc[:shift])
            candidates.append(cand)
            candidates.append(tuple(reversed(cand)))
        return min(candidates)

    def canonical_with_focus(cyc, fixed_edge):
        if fixed_edge not in cyc:
            raise ValueError("focus_edge not in cycle")
        m = len(cyc)
        candidates = []
        for i in range(m):
            if cyc[i] == fixed_edge:
                candidates.append(tuple(cyc[i:] + cyc[:i]))
        rev = list(reversed(cyc))
        for i in range(m):
            if rev[i] == fixed_edge:
                candidates.append(tuple(rev[i:] + rev[:i]))
        return list(min(candidates))

    seen_raw = set()
    seen_edge_sets = set()
    out = []

    for cyc in raw_cycles:
        m = len(cyc)
        if m < min_length or m > max_length:
            continue

        key = canonical_edge_cycle(cyc)
        if key in seen_raw:
            continue
        seen_raw.add(key)
        edges_seq = list(key)

        if focus_edge is not None and focus_edge not in edges_seq:
            continue

        # Dedupe by edge set: same hyperedges => same cycle for your use case
        edge_set_key = frozenset(edges_seq)
        if edge_set_key in seen_edge_sets:
            continue

        # Consecutive intersections must be non-empty
        S_list = []
        empty_intersection = False
        for i in range(m):
            Si = set(edge_dict[edges_seq[i]]) & set(edge_dict[edges_seq[(i + 1) % m]])
            if not Si:
                empty_intersection = True
                break
            S_list.append(list(Si))
        if empty_intersection:
            continue

        # Check existence of at least one valid vertex assignment for gamma condition
        cycle_edge_set = set(edges_seq)
        has_valid_assignment = False
        for combo in it.product(*S_list):
            verts = [None] * m
            for i in range(m):
                verts[(i + 1) % m] = combo[i]

            # Berge-cycle: vertices must be distinct
            if len(set(verts)) != m:
                continue

            # Gamma condition on v2..vm, only within the cycle edges
            valid = True
            for i in range(1, m):
                vi = verts[i]
                expected = {edges_seq[(i - 1) % m], edges_seq[i]}
                actual = v2e.get(vi, set()) & cycle_edge_set
                if actual != expected:
                    valid = False
                    if debug:
                        print(f"DEBUG: combo={combo} fails at vertex {vi}: expected {expected}, actual {actual}")
                    break

            if valid:
                has_valid_assignment = True
                break

        if not has_valid_assignment:
            continue

        seen_edge_sets.add(edge_set_key)
        if focus_edge is not None:
            out.append(canonical_with_focus(edges_seq, focus_edge))
        else:
            out.append(list(canonical_edge_cycle(edges_seq)))

        if max_cycles is not None and len(out) >= max_cycles:
            break

    if debug:
        print("DEBUG: gamma-cycles found:", len(out))
    return out


def find_gamma_cycles_per_focus(edge_dict, min_length=3, max_length=None, max_cycles=None, debug=False):
    """
    Return {focus_edge: [gamma cycles containing focus_edge]}.
    Each cycle is an edge sequence starting from that focus edge.
    """
    result = {}
    for focus in edge_dict:
        result[focus] = find_gamma_cycles_edgegraph(
            edge_dict,
            min_length=min_length,
            max_length=max_length,
            max_cycles=max_cycles,
            focus_edge=focus,
            debug=debug,
        )
    return result

def generate_trees_for_cycles_per_focus(edge_dict, cycles,
                                        focus_edge=None,
                                        restrict_to_f=False, dedupe=True,
                                        max_trees_per_focus=None, verbose=False):
    """
    For each gamma-cycle, generate spanning trees of I_H(F), where
    F = E(cycle) \\ {focus} and focus is the fixed edge f (Theorem 1).

    Input cycles can be:
      - list of edge-lists, e.g. ["e569","e349","e4578"]
      - list of dicts with key 'edges'
    If focus_edge is None, the function tries cyc['focus'] when present;
    otherwise it falls back to all edges in the cycle (backward-compatible).

    Returns a list of dicts with:
      cycle_edges, focus, F_edges, tree, tree_edges, canonical_cycle
    """

    def _canonicalize_cycle_with_focus(cycle_edges, focus):
        cycle = list(cycle_edges)
        if focus not in cycle:
            raise ValueError("focus_edge not in cycle")
        candidates = []

        # original orientation
        for i, e in enumerate(cycle):
            if e == focus:
                candidates.append(tuple(cycle[i:] + cycle[:i]))

        # reversed orientation
        rev = list(reversed(cycle))
        for i, e in enumerate(rev):
            if e == focus:
                candidates.append(tuple(rev[i:] + rev[:i]))

        return list(min(candidates))

    results = []
    seen = set()

    for cyc in cycles:
        edges_seq = cyc["edges"] if isinstance(cyc, dict) and "edges" in cyc else list(cyc)

        if focus_edge is not None:
            focuses = [focus_edge]
        elif isinstance(cyc, dict) and "focus" in cyc:
            focuses = [cyc["focus"]]
        else:
            # Backward compatibility (old behavior)
            focuses = list(edges_seq)

        for focus in focuses:
            if focus not in edges_seq:
                if verbose:
                    print(f"[skip] focus={focus} not in cycle={edges_seq}")
                continue

            # Canonical orientation of the cycle w.r.t. focus
            canonical = _canonicalize_cycle_with_focus(edges_seq, focus)

            # F = E(cycle) \ {focus}
            F_edges = [e for e in canonical if e != focus]
            if not F_edges:
                if verbose:
                    print(f"[skip] cycle={canonical}, focus={focus}: empty F")
                continue

            # Build IH(F): intersection graph on F only
            Gf_full = build_intersection_graph_on_N(edge_dict, focus, restrict_to_f=restrict_to_f)
            Gf_sub = Gf_full.subgraph(F_edges).copy()

            # Need connected IH(F) to have spanning trees
            if not nx.is_connected(Gf_sub):
                if verbose:
                    print(f"[skip] cycle={canonical}, focus={focus}: IH(F) disconnected")
                continue

            trees = all_spanning_trees(Gf_sub)
            if max_trees_per_focus is not None:
                trees = trees[:max_trees_per_focus]

            for T in trees:
                tree_edges = tuple(sorted(tuple(sorted(e)) for e in T.edges()))
                cycle_key = frozenset(canonical)  # permutation-invariant on cycle edges
                key = (cycle_key, focus, tree_edges)

                if dedupe and key in seen:
                    continue
                seen.add(key)

                results.append({
                    "cycle_edges": list(canonical),
                    "focus": focus,
                    "F_edges": list(F_edges),
                    "canonical_cycle": list(canonical),
                    "tree": T,
                    "tree_edges": tree_edges,
                })

            if verbose:
                print(f"cycle={canonical} focus={focus} -> {len(trees)} spanning trees of IH(F)")

    return results

def find_berge_cycles_edgegraph(edge_dict, min_length=2, max_length=None, max_cycles=None,
                                focus_edge=None, debug=False):
    """
    Find Berge-cycles in the edge-adjacency graph (hyperedges as nodes).
    Returns cycles as edge sequences only (no vertex output).

    If focus_edge is set, only cycles containing that edge are returned and each
    cycle is oriented to start at focus_edge. Cycles are deduped by edge-set.
    """
    if max_length is None:
        max_length = len(edge_dict)

    # Build edge-adjacency graph: nodes are edge names
    Eg = nx.Graph()
    Eg.add_nodes_from(edge_dict.keys())
    for a, b in it.combinations(edge_dict.keys(), 2):
        S = set(edge_dict[a]) & set(edge_dict[b])
        if S:
            Eg.add_edge(a, b, S=set(S))

    # Enumerate all simple cycles via directed expansion
    D = nx.DiGraph()
    D.add_nodes_from(Eg.nodes())
    for u, v in Eg.edges():
        D.add_edge(u, v)
        D.add_edge(v, u)

    raw_cycles = list(nx.simple_cycles(D))
    if debug:
        print("DEBUG: raw edge-cycles found:", len(raw_cycles))

    def canonical_edge_cycle(cyc):
        m = len(cyc)
        candidates = []
        for shift in range(m):
            cand = tuple(cyc[shift:] + cyc[:shift])
            candidates.append(cand)
            candidates.append(tuple(reversed(cand)))
        return min(candidates)

    def canonical_with_focus(cyc, fixed_edge):
        if fixed_edge not in cyc:
            raise ValueError("focus_edge not in cycle")
        m = len(cyc)
        candidates = []
        for i in range(m):
            if cyc[i] == fixed_edge:
                candidates.append(tuple(cyc[i:] + cyc[:i]))
        rev = list(reversed(cyc))
        for i in range(m):
            if rev[i] == fixed_edge:
                candidates.append(tuple(rev[i:] + rev[:i]))
        return list(min(candidates))

    seen_raw = set()
    seen_edge_sets = set()
    out = []

    for cyc in raw_cycles:
        m = len(cyc)
        if m < min_length or m > max_length:
            continue

        key = canonical_edge_cycle(cyc)
        if key in seen_raw:
            continue
        seen_raw.add(key)
        edges_seq = list(key)

        if focus_edge is not None and focus_edge not in edges_seq:
            continue

        # Dedupe by edge set: same hyperedges => same cycle for your use case
        edge_set_key = frozenset(edges_seq)
        if edge_set_key in seen_edge_sets:
            continue

        # Consecutive intersections must be non-empty
        S_list = []
        empty_intersection = False
        for i in range(m):
            Si = set(edge_dict[edges_seq[i]]) & set(edge_dict[edges_seq[(i + 1) % m]])
            if not Si:
                empty_intersection = True
                break
            S_list.append(list(Si))
        if empty_intersection:
            continue

        # Check existence of at least one valid vertex assignment for Berge condition
        has_valid_assignment = False
        for combo in it.product(*S_list):
            verts = [None] * m
            for i in range(m):
                verts[(i + 1) % m] = combo[i]

            # Berge-cycle: vertices must be distinct
            if len(set(verts)) != m:
                continue

            has_valid_assignment = True
            break

        if not has_valid_assignment:
            continue

        seen_edge_sets.add(edge_set_key)
        if focus_edge is not None:
            out.append(canonical_with_focus(edges_seq, focus_edge))
        else:
            out.append(list(canonical_edge_cycle(edges_seq)))

        if max_cycles is not None and len(out) >= max_cycles:
            break

    if debug:
        print("DEBUG: berge-cycles found:", len(out))
    return out

def find_berge_cycles_per_focus(edge_dict, min_length=2, max_length=None, max_cycles=None, debug=False):
    """
    Return {focus_edge: [Berge cycles containing focus_edge]}.
    Each cycle is an edge sequence starting from that focus edge.
    """
    result = {}
    for focus in edge_dict:
        result[focus] = find_berge_cycles_edgegraph(
            edge_dict,
            min_length=min_length,
            max_length=max_length,
            max_cycles=max_cycles,
            focus_edge=focus,
            debug=debug,
        )
    return result

