import re
import numpy as np
import cvxpy as cp
import math

def parse_ieq_line(line):
    s = line.replace(' ', '').replace('≤', '<=') 
    coeffs = {}
    for sign, num, var in re.findall(r'([+-]?)(\d*)(x\d+)', s):
        coef = int(num) if num else 1
        if sign == '-':
            coef *= -1
        idx = int(var[1:])
        coeffs[idx] = coeffs.get(idx, 0) + coef   
    rhs_match = re.search(r'<=\s*(-?\d+)', s)
    rhs = int(rhs_match.group(1)) if rhs_match else 0
    return coeffs, rhs

def is_standard_linearization(coeffs, rhs, vertex_range, edge_range):
    """Classify inequality as standard or nonstandard linearization."""
    keys = list(coeffs.keys())

    # α-type: x_v ≤ 1
    if len(keys) == 1:
        k = keys[0]
        if k in vertex_range and coeffs[k] == 1 and rhs == 1:
            return 'alpha (x_v ≤ 1)'
        if k in edge_range and coeffs[k] == -1 and rhs == 0:
            return 'nu (-x_e ≤ 0)'

    # δ-type: -x_v + x_e ≤ 0
    if len(keys) == 2 and rhs == 0:
        vs = [k for k in keys if k in vertex_range]
        es = [k for k in keys if k in edge_range]
        if len(vs) == 1 and len(es) == 1:
            if coeffs[vs[0]] == -1 and coeffs[es[0]] == 1:
                return 'delta (-x_v + x_e ≤ 0)'

    # ε-type: ∑ x_v - x_e ≤ |e| - 1
    vs = [k for k in coeffs if k in vertex_range]
    es = [k for k in coeffs if k in edge_range]
    if len(es) == 1 and all(coeffs[v] == 1 for v in vs) and coeffs[es[0]] == -1:
        if rhs == len(vs) - 1:
            return 'epsilon (∑x_v - x_e ≤ |e|-1)'

    return 'nonstandard'

def classify_poi_ieq_file(filepath, vertex_range, edge_range, output_txt=None):
    """Classifies each inequality in a .poi.ieq file."""
    results = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract inequality lines
    ineq_lines = []
    in_section = False
    for line in lines:
        if 'INEQUALITIES_SECTION' in line:
            in_section = True
            continue
        if 'END' in line:
            break
        if in_section and '<=' in line:
            ineq_lines.append(line.strip())

    # Classify each inequality
    for i, line in enumerate(ineq_lines):
        coeffs, rhs = parse_ieq_line(line)
        label = is_standard_linearization(coeffs, rhs, vertex_range, edge_range)
        results.append((i+1, line, coeffs, rhs, label))

    # Output
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as out:
            out.write("INEQUALITIES_SECTION\n")
            for i, line, _, _, label in results:
                stripped = line[line.find(')') + 1:].strip()
                out.write(f"({i:3d}) {stripped:<65} ==> {label}\n")
            out.write("END\n")
    else:
        for i, line, _, _, label in results:
            print(f"({i:3d}) {line:<65} ==> {label}")

    return results

def count_standard_vs_nonstandard(classified_lines):
    """Counts standard inequality types and nonstandard ones."""
    standard_keywords = ("alpha", "nu", "delta", "epsilon")
    counts = {k: 0 for k in standard_keywords}
    nonstandard_count = 0

    for _, _, _, _, label in classified_lines:
        matched = False
        for k in standard_keywords:
            if k in label.lower():
                counts[k] += 1
                matched = True
                break
        if not matched:
            nonstandard_count += 1

    print("Standard Inequalities:")
    for k, v in counts.items():
        print(f"  {k:>7}: {v}")
    print(f"Nonstandard: {nonstandard_count}")

# solvers' helpers
_EPS = 1e-9

def _pick_lp_solver():
    """Pick a reasonable LP solver available to cvxpy."""
    for s in ("GLPK", "ECOS", "SCS"):
        if s in cp.installed_solvers():
            return s
    return None

def _solve_min_bu(A, b, c, solver=None):
    """
    Solve   min  b^T u   s.t.   A^T u = c,  u >= 0.
    Returns (status, ub_opt, u_vec).
    status in {'ok','infeasible','failed'}.
    """
    m, n = A.shape
    u = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(b @ u), [A.T @ u == c])

    try_solvers = []
    if solver:
        try_solvers.append(solver)
    s0 = _pick_lp_solver()
    if s0 and s0 not in try_solvers:
        try_solvers.append(s0)
    for s in try_solvers + ["GLPK", "ECOS", "SCS"]:
        if s is None:
            continue
        try:
            prob.solve(solver=s, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                val = float(b @ u.value)
                return "ok", val, np.asarray(u.value).reshape(-1)
            if prob.status in ("infeasible", "infeasible_inaccurate"):
                return "infeasible", np.inf, None
        except Exception:
            pass
    return "failed", np.inf, None

def _safe_floor(x):
    return np.floor(x + _EPS)

# CG rank tests

def cg_rank1_certificate(A, b, c, d, tol=1e-7, solver=None):
    """
    Check whether   c^T x <= d   is valid for the first CG closure C(P) of P={Ax<=b}.

    Theory: let u* solve  min b^T u  s.t. A^T u = c, u>=0.
            The rank-1 CG cut is   c^T x <= floor(b^T u*).
            Our inequality is valid for C(P) iff  floor(b^T u*) <= d.

    Returns dict with:
      in_rank1   : bool  (valid for C(P); i.e., rank <= 1)
      exact_rank1: bool  (rank == 1; we assume the input is nonstandard, so in_rank1 ⇒ exact)
      floor_rhs  : int   (floor(b^T u*))
      ub_opt     : float (b^T u*)
      dominates  : bool  (True if floor_rhs < d; our inequality is weaker than the CG cut)
      status     : str   ('ok', 'infeasible', 'failed')
      u          : np.ndarray | None  (multiplier; None if not 'ok')
    """
    stat, ub_opt, u = _solve_min_bu(A, b, c, solver)
    if stat != "ok":
        return dict(in_rank1=False, exact_rank1=False, floor_rhs=None,
                    ub_opt=None, dominates=False, status=stat, u=None)

    f = _safe_floor(ub_opt)
    in_rank1   = (f <= d + tol)
    # we are only testing nonstandard inequalities (rank >= 1),
    exact_rank1 = in_rank1
    dominates   = (f < d - tol)
    return dict(in_rank1=in_rank1, exact_rank1=exact_rank1, floor_rhs=int(f),
                ub_opt=ub_opt, dominates=dominates, status="ok", u=u)

def cg_rank2_test(A_ext, b_ext, c, d, tol=1e-7, solver=None):
    """
    Check whether  c^T x <= d  is valid for the CG-closure of the extended base:
      P_ext = {x : A_ext x <= b_ext}.
    This captures your 'rank-2 = combinations of rank-0 and rank-1 cuts' viewpoint.

    Returns dict with:
      in_rank2 : bool (valid for C(P_ext); i.e., rank <= 2 under our model)
      floor_rhs: int
      ub_opt   : float
      dominates: bool
      status   : str
    """
    stat, ub_opt, _ = _solve_min_bu(A_ext, b_ext, c, solver)
    if stat != "ok":
        return dict(in_rank2=False, floor_rhs=None, ub_opt=None,
                    dominates=False, status=stat)

    f = _safe_floor(ub_opt)
    return dict(in_rank2=(f <= d + tol), floor_rhs=int(f),
                ub_opt=ub_opt, dominates=(f < d - tol), status="ok")

# builders

def build_base_from_standard_with_ids(results, num_vars):
    """
    Build (A0, b0) from standard rows AND return row_to_lineidx:
    for each row of A0, the index of the corresponding line in `results`.
    """
    A_rows, b_vals, row_to_lineidx = [], [], []
    for line_idx, res in enumerate(results):
        if not res['category'].startswith('nonstandard'):
            row = np.zeros(num_vars)
            for idx, val in res['coeffs'].items():
                row[idx - 1] = val
            A_rows.append(row)
            b_vals.append(res['rhs'])
            row_to_lineidx.append(line_idx)
    if not A_rows:
        return np.zeros((0, num_vars)), np.zeros(0), []
    return np.vstack(A_rows), np.asarray(b_vals), row_to_lineidx

def build_extended_from_rank1(results, num_vars):
    """
    Build the 'rank-2 base' as:
      - all rank-0 (standard) inequalities, and
      - for each nonstandard inequality proven rank-1, the **actual** CG cut
        with RHS = floor(b^T u*), stored in res['r1']['floor_rhs'].
    """
    A_rows, b_vals = [], []

    # Add all standard (rank-0)
    for res in results:
        if not res['category'].startswith('nonstandard'):
            row = np.zeros(num_vars)
            for idx, val in res['coeffs'].items():
                row[idx - 1] = val
            A_rows.append(row)
            b_vals.append(res['rhs'])

    # Add all proven rank-1 cuts (same c, floored RHS)
    for res in results:
        if res.get('cg_rank') == 1 and res.get('r1', {}).get('floor_rhs') is not None:
            row = np.zeros(num_vars)
            for idx, val in res['coeffs'].items():
                row[idx - 1] = val
            A_rows.append(row)
            b_vals.append(res['r1']['floor_rhs'])

    if not A_rows:
        return np.zeros((0, num_vars)), np.zeros(0)
    return np.vstack(A_rows), np.asarray(b_vals)

# main pipeline

def rank_nonstandard_inequalities(results, num_vars, tol=1e-7, solver=None, verbose=True):
    """
    Given 'results' (each item has keys: 'coeffs', 'rhs', 'category', 'line'),
    assign CG ranks to **nonstandard** inequalities only:
       - rank = 1  if valid for C(P) (built from standard constraints)
       - rank = 2  if NOT in C(P) but valid for the closure of the base
                   augmented with all derived rank-1 cuts
       - otherwise: rank > 2 (unknown with our information)
    Adds to each nonstandard item:
       res['cg_rank'] in {1,2,None}
       res['r1']  : diagnostic dict from rank-1 check
       res['r2']  : diagnostic dict from rank-2 check (if needed)
    Returns the updated list and simple summary counts.
    """
    # Build base from standard (rank-0)
    A0, b0, row2line= build_base_from_standard_with_ids(results, num_vars)

    # ---- rank 1 ----
    r1_found, nonstd_total = 0, 0
    if verbose:
        print("Checking rank = 1 for nonstandard inequalities")
        print("-" * 80)
    for res in results:
        if res['category'].startswith('nonstandard'):
            nonstd_total += 1
            # form c and d
            c = np.zeros(num_vars)
            for idx, val in res['coeffs'].items():
                c[idx - 1] = val
            d = res['rhs']

            r1 = cg_rank1_certificate(A0, b0, c, d, tol=tol, solver=solver)
            res['r1'] = r1
            if r1['exact_rank1']:
                res['cg_rank'] = 1
                r1_found += 1
                if verbose:
                    tag = "dominated" if r1['dominates'] else "tight"
                    print(f"{res['line']:<70}  ==>  rank = 1 ({tag}); floor={r1['floor_rhs']}")
            else:
                res['cg_rank'] = None
                if verbose:
                    print(f"{res['line']:<70}  ==>  not in C(P) (status={r1['status']})")

    if verbose:
        print("-" * 80)
        print(f"Nonstandard total : {nonstd_total}")
        print(f"Rank = 1 found    : {r1_found}")
        print(f"To test for rank 2: {nonstd_total - r1_found}")

    # ---- rank 2 ----
    Aext, bext = build_extended_from_rank1(results, num_vars)
    r2_found = 0
    if verbose:
        print("\nChecking rank = 2 for remaining inequalities")
        print("-" * 80)
    for res in results:
        if res['category'].startswith('nonstandard') and res['cg_rank'] is None:
            c = np.zeros(num_vars)
            for idx, val in res['coeffs'].items():
                c[idx - 1] = val
            d = res['rhs']

            r2 = cg_rank2_test(Aext, bext, c, d, tol=tol, solver=solver)
            res['r2'] = r2
            if r2['in_rank2']:
                res['cg_rank'] = 2
                r2_found += 1
                if verbose:
                    tag = "dominated" if r2['dominates'] else "tight"
                    print(f"{res['line']:<70}  ==>  rank = 2 ({tag}); floor={r2['floor_rhs']}")
            else:
                if verbose:
                    print(f"{res['line']:<70}  ==>  rank > 2 (or unknown); status={r2['status']}")

    if verbose:
        print("-" * 80)
        print(f"Rank = 2 found : {r2_found}")
        print(f"Unresolved     : {(nonstd_total - r1_found) - r2_found}")

    return results, dict(nonstandard=nonstd_total, rank1=r1_found, rank2=r2_found)

def _tuples_to_results(classified_tuples):
    """
    Convert output of classify_poi_ieq_file (i, line, coeffs, rhs, label)
    to the dict format expected by rank_nonstandard_inequalities.
    """
    results = []
    for _, line, coeffs, rhs, label in classified_tuples:
        results.append({
            "line": line,
            "coeffs": coeffs,
            "rhs": rhs,
            "category": label
        })
    return results

def run_rank_from_classified(classified, vertex_range, edge_range,
                             solver="ECOS", tol=1e-7, verbose=True):
    """
    Use your existing 'classified' output to assign CG ranks and print results.
    """
    # Convert tuples -> dicts
    results = _tuples_to_results(classified)

    # Number of variables
    num_vars = max(max(vertex_range), max(edge_range))

    results, summary = rank_nonstandard_inequalities(
        results, num_vars, tol=tol, solver=solver, verbose=verbose
    )

    # Clean recap (nonstandard only)
    print("\n=== Final recap (nonstandard only) ===")
    unresolved = 0
    for r in results:
        if not r["category"].startswith("nonstandard"):
            continue
        rank = r.get("cg_rank")
        if rank == 1:
            info = r.get("r1", {})
            f = info.get("floor_rhs")
            ub = info.get("ub_opt")
            ub_str = f"{ub:.6g}" if ub is not None else "n/a"
            tag = "dominated" if info.get("dominates") else "tight"
            print(f"[rank=1 | {tag}] floor={f}  ub*={ub_str}  :: {r['line']}")
        elif rank == 2:
            info = r.get("r2", {})
            f = info.get("floor_rhs")
            ub = info.get("ub_opt")
            ub_str = f"{ub:.6g}" if ub is not None else "n/a"
            tag = "dominated" if info.get("dominates") else "tight"
            print(f"[rank=2 | {tag}] floor={f}  ub*={ub_str}  :: {r['line']}")
        else:
            unresolved += 1
            info = r.get("r2", {}) or r.get("r1", {})
            print(f"[rank≥3?] status={info.get('status','n/a')} :: {r['line']}")

    print("\nSummary (nonstandard only): "
          f"total={summary['nonstandard']}  rank1={summary['rank1']}  "
          f"rank2={summary['rank2']}  unresolved={unresolved}")

    return results, summary

# retrieve edge sets from incidence matrix

def edge_sets_from_incidence_matrix(M, vertex_range, edge_range):
    """
    M: numpy array shape (|E|, |V|) with entries in {0,1}.
    Returns: dict edge_var_index -> set({vertex indices}).
    """
    assert M.shape == (len(edge_range), len(vertex_range)), \
        f"Incidence shape {M.shape} != ({len(edge_range)},{len(vertex_range)})"
    edge_sets = {}
    for i, e in enumerate(edge_range):
        verts = {vertex_range[j] for j in range(len(vertex_range)) if M[i, j] == 1}
        edge_sets[e] = verts
    return edge_sets

# find running–intersection property

def _find_running_intersection_order(sets_list):
    """
    Backtracking search for a running–intersection ordering.
    sets_list: list of (idx, set_of_vertices).
    Returns a list of indices into sets_list if an order exists, else None.
    """
    m = len(sets_list)
    sets_only = [s for _, s in sets_list]

    def ok_to_place(k, order):
        S_union = set()
        for j in order:
            S_union |= sets_only[j]
        inter = sets_only[k] & S_union
        if not inter:
            return True
        for j in order:
            if inter <= sets_only[j]:
                return True
        return False

    order = []
    used = [False] * m

    def backtrack():
        if len(order) == m:
            return True
        for k in range(m):
            if used[k]:
                continue
            if ok_to_place(k, order):
                used[k] = True
                order.append(k)
                if backtrack():
                    return True
                order.pop()
                used[k] = False
        return False

    return order if backtrack() else None

# matching the vertices in N(s_i) to -1 vertices with dfs

def _max_bipartite_matching(graph, left_nodes, right_nodes):
    """
    graph: dict L -> set(R) (admissible edges only).
    Returns (size, Rmatch) where Rmatch[r]=l for matched right nodes.
    """
    Rmatch = {}

    def dfs(l, seen):
        for r in graph.get(l, ()):
            if r in seen:
                continue
            seen.add(r)
            if r not in Rmatch or dfs(Rmatch[r], seen):
                Rmatch[r] = l
                return True
        return False

    size = 0
    for l in left_nodes:
        if dfs(l, set()):
            size += 1
    return size, Rmatch

# RI detector 
def is_running_intersection_inequality(coeffs, rhs, vertex_range, edge_range, edge_sets,
                                       require_adjacent=True, verbose=False):
    """
    Decide whether (coeffs, rhs) is a Running-Intersection inequality.

    Exact LHS pattern (Def. 1):
      - edges:  -1 on a unique center e0; +1 on neighbors {e_i}; 0 otherwise
      - vertices: +1 on e0 \ (U_i e_i); for each nonempty N(e0∩e_i) pick ONE u_i∈N with coef -1;
                  all other vertices have coef 0
    RHS: n0 + #{i : N(e0∩e_i) ≠ ∅} - 1, where n0 = | e0 \ U_i e_i |.
    """
    # Edge pattern 
    edge_coefs = {e: coeffs.get(e, 0) for e in edge_range}
    centers = [e for e, a in edge_coefs.items() if a == -1]
    if len(centers) != 1:
        return False, {'reason': f'need exactly one edge with -1 (center); got {centers}'}
    e0 = centers[0]

    neighbors = [e for e, a in edge_coefs.items() if a == +1]
    other_edges_bad = [e for e, a in edge_coefs.items()
                       if e != e0 and e not in neighbors and a != 0]
    if other_edges_bad:
        return False, {'reason': f'unexpected nonzero edge coeffs: {other_edges_bad}'}

    if require_adjacent:
        for e in neighbors:
            if not (edge_sets[e0] & edge_sets[e]):
                return False, {'reason': f'neighbor {e} not adjacent to center {e0}'}

    # Build S = {s_i = e0 ∩ e_i} and find a running–intersection order
    S_pairs = []
    for e in neighbors:
        s = edge_sets[e0] & edge_sets[e]
        if not s:
            return False, {'reason': f'e0∩{e} is empty'}
        S_pairs.append((e, s))

    order_idx = _find_running_intersection_order([(i, s) for i, (_, s) in enumerate(S_pairs)])
    if order_idx is None:
        return False, {'reason': 'no running–intersection ordering exists for {e0∩e_i}'}

    S_pairs_ordered = [S_pairs[i] for i in order_idx]
    order_edges = [e for e, _ in S_pairs_ordered]
    order_sets  = [s for _, s in S_pairs_ordered]

    N_sets = []
    U_so_far = set()
    for s in order_sets:
        Nsi = s & U_so_far
        N_sets.append(Nsi)
        U_so_far |= s

    nonempty_ids = [i for i, Nsi in enumerate(N_sets) if Nsi]

    # Vertex pattern: +1 on e0\⋃neighbors, -1 one per nonempty N(s_i), 0 elsewhere
    vertex_coefs = {v: coeffs.get(v, 0) for v in vertex_range}
    if any(a not in (-1, 0, 1) for a in vertex_coefs.values()):
        return False, {'reason': 'vertex coefficients must be in {-1,0,1}'}

    cover_neighbors = set().union(*(edge_sets[e] for e in neighbors)) if neighbors else set()
    Vplus_required  = edge_sets[e0] - cover_neighbors
    Vplus_actual    = {v for v, a in vertex_coefs.items() if a == +1}
    Vminus_actual   = {v for v, a in vertex_coefs.items() if a == -1}

    if Vplus_actual != Vplus_required:
        return False, {'reason': 'vertex +1 set ≠ e0 \\ ⋃neighbors',
                       'Vplus_actual': sorted(Vplus_actual),
                       'Vplus_required': sorted(Vplus_required)}

    if len(Vminus_actual) != len(nonempty_ids):
        return False, {'reason': f'|Vminus|={len(Vminus_actual)} != number of nonempty N(s_i)={len(nonempty_ids)}',
                       'Vminus': sorted(Vminus_actual)}

    # Each -1 must be assigned to a distinct nonempty N(s_i)
    graph = {i: (N_sets[i] & Vminus_actual) for i in nonempty_ids}
    size, Rmatch = _max_bipartite_matching(graph, nonempty_ids, Vminus_actual)
    if size != len(nonempty_ids):
        return False, {'reason': 'cannot assign distinct u_i ∈ N(s_i) to the -1 vertices'}

    chosen_u = {i: r for r, i in Rmatch.items()}

    # No other vertex is allowed to be nonzero
    allowed_vertices = Vplus_required | Vminus_actual
    others = {v for v, a in vertex_coefs.items() if a != 0 and v not in allowed_vertices}
    if others:
        return False, {'reason': 'unexpected nonzero vertex coeffs outside required pattern',
                       'bad_vertices': sorted(others)}

    # RHS check 
    n0 = len(Vplus_required)
    expected_rhs = n0 + len(nonempty_ids) - 1
    if rhs != expected_rhs:
        return False, {'reason': f'RHS mismatch: expected {expected_rhs}, got {rhs}',
                       'n0': n0, 'num_nonempty': len(nonempty_ids)}

    # Success
    cert = dict(
        center=e0,
        neighbors=order_edges,
        order_edges=order_edges,
        order_sets=order_sets,
        N_sets=N_sets,
        chosen_u=chosen_u,     
        n0=n0,
        expected_rhs=expected_rhs
    )
    return True, cert

# annotate
def annotate_running_intersection(results, vertex_range, edge_range, edge_sets, verbose=True):
    """
    For each nonstandard inequality in 'results', mark whether it is a
    running–intersection inequality and attach the certificate/reason.
    """
    found = 0
    for r in results:
        if not r['category'].startswith('nonstandard'):
            continue
        ok, info = is_running_intersection_inequality(
            r['coeffs'], r['rhs'], vertex_range, edge_range, edge_sets, require_adjacent=True
        )
        r['is_RI'] = ok
        r['RI_info'] = info
        if verbose:
            if ok:
                found += 1
                print(f"[RI] center={info['center']}, neighbors={info['neighbors']} :: {r['line']}")
            else:
                print(f"[not RI] reason={info.get('reason','')} :: {r['line']}")
    if verbose:
        print(f"\nRunning–intersection inequalities found: {found}")
    return results

# helpers for beta-cycle detection
def _edge_adjacent(e1, e2, edge_sets):
    return len(edge_sets[e1] & edge_sets[e2]) > 0

def _build_edge_adjacency(edges, edge_sets):
    edges = list(edges)
    idx = {e:i for i,e in enumerate(edges)}
    adj = [set() for _ in edges]
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            if _edge_adjacent(edges[i], edges[j], edge_sets):
                adj[i].add(j); adj[j].add(i)
    return edges, adj

def _vertices_ok_condition_a(Ecycle, edge_sets):
    """(a) every vertex is in at most two edges among E(C)."""
    cnt = {}
    for e in Ecycle:
        for v in edge_sets[e]:
            cnt[v] = cnt.get(v, 0) + 1
            if cnt[v] > 2:
                return False
    return True

def _canonicalize_cycle(order):
    """Normalize a cyclic list up to rotation+reversal to deduplicate."""
    m = len(order)
    # all rotations of order
    rots = [tuple(order[i:]+order[:i]) for i in range(m)]
    # and of reversed order
    rrev = list(reversed(order))
    rots += [tuple(rrev[i:]+rrev[:i]) for i in range(m)]
    return min(rots)

def _all_cycle_orders(edges, adj, max_orders=10000):
    """
    Enumerate all Hamiltonian cycles (edge orders) in the edge-intersection graph.
    Deduplicate up to rotation + reversal.
    """
    n = len(edges)
    if n < 3: 
        return []
    seen = set()
    out  = []

    # try every start
    for s in range(n):
        path = [s]
        used = {s}
        def dfs():
            nonlocal out
            if len(path) == n:
                # must close cycle
                if path[0] in adj[path[-1]]:
                    order = [edges[i] for i in path]
                    key = _canonicalize_cycle(order)
                    if key not in seen:
                        seen.add(key)
                        out.append(order)
                return
            u = path[-1]
            for v in adj[u]:
                if v in used:
                    continue
                if len(path) == n-1 and (path[0] not in adj[v]):
                    continue
                used.add(v); path.append(v)
                dfs()
                path.pop(); used.remove(v)
        dfs()
        if len(out) >= max_orders:
            break
    return out

def _rotations_and_directions(order):
    m = len(order)
    for r in range(m):
        o1 = order[r:] + order[:r]
        yield o1
        yield list(reversed(o1))

def _floor_half(k): 
    return k // 2

# detector
def is_odd_beta_cycle_inequality(coeffs, rhs, vertex_range, edge_range, edge_sets,
                                 verbose=False, strict=False, max_orders=10000):
    # coefficients split
    edge_coef = {e: coeffs.get(e, 0) for e in edge_range}
    vert_coef = {v: coeffs.get(v, 0) for v in vertex_range}

    # allow only {-1,0,1} where nonzero
    if any(a not in (-1,0,1) for a in edge_coef.values() if a != 0):
        return False, {'reason':'edge coeffs not in {-1,0,1}'}
    if any(a not in (-1,0,1) for a in vert_coef.values() if a != 0):
        return False, {'reason':'vertex coeffs not in {-1,0,1}'}

    E_minus = {e for e,a in edge_coef.items() if a == -1}
    E_plus  = {e for e,a in edge_coef.items() if a == +1}
    if not E_minus:
        return False, {'reason':'no E^- edges'}
    k = len(E_minus)
    if k % 2 == 0:
        return False, {'reason':f'|E^-| even ({k})'}

    Ecycle = E_minus | E_plus
    m = len(Ecycle)
    if m < 3:
        return False, {'reason':f'm={m} < 3'}

    # (a) check on chosen edge set (does not depend on order)
    if not _vertices_ok_condition_a(Ecycle, edge_sets):
        return False, {'reason':'(a) violated: some vertex in >2 chosen edges'}

    # adjacency among these edges; enumerate all Hamiltonian cycles
    elist, adj = _build_edge_adjacency(Ecycle, edge_sets)
    all_orders = _all_cycle_orders(elist, adj, max_orders=max_orders)
    if not all_orders:
        return False, {'reason':'no Hamiltonian cycle in edge-intersection graph'}

    # unions used for S1 (not S2)
    U_minus = set().union(*(edge_sets[e] for e in E_minus)) if E_minus else set()
    U_plus  = set().union(*(edge_sets[e] for e in E_plus))  if E_plus  else set()
    S1 = U_minus - U_plus

    Vplus  = {v for v,a in vert_coef.items() if a == +1}
    Vminus = {v for v,a in vert_coef.items() if a == -1}
    if Vplus != S1:
        # If +1 vertex set is not S1, no need to try orders (orientation cannot fix S1).
        return False, {'reason':'vertex +1 set ≠ S1'}

    # try all cycle orders and their rotations/directions
    for base in all_orders:
        for order in _rotations_and_directions(base):
            if order[0] not in E_minus:
                continue

            # intersections of consecutive edges along 'order'
            inters = []
            ok_inters = True
            for i in range(m):
                e = order[i]; f = order[(i+1) % m]
                S = edge_sets[e] & edge_sets[f]
                if not S:
                    ok_inters = False; break
                inters.append(S)
            if not ok_inters:
                continue

            # enumerate all choices of one distinct vertex from each intersection
            chosen = []
            usedV  = set()
            def choose(idx):
                if idx == m:
                    # we have a concrete V(C) = set(chosen)
                    V_C = set(chosen)
                    S2  = V_C - U_minus

                    # LHS check now uses correct S2
                    if Vminus != S2:
                        return False

                    # RHS: count consecutive NEG pairs along this order
                    neg_pairs = sum(1 for i in range(m)
                                    if (order[i] in E_minus and order[(i+1)%m] in E_minus))
                    rhs_expected = len(S1) - neg_pairs + _floor_half(k)
                    if rhs != rhs_expected:
                        return False

                    # (b)(c)(d) conditions
                    if strict:
                        # build maps for f-sequence parity
                        pos = {order[i]: i+1 for i in range(m)}
                        neg_positions = sorted(pos[e] for e in E_minus)
                        p = max(neg_positions)
                        D = {order[i-1] for i in range(p+1, m+1)}
                        f_seq   = [order[i-1] for i in neg_positions]
                        f_index = {f_seq[i]: i+1 for i in range(len(f_seq))}

                        # (b)
                        for i in range(1, m+1):
                            ei = order[i-1]
                            if ei in E_plus and ei not in D:
                                neg_neighbors = [e for e in E_minus if _edge_adjacent(ei, e, edge_sets)]
                                allowed = {order[(i-2)%m], order[i%m]}
                                if any(e not in allowed for e in neg_neighbors):
                                    return False
                        # (c)
                        for e_d in D:
                            for e_neg in E_minus:
                                if e_d != e_neg and _edge_adjacent(e_d, e_neg, edge_sets):
                                    if f_index[e_neg] % 2 == 0:
                                        return False
                        # (d1) or (d2)
                        ok_d1 = True
                        for v in S1:
                            neg_containing = [e for e in E_minus if v in edge_sets[e]]
                            if len(neg_containing) == 1:
                                continue
                            if len(neg_containing) == 2:
                                i1 = f_index[neg_containing[0]]
                                i2 = f_index[neg_containing[1]]
                                if (i1 % 2) == (i2 % 2):
                                    ok_d1 = False; break
                            else:
                                ok_d1 = False; break
                        e1, em = order[0], order[-1]
                        ep, ep1 = order[p-1], order[p % m]
                        ok_d2 = all((not _edge_adjacent(e_neg, e_d, edge_sets)) or
                                    ((e_neg, e_d) in {(e1,em),(ep,ep1)})
                                    for e_neg in E_minus for e_d in D)
                        if not (ok_d1 or ok_d2):
                            return False

                    # success
                    cert = dict(
                        order=order, E_minus=sorted(E_minus), E_plus=sorted(E_plus),
                        k=k, S1=sorted(S1), S2=sorted(S2),
                        rhs_expected=rhs_expected, consecutive_neg_pairs=neg_pairs
                    )
                    raise StopIteration(cert)  # early exit via exception
                # choose one vertex from inters[idx], all distinct
                for v in inters[idx]:
                    if v in usedV: 
                        continue
                    usedV.add(v); chosen.append(v)
                    if choose(idx+1) is not False:
                        pass
                    chosen.pop(); usedV.remove(v)
                return False

            try:
                choose(0)
            except StopIteration as ex:
                return True, ex.args[0]

    return False, {'reason':'exhausted all orders/choices; no match'}

# annotate 
def annotate_odd_beta_cycle(results, vertex_range, edge_range, edge_sets,
                            verbose=True, strict=False, max_orders=10000):
    found = 0
    for r in results:
        if not r['category'].startswith('nonstandard'):
            continue
        ok, info = is_odd_beta_cycle_inequality(
            r['coeffs'], r['rhs'], vertex_range, edge_range, edge_sets,
            strict=strict, max_orders=max_orders
        )
        r['is_beta_cycle'] = ok
        r['beta_info'] = info
        if verbose:
            if ok:
                found += 1
                print(f"[β-cycle] order={info['order']}  E^-={info['E_minus']}  E^+={info['E_plus']}  :: {r['line']}")
            else:
                print(f"[not β-cycle] reason={info.get('reason','')} :: {r['line']}")
    if verbose:
        print(f"\nOdd β-cycle inequalities found: {found}")
    return results

def is_flower_inequality(
    coeffs, rhs, vertex_range, edge_range, edge_sets,
    allow_empty_T=False, normalize=True, strict=True, tol=0
):
    import math

    def _gcd_normalize(c, r):
        g = 0
        for v in c.values():
            g = math.gcd(g, int(abs(v)))
        g = math.gcd(g, int(abs(r)))
        if g > 1:
            return {k: int(v // g) for k, v in c.items()}, int(r // g)
        return c, r

    if normalize:
        coeffs, rhs = _gcd_normalize(coeffs, rhs)

    edge_coef = {e: int(coeffs.get(e, 0)) for e in edge_range}
    vert_coef = {v: int(coeffs.get(v, 0)) for v in vertex_range}

    # center must be the unique edge with coefficient -1
    E_minus = [e for e,a in edge_coef.items() if a == -1]
    if len(E_minus) != 1:
        return False, {'reason': 'need exactly one center edge with coeff -1', 'E_minus': E_minus}
    f = E_minus[0]

    # T = edges with +1
    T = {e for e,a in edge_coef.items() if a == +1}
    if not allow_empty_T and len(T) == 0:
        return False, {'reason': 'T must be nonempty (set allow_empty_T=True to permit empty T)'}
    if strict:
        other_edges = {e for e,a in edge_coef.items() if a != 0 and (e != f and e not in T)}
        if other_edges:
            return False, {'reason': 'edge coefficients outside {T and -f}', 'bad_edges': sorted(other_edges)}

    # adjacency & triple-intersection conditions
    f_set = edge_sets[f]
    for e in T:
        if not (f_set & edge_sets[e]):
            return False, {'reason': f'edge {e} in T not adjacent to f={f}'}
    T_list = list(T)
    for i in range(len(T_list)):
        for j in range(i+1, len(T_list)):
            if f_set & edge_sets[T_list[i]] & edge_sets[T_list[j]]:
                return False, {'reason': f'triple intersection not empty for f,{T_list[i]},{T_list[j]}'}

    union_T = set().union(*(edge_sets[e] for e in T)) if T else set()

    # vertices: +1 exactly on f \ ⋃T
    Vplus_required = f_set - union_T
    Vplus  = {v for v,a in vert_coef.items() if a == +1}
    Vminus = {v for v,a in vert_coef.items() if a == -1}
    if strict and Vminus:
        return False, {'reason': 'vertex negatives present', 'Vminus': sorted(Vminus)}
    if Vplus != Vplus_required:
        return False, {'reason': 'vertex +1 set mismatch',
                       'Vplus_actual': sorted(Vplus),
                       'Vplus_required': sorted(Vplus_required)}

    rhs_expected = len(f_set - union_T) + len(T) - 1
    if abs(rhs - rhs_expected) > tol:
        return False, {'reason': f'RHS mismatch: got {rhs}, expected {rhs_expected}',
                       'rhs_expected': rhs_expected}

    if strict:
        other_v = {v for v,a in vert_coef.items() if a != 0 and v not in Vplus_required}
        if other_v:
            return False, {'reason': 'vertex coefficients outside required set',
                           'bad_vertices': sorted(other_v)}

    info = dict(
        f=f,
        T=sorted(T),
        Vplus_required=sorted(Vplus_required),
        rhs_expected=rhs_expected,
        center_size=len(f_set),
        neighborhood_size=len(T)
    )
    return True, info

def annotate_flower(results, vertex_range, edge_range, edge_sets,
                   verbose=True, allow_empty_T=False, normalize=True, strict=True):
    """
    For each nonstandard inequality in 'results', mark whether it is a
    flower inequality and attach the certificate/reason.
    """
    found = 0
    for r in results:
        if not r['category'].startswith('nonstandard'):
            continue
        ok, info = is_flower_inequality(
            r['coeffs'], r['rhs'], vertex_range, edge_range, edge_sets,
            allow_empty_T=allow_empty_T, normalize=normalize, strict=strict
        )
        r['is_flower'] = ok
        r['flower_info'] = info
        if verbose:
            if ok:
                found += 1
                print(f"[Flower] center={info['f']}, T={info['T']} :: {r['line']}")
            else:
                print(f"[not Flower] reason={info.get('reason','')} :: {r['line']}")
    if verbose:
        print(f"\nFlower inequalities found: {found}")
    return results

def report_other_nonstandard(results, *, show_reasons=True, max_print=None):
    """
    Print nonstandard inequalities that are neither RI, β-cycle, nor Flower.

    Assumes you've already run:
        annotate_running_intersection(...)
        annotate_odd_beta_cycle(...)
        annotate_flower(...)

    Parameters
    ----------
    results : list[dict]
        Each dict has at least: 'line', 'coeffs', 'rhs', 'category'
        and (after annotation) optional flags/info:
            'is_RI', 'RI_info'
            'is_beta_cycle', 'beta_info'
            'is_flower', 'flower_info'
    show_reasons : bool
        If True, prints the first available rejection reason from each detector.
    max_print : int | None
        If set, limit the number of lines printed for the “other” group.

    Returns
    -------
    other_list : list[dict]
        The subset of result dicts that are nonstandard and in none of the 3 families.
    summary : dict
        Counts for bookkeeping.
    """
    total_nonstd = 0
    n_RI = n_beta = n_flower = 0
    missing_flags = {"is_RI": 0, "is_beta_cycle": 0, "is_flower": 0}

    others = []

    for r in results:
        if not r.get('category', '').startswith('nonstandard'):
            continue
        total_nonstd += 1

        is_RI       = r.get('is_RI', False)
        is_beta     = r.get('is_beta_cycle', False)
        is_flower   = r.get('is_flower', False)

        # Track if a flag is missing (to catch cases where an annotator wasn't run)
        if 'is_RI' not in r:            missing_flags['is_RI'] += 1
        if 'is_beta_cycle' not in r:    missing_flags['is_beta_cycle'] += 1
        if 'is_flower' not in r:        missing_flags['is_flower'] += 1

        n_RI     += int(is_RI)
        n_beta   += int(is_beta)
        n_flower += int(is_flower)

        if not (is_RI or is_beta or is_flower):
            others.append(r)

    # Print report
    print("=== Family coverage among NONSTANDARD inequalities ===")
    print(f"Total nonstandard: {total_nonstd}")
    print(f"  Running-Intersection: {n_RI}")
    print(f"  Odd β-cycle:          {n_beta}")
    print(f"  Flower:               {n_flower}")
    print(f"  Other (none of the above): {len(others)}")

    # Warn if any flags are missing (suggests an annotator wasn’t run)
    if any(missing_flags.values()):
        print("\n[Note] Some flags are missing (did you run all annotators?)")
        for k, v in missing_flags.items():
            if v:
                print(f"  missing {k}: {v} rows")

    # Print the “other” ones with reasons if requested
    if len(others) > 0:
        print("\n--- Nonstandard inequalities in NONE of the 3 families ---")
        count = 0
        for r in others:
            if max_print is not None and count >= max_print:
                print(f"... (stopped at max_print={max_print})")
                break
            line = r.get('line', '').strip()
            print(line)
            if show_reasons:
                ri_reason   = r.get('RI_info', {}).get('reason', 'n/a')
                beta_reason = r.get('beta_info', {}).get('reason', 'n/a')
                flower_reason = r.get('flower_info', {}).get('reason', 'n/a')
                print(f"   RI:    {ri_reason}")
                print(f"   β:     {beta_reason}")
                print(f"   Flower:{flower_reason}")
            count += 1

    summary = dict(
        total_nonstandard=total_nonstd,
        RI=n_RI,
        beta=n_beta,
        flower=n_flower,
        other=len(others),
        missing_flags=missing_flags,
    )
    return others, summary

def analyze_unclassified_rank1(rank_dicts, classified):
    print("\nAnalyzing unclassified rank-1 inequalities:")
    print("-" * 80)

    # Build mapping: A0 row index -> line index in `classified` (standard rows only)
    std_row_to_line = [
        i for i, (_, _, _, _, cat) in enumerate(classified)
        if not cat.startswith('nonstandard')
    ]

    found = 0
    skipped_standard = 0
    skipped_family = 0
    skipped_not_rank1 = 0

    for r in rank_dicts:
        # Skip conditions
        if not r['category'].startswith('nonstandard'):
            skipped_standard += 1
            continue
        if r.get('is_RI') or r.get('is_beta_cycle') or r.get('is_flower'):
            skipped_family += 1
            continue
        if not r.get('in_rank1'):
            skipped_not_rank1 += 1
            continue

        found += 1
        print(f"\nInequality {found}:")
        print(f"  {r['line']}")
        if r.get('ub_opt') is not None:
            print(f"  Optimal value: {r['ub_opt']:.6f}")

        # Track coefficient contributions
        coeff_contributions = {}

        u = r.get('u')
        if u is not None:
            print("\n  Standard inequalities with significant multipliers:")
            significant_idx = np.where(np.array(u) > 1e-5)[0]

            for ridx in significant_idx:
                # map A0 row (ridx) to the original line index in `classified`
                if ridx >= len(std_row_to_line):
                    continue
                line_idx = std_row_to_line[ridx]
                _, line, coeffs_std, _, cat = classified[line_idx]

                # cat here is guaranteed to be standard
                print(f"    #{line_idx+1} [{cat}] (coef={u[ridx]:.6f}):")
                print(f"      {line}")

                # Track contributions to each variable using the mapped standard row coeffs
                for var, coef in coeffs_std.items():
                    if var not in coeff_contributions:
                        coeff_contributions[var] = []
                    coeff_contributions[var].append(coef * u[ridx])

            # Show how coefficients are formed
            print("\n  Coefficient formation:")
            target_coeffs = r['coeffs']
            for var in sorted(target_coeffs.keys()):
                target = target_coeffs[var]
                contributions = coeff_contributions.get(var, [])
                total = sum(contributions)
                if contributions:
                    terms = " + ".join(f"({c:.6f})" for c in contributions)
                else:
                    terms = "(no standard contribution > 1e-5)"
                print(f"    x{var}: {target} = {terms} = {total:.6f}")

        print("-" * 80)

    print("\nAnalysis Summary:")
    print(f"Total inequalities processed: {len(rank_dicts)}")
    print(f"Skipped standard inequalities: {skipped_standard}")
    print(f"Skipped known family inequalities: {skipped_family}")
    print(f"Skipped non-rank-1 inequalities: {skipped_not_rank1}")
    print(f"Found unclassified rank-1 inequalities: {found}")