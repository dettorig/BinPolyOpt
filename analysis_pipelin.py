import numpy as np
import spanning_tree as st
import inequality_classifier as ic


def _gcd_normalize_dense(a, b):
    a = np.asarray(a, dtype=int).copy()
    b = int(b)
    pool = np.append(np.abs(a[a != 0]), abs(b))
    if pool.size:
        g = np.gcd.reduce(pool.astype(int))
        if g > 1:
            a //= g
            b //= g
    return tuple(a.tolist()), int(b)

def _coeffs_to_dense_tuple(coeffs, rhs, n_vars):
    a = np.zeros(n_vars, dtype=int)
    for k, v in coeffs.items():
        a[int(k) - 1] = int(v)
    return _gcd_normalize_dense(a, rhs)

def isolate_porta_spanning_tree_facets(
    edge_dict,
    inc_matrix,
    mapping_path,
    porta_ieq_path,
    min_len=2,
    max_len=None,
    enumerate_reps=True,
    print_limit=None,
    results_dicts=None,              
    exclude_known_families=False,    
):
    # setup variable order used by ST generator
    st.edge_dict = edge_dict
    st.inc_matrix = inc_matrix
    st.n_vertices = inc_matrix.shape[1]
    st.n_edges = inc_matrix.shape[0]
    st.FIXED_EDGE_ORDER = st.read_fixed_edge_order(mapping_path)

    # generate all ST inequalities from berge-cycles
    g_by_focus = st.find_berge_cycles_per_focus(
        edge_dict,
        min_length=min_len,
        max_length=(len(edge_dict) if max_len is None else max_len),
        max_cycles=None,
        debug=False,
    )

    per_focus = []
    for f, cycles_f in g_by_focus.items():
        per_focus.extend(
            st.generate_trees_for_cycles_per_focus(
                edge_dict=edge_dict,
                cycles=cycles_f,
                focus_edge=f,
                restrict_to_f=False,
                dedupe=True,
                max_trees_per_focus=None,
                verbose=False,
            )
        )

    A_st, b_st, metas_st, var_names, var_index = st.collect_spanningtree_inequalities_from_berge_cycles(
        edge_dict=edge_dict,
        inc_matrix=inc_matrix,
        per_focus=per_focus,
        enumerate_reps=enumerate_reps,
        dedupe=True,
    )

    st_keys = {_gcd_normalize_dense(a, b) for a, b in zip(A_st, b_st)}
    n_vars = st.n_vertices + st.n_edges

    # classify PORTA once 
    classified = ic.classify_poi_ieq_file(
        porta_ieq_path,
        vertex_range=range(1, st.n_vertices + 1),
        edge_range=range(st.n_vertices + 1, st.n_vertices + st.n_edges + 1),
        output_txt=None,
    )

    # use existing results_dicts
    if results_dicts is None:
        rank_dicts, _ = ic.run_rank_from_classified(
            classified,
            vertex_range=range(1, st.n_vertices + 1),
            edge_range=range(st.n_vertices + 1, st.n_vertices + st.n_edges + 1),
            solver="ECOS",
            tol=1e-7,
            verbose=False,
        )
    else:
        rank_dicts = results_dicts

    # build a fast lookup from normalized signature -> row tuple from classified
    sig_to_classified = {}
    for row in classified:
        i, line, coeffs, rhs, label = row
        sig = _coeffs_to_dense_tuple(coeffs, rhs, n_vars)
        sig_to_classified[sig] = row

    # ONLY nonstandard rank-1 rows matched as ST
    nonstandard_rank1_st = []
    seen_sigs = set()  # avoid duplicates
    for r in rank_dicts:
        if r.get("category") != "nonstandard":
            continue
        if not bool(r.get("in_rank1", False)):
            continue
        if exclude_known_families and (r.get("is_RI") or r.get("is_beta_cycle") or r.get("is_flower")):
            continue

        sig = _coeffs_to_dense_tuple(r["coeffs"], r["rhs"], n_vars)
        if sig not in st_keys:
            continue
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        # Attach classified row info so you still have idx/label/line once, no duplicate table
        row = sig_to_classified.get(sig, None)
        nonstandard_rank1_st.append({
            "sig": sig,
            "result": r,
            "classified_row": row,  # (i, line, coeffs, rhs, label) or None
        })

    # strict "unclassified" subset (outside RI/beta/flower)
    unclassified_rank1_st = [
        x for x in nonstandard_rank1_st
        if not (x["result"].get("is_RI") or x["result"].get("is_beta_cycle") or x["result"].get("is_flower"))
    ]

    # known-family subset among matched
    known_family_rank1_st = [
        x for x in nonstandard_rank1_st
        if (x["result"].get("is_RI") or x["result"].get("is_beta_cycle") or x["result"].get("is_flower"))
    ]

    # PRINT RESULTS
    print("\n=== Spanning-tree matches inside PORTA (nonstandard rank-1 only) ===")
    print(f"Generated ST inequalities (deduped): {len(st_keys)}")
    print(f"Matched nonstandard rank-1 PORTA rows: {len(nonstandard_rank1_st)}")
    print(f"  - matched known-family (RI/beta/flower): {len(known_family_rank1_st)}")
    print(f"  - matched unclassified rank-1: {len(unclassified_rank1_st)}")

    print("\n--- Matched nonstandard rank-1 rows (single list, no duplicates) ---")
    rows_to_print = nonstandard_rank1_st if print_limit is None else nonstandard_rank1_st[:print_limit]
    for x in rows_to_print:
        r = x["result"]
        row = x["classified_row"]
        flags = []
        if r.get("is_RI"): flags.append("RI")
        if r.get("is_beta_cycle"): flags.append("beta")
        if r.get("is_flower"): flags.append("flower")
        fam = ",".join(flags) if flags else "unclassified"

        if row is not None:
            i, line, coeffs, rhs, label = row
            print(f"[idx={i}] [{fam}] {line}")
        else:
            print(f"[idx=?] [{fam}] {r['line']}")

    if print_limit is not None and len(nonstandard_rank1_st) > print_limit:
        print(f"... ({len(nonstandard_rank1_st) - print_limit} more)")

    return {
        "st_keys": st_keys,
        "matched_nonstandard_rank1_st": nonstandard_rank1_st,
        "known_family_rank1_st": known_family_rank1_st,
        "unclassified_rank1_st": unclassified_rank1_st,
        "classified": classified,
        "rank_dicts": rank_dicts,
        "A_st": A_st,
        "b_st": b_st,
        "metas_st": metas_st,
        "var_names": var_names,
    }