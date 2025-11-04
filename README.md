# BinPolyOpt
Tools to generate, classify and analyze hypergraph spanning-tree inequalities and their CG-rank/dominance given PORTA output.

This project builds polyhedral systems from a hypergraph incidence matrix, enumerates spanning-tree inequalities, classifies inequalities into standard families (α, ν, δ, ε), tests CG rank (rank‑1 / rank‑2) using CVXPY, detects structured families (running‑intersection, odd β‑cycle, flower), compares generated cuts to PORTA output, and runs dominance checks to see whether candidate spanning‑tree inequalities are implied by the PORTA system.

# Key features

- Read incidence matrix → hypergraph (read_incidence_to_edge_dict)
- Generate PORTA input and run xporta to create .poi.ieq (generate_polytope.py)
- Parse and classify inequalities from .poi.ieq (inequality_classifier.py)
- Enumerate intersection graphs, spanning trees and representative assignments (spanning_tree.py)
- Build candidate spanning‑tree cuts, normalize/dedupe and compare with PORTA cuts
- Dominance/LP checks using CVXPY to prove domination
- Annotators for three known families: Running‑Intersection, odd β‑cycle, Flower
