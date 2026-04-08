#!/usr/bin/env python3
"""
run_all_states.py

Orchestrator script that uses infomap_helpers.py to compute GS + SA for all states'
graphml files and save JSON + plot outputs.

Example:
 python run_all_states.py --states Texas,New_Hampshire,78 --input_dir state_networks_graphml \
    --overwrite --sa_t0 0.1 --sa_cooling 0.95 --sa_steps_per_T 5 --sa_min_T 1e-6 --sa_replicates 3 --rng_seed 42
"""
import os, sys, json, argparse
from glob import glob
from pathlib import Path
import numpy as np
import networkx as nx
from typing import Dict, Any

# import helpers (assumes infomap_helpers.py is in same dir or installed)
import infomap_helpers as ih

def ensure_dirs():
    Path("GS_results/plots").mkdir(parents=True, exist_ok=True)
    Path("SA_results/plots").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

def state_name_from_path(p):
    # keep original stem (case sensitive as filenames)
    return Path(p).stem

def save_json_atomic(obj, outpath):
    tmp = outpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, outpath)

def process_state(graphml_path: str, sa_params: Dict[str,Any], overwrite: bool=False):
    state = state_name_from_path(graphml_path)
    print(f"\n=== Processing {state} ===")
    try:
        G = nx.read_graphml(graphml_path)
        # normalize node ids to 0..N-1 if necessary
        nodes = list(G.nodes())
        try:
            ints = [int(n) for n in nodes]
            if set(ints) != set(range(len(ints))):
                mapping = {n:i for i,n in enumerate(nodes)}
                G = nx.relabel_nodes(G, mapping, copy=True)
        except Exception:
            mapping = {n:i for i,n in enumerate(nodes)}
            G = nx.relabel_nodes(G, mapping, copy=True)
        N = G.number_of_nodes()
        edges = [(u, v, float(d.get("weight",1.0))) for u,v,d in G.edges(data=True)]
        A = ih.build_adj_from_edge_list(N, edges, directed=True)
        # compute pi
        try:
            pi = ih.compute_pi_from_graph(G)
        except Exception:
            pi = np.ones(N)/N
        ensure_dirs()

        # ---------- Greedy Search ----------
        gs_json = f"GS_results/{state}_GS_results.json"
        gs_plots_dir = Path("GS_results/plots")
        if (not overwrite) and Path(gs_json).exists():
            print("GS result exists; loading partition from JSON...")
            with open(gs_json) as f:
                gs_obj = json.load(f)
            membership = gs_obj.get("membership_matrix")
            if membership and isinstance(membership, list):
                M = np.array(membership)
                if M.ndim == 2:
                    final_partition_gs = np.argmax(M, axis=1).tolist()
                else:
                    final_partition_gs = [int(x) for x in membership]
            else:
                raise RuntimeError("GS JSON found but membership_matrix missing/invalid.")
        else:
            print("Running greedy (this may take some time)...")
            final_partition_gs, history = ih.greedy_agglomerative_map(A, pi=pi)
            L_gs = ih.map_equation_L(A, final_partition_gs, pi=pi)
            ncomms = int(len(set(final_partition_gs)))
            membership_gs = ih.partition_to_membership_matrix(final_partition_gs).tolist()
            # save plots (greedy history, module sizes, sorted adjacency, network map)
            hist_path = gs_plots_dir / f"{state}_GS_greedy_history.png"
            ih.plot_greedy_history(history, save_path=str(hist_path))
            modsize_path = gs_plots_dir / f"{state}_GS_modulesize.png"
            #ih.plot_module_sizes(final_partition_gs, save_path=str(modsize_path))
            sorted_adj_path = gs_plots_dir / f"{state}_GS_sorted_adjacency.png"
            #ih.plot_sorted_adjacency(A, final_partition_gs, save_path=str(sorted_adj_path))
            # network map by module if geographic coords present — try and if fails produce non-geo plot
            map_path = gs_plots_dir / f"{state}_GS_modules_map.png"
            try:
                ih.plot_network_by_module(state, G=G, modules=final_partition_gs, save_path=str(map_path), verbose=False)
            except Exception:
                # fallback: simple network draw
                try:
                    ih.plot_network_by_module(state, G=G, modules=final_partition_gs, save_path=str(map_path), verbose=False)
                except Exception:
                    print("Warning: network map unavailable.")
            gs_obj = {
                "state": state,
                "method": "GS",
                "L_value": float(L_gs),
                "n_communities": ncomms,
                "membership_matrix": membership_gs,
                "plots": [str(hist_path), str(modsize_path), str(sorted_adj_path), str(map_path)]
            }
            save_json_atomic(gs_obj, gs_json)
            print(f"Saved GS results -> {gs_json}")

        # ---------- Simulated Annealing ---------- #
        sa_json = f"SA_results/{state}_SA_results.json"
        sa_plots_dir = Path("SA_results/plots")

        if (not overwrite) and Path(sa_json).exists():
            print("SA result exists; skipping unless overwrite.")
        else:
            print("Running simulated annealing (initialized from GS partition)...")
            init_part = final_partition_gs
            best_part, sa_history = ih.simulated_annealing_refine(
                A,
                init_part,
                pi=pi,
                T0=sa_params.get("T0", 0.05),
                cooling_rate=sa_params.get("cooling_rate", 0.98),
                steps_per_T=sa_params.get("steps_per_T", 1),
                min_T=sa_params.get("min_T", 1e-5),
                replicates=sa_params.get("replicates", 1),
                rng_seed=sa_params.get("rng_seed", None)
            )

            L_sa = ih.map_equation_L(A, best_part, pi=pi)
            ncomms_sa = int(len(set(best_part)))
            membership_sa = ih.partition_to_membership_matrix(best_part).tolist()

            # --- only keep the module map plot ---
            sa_map = sa_plots_dir / f"{state}_SA_modules_map.png"
            try:
                ih.plot_network_by_module(state, G=G, modules=best_part, save_path=str(sa_map), verbose=False)
            except Exception:
                print("Warning: SA map plot failed.")

            sa_obj = {
                "state": state,
                "method": "SA",
                "L_value": float(L_sa),
                "n_communities": ncomms_sa,
                "membership_matrix": membership_sa,
                "plots": [str(sa_map)],
                "history": sa_history,
                "sa_params": sa_params
            }
            save_json_atomic(sa_obj, sa_json)
            print(f"Saved SA results -> {sa_json}")

        return True, None
    except Exception as e:
        errpath = f"logs/{state}_error.json"
        with open(errpath, "w") as f:
            json.dump({"state": state, "error": str(e)}, f, indent=2)
        print(f"ERROR processing {state}: {e} (see {errpath})")
        return False, {"state": state, "error": str(e)}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--states", default="all", help="Comma-separated stems e.g. 'Texas,New_Hampshire,78' or 'all'")
    p.add_argument("--input_dir", default="state_networks_graphml")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sa_t0", type=float, default=0.05)
    p.add_argument("--sa_cooling", type=float, default=0.98)
    p.add_argument("--sa_steps_per_T", type=int, default=1)
    p.add_argument("--sa_min_T", type=float, default=1e-5)
    p.add_argument("--sa_replicates", type=int, default=1)
    p.add_argument("--rng_seed", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    sa_params = {
        "T0": args.sa_t0,
        "cooling_rate": args.sa_cooling,
        "steps_per_T": args.sa_steps_per_T,
        "min_T": args.sa_min_T,
        "replicates": args.sa_replicates,
        "rng_seed": args.rng_seed
    }
    all_files = sorted(glob(os.path.join(args.input_dir, "*.graphml")))
    if len(all_files) == 0:
        print("No graphml files found in", args.input_dir); return
    if args.states.lower() == "all":
        to_run = all_files
    else:
        wanted = [s.strip() for s in args.states.split(",") if s.strip()]
        stem_to_path = {Path(p).stem: p for p in all_files}
        to_run = []
        for w in wanted:
            if w in stem_to_path:
                to_run.append(stem_to_path[w])
            else:
                print(f"Warning: {w} not found in {args.input_dir}")
    results = {"success": [], "failed": []}
    for pth in to_run:
        ok, err = process_state(pth, sa_params, overwrite=args.overwrite)
        if ok:
            results["success"].append(state_name_from_path(pth))
        else:
            results["failed"].append(state_name_from_path(pth))
    print("Finished. Success:", results["success"], "Failed:", results["failed"])

if __name__ == "__main__":
    main()
