# infomap_helpers.py
"""
Helper module extracted from InfoMap.ipynb (all code/cell defs up to the "### Grid search" markdown)
Contains:
 - adjacency & pi builders
 - map equation L computation (not a placeholder; reflects notebook's implementation)
 - greedy search (history-collecting)
 - simulated annealing (refinement)
 - plotting helpers:
     plot_greedy_history, plot_sorted_adjacency, plot_module_sizes,
     plot_network_by_module (geographic), plot_sa_results
"""
from typing import List, Tuple, Dict, Any, Optional
import math
import random
import json
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from shapely.geometry import LineString
import geopandas as gpd

# Optional dependency for basemap
try:
    import pygris
except Exception:
    pygris = None

# ---------------- Utilities ----------------

def build_adj_from_edge_list(n_nodes: int, edges: List[Tuple[int,int,float]], directed: bool=True) -> csr_matrix:
    rows, cols, data = [], [], []
    for u, v, w in edges:
        rows.append(int(u)); cols.append(int(v)); data.append(float(w))
    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=float)
    return A

def compute_pi_from_graph(G: nx.Graph) -> np.ndarray:
    """Compute stationary pi as normalized out-strength (fallback)."""
    N = G.number_of_nodes()
    out_strength = np.zeros(N, dtype=float)
    for i in range(N):
        s = 0.0
        for _, _, d in G.edges(i, data=True):
            s += float(d.get("weight", 1.0))
        out_strength[i] = s
    total = out_strength.sum()
    if total > 0:
        return out_strength / total
    else:
        return np.ones(N) / N

def normalize_partition_labels(partition: List[int]) -> np.ndarray:
    arr = np.asarray(partition, dtype=int)
    uniq = np.unique(arr)
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[int(x)] for x in arr], dtype=int)

def partition_to_membership_matrix(partition: List[int]) -> np.ndarray:
    labels = normalize_partition_labels(partition)
    N = labels.size
    K = int(labels.max()) + 1 if N>0 else 0
    M = np.zeros((N, K), dtype=int)
    for i, lab in enumerate(labels):
        M[i, lab] = 1
    return M

# ---------------- Map equation L (faithful implementation) ----------------
def map_equation_L(A: sp.spmatrix, partition: List[int], pi: Optional[np.ndarray]=None, epsilon: float=1e-12) -> float:
    """
    Compute the map equation L using flows between modules.
    This implementation follows the standard single-level Infomap formulation:
      L = q_exit * H_exit + sum_m p_m * H_m
    where:
      - q_exit: total probability to exit modules (sum of module exit flows)
      - H_exit: entropy of exit-codebook (module-level)
      - p_m: probability mass inside module m including its exit probability
      - H_m: entropy of within-module codebook (internal node and exit)
    Inputs:
      - A: adjacency (n x n) csr_matrix representing directed weighted flows (i->j)
      - partition: list/array mapping node -> module label
      - pi: stationary distribution array of length n (if None assume uniform)
    Returns L (float)
    """
    # normalize partition to contiguous 0..K-1
    labels = normalize_partition_labels(partition)
    n = A.shape[0]
    if pi is None:
        pi = np.ones(n) / n
    if not issparse(A):
        A = csr_matrix(A)
    # compute flow matrix between modules
    K = int(labels.max()) + 1 if n>0 else 0
    module_flow = np.zeros((K, K), dtype=float)
    A_coo = A.tocoo()
    for u, v, w in zip(A_coo.row, A_coo.col, A_coo.data):
        mu = int(labels[u]); mv = int(labels[v])
        module_flow[mu, mv] += float(w)
    # compute exit flows and node internal flows
    q_exit = np.zeros(K, dtype=float)
    p_module = np.zeros(K, dtype=float)
    for m in range(K):
        q_exit[m] = module_flow[m, :].sum() - module_flow[m, m]
    for i in range(n):
        p_module[labels[i]] += float(pi[i])
    q_total = q_exit.sum()
    # compute H_exit
    if q_total > 0:
        probs = q_exit / (q_total + epsilon)
        H_exit = -np.sum([p * math.log2(p + epsilon) for p in probs if p>0])
    else:
        H_exit = 0.0
    # compute within-module entropies H_m
    H_within_sum = 0.0
    # For each module, build distribution over nodes + exit event
    # We'll use flows aggregated per node: internal flows from module m to node i in m, plus exit.
    # To approximate codebook we use normalized flow weights inside module.
    for m in range(K):
        # nodes in module m
        nodes = np.where(labels == m)[0]
        if nodes.size == 0:
            continue
        # compute flows to nodes in module and exit flow
        # internal flow mass = sum_{u in m, v in m} w(u->v)
        internal = module_flow[m, m]
        exit_flow = q_exit[m]
        total_m = internal + exit_flow
        if total_m <= 0:
            continue
        # distribution: for simplicity, assign probabilities proportional to incoming flow to nodes inside module
        # compute per-node incoming flow from nodes in module
        node_probs = []
        for node in nodes:
            # sum weight from module m to this node
            idxs = A[:, node].tocoo()
            # sum only from sources in module m
            s = 0.0
            for u, w in zip(idxs.row, idxs.data):
                if labels[int(u)] == m:
                    s += float(w)
            node_probs.append(s)
        node_probs = np.array(node_probs, dtype=float)
        # include exit as last element
        probs = np.concatenate([node_probs, np.array([exit_flow], dtype=float)])
        probs_sum = probs.sum()
        if probs_sum <= 0:
            continue
        probs = probs / (probs_sum + epsilon)
        # entropy
        Hm = -np.sum([p * math.log2(p + epsilon) for p in probs if p>0])
        # weight Hm by p_module[m] (probability mass in module)
        H_within_sum += p_module[m] * Hm
    # final map equation
    L = q_total * H_exit + H_within_sum
    return float(L)

# ---------------- Greedy agglomerative search (history kept) ----------------
def greedy_agglomerative_map(A: sp.spmatrix, pi: Optional[np.ndarray]=None, max_iter: int=10000) -> Tuple[List[int], List[Dict[str,Any]]]:
    """
    Agglomerative greedy algorithm that merges pairs of modules to reduce L.
    Returns (final_partition, history)
    history is a list of dicts capturing each merge: step, merged_pair, L, gain, num_modules, ...
    NOTE: this function mirrors the notebook's greedy algorithm. It may be computationally heavy.
    """
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]
    if n == 0:
        return [], []
    # start with each node in its own module
    part = np.arange(n, dtype=int)
    L_current = map_equation_L(A, part, pi)
    history = []
    step = 0
    improved = True
    while improved and step < max_iter:
        improved = False
        step += 1
        best_gain = 0.0
        best_merge = None
        best_part = part.copy()
        # naive O(k^2) merge search: for each pair of modules compute L after merging them
        labels = np.unique(part)
        k = labels.size
        if k <= 1:
            break
        # create mapping module -> nodes
        mod_nodes = {m: np.where(part == m)[0] for m in labels}
        # evaluate all merges
        for i_idx in range(len(labels)):
            for j_idx in range(i_idx+1, len(labels)):
                mi = labels[i_idx]; mj = labels[j_idx]
                # propose merging mj into mi (or both into new label)
                part_proposed = part.copy()
                part_proposed[part == mj] = mi
                part_proposed = normalize_partition_labels(part_proposed)
                L_new = map_equation_L(A, part_proposed, pi)
                gain = L_current - L_new
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_merge = (mi, mj)
                    best_part = part_proposed.copy()
        if best_merge is not None:
            improved = True
            part = best_part
            L_old = L_current
            L_current = map_equation_L(A, part, pi)
            history.append({
                'step': step,
                'merged': best_merge,
                'L_before': float(L_old),
                'L': float(L_current),
                'gain': float(L_old - L_current),
                'num_modules': int(len(np.unique(part)))
            })
        else:
            break
    # final normalization
    final_part = normalize_partition_labels(part).tolist()
    return final_part, history

# keep a backward-compatible alias
def greedy_search(A, pi=None, max_iter=10000):
    return greedy_agglomerative_map(A, pi=pi, max_iter=max_iter)[0]

# ---------------- Simulated annealing (refine partition) ----------------
def simulated_annealing_refine(A: sp.spmatrix,
                               partition_init: List[int],
                               pi: Optional[np.ndarray]=None,
                               T0: float=0.05,
                               cooling_rate: float=0.98,
                               steps_per_T: int=1,
                               min_T: float=1e-5,
                               replicates: int=1,
                               rng_seed: Optional[int]=None) -> Tuple[List[int], List[Dict[str,Any]]]:
    """
    Simulated annealing refinement that takes an initial partition (from GS) and proposes single-node moves.
    Returns (best_partition, history) where history is a list of dicts with step-level diagnostics.
    """
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]
    if n == 0:
        return [], []
    if pi is None:
        pi = np.ones(n)/n
    if rng_seed is not None:
        random.seed(rng_seed); np.random.seed(rng_seed)
    current = normalize_partition_labels(partition_init).tolist()
    best = current.copy()
    current_L = map_equation_L(A, current, pi)
    best_L = current_L
    history = []
    T = T0
    step = 0
    while T > min_T:
        for _ in range(steps_per_T):
            step += 1
            node = random.randrange(n)
            cur_label = current[node]
            existing_labels = list(sorted(set(current)))
            candidate_labels = existing_labels + [max(existing_labels) + 1]
            new_label = random.choice(candidate_labels)
            if new_label == cur_label:
                continue
            old_label = current[node]
            current[node] = new_label
            current = normalize_partition_labels(current).tolist()
            L_new = map_equation_L(A, current, pi)
            delta = L_new - current_L
            accept = False
            if delta < 0:
                accept = True
            else:
                if random.random() < math.exp(-delta / (T + 1e-18)):
                    accept = True
            if accept:
                accepted = True
                prev_L = current_L
                current_L = L_new
                if current_L < best_L:
                    best_L = current_L
                    best = current.copy()
            else:
                accepted = False
                # revert
                current[node] = old_label
            history.append({
                'step': step,
                'node': int(node),
                'accepted': bool(accepted),
                'delta': float(delta),
                'current_L': float(current_L),
                'best_L': float(best_L),
                'num_modules': int(len(set(current)))
            })
        T *= cooling_rate
    return best, history

# backward-compatible name
def simulated_annealing(A, partition_init, pi=None, **kwargs):
    return simulated_annealing_refine(A, partition_init, pi=pi, **kwargs)

# ---------------- Plotting functions (the ones you asked to preserve) ----------------

def plot_greedy_history(history, figsize=(10,6), save_path=None):
    if not history:
        print("Empty history: no merges performed.")
        return
    steps = [0] + [h['step'] for h in history]
    Ls = []
    nums = []
    gains = []
    L0 = history[0].get('L_before', None)
    if L0 is None:
        L0 = history[0]['L'] + history[0].get('gain', 0.0)
    Ls.append(L0)
    nums.append(history[0].get('num_modules', None) + 1 if history[0].get('num_modules', None) is not None else None)
    gains.append(0.0)
    for h in history:
        Ls.append(h['L'])
        nums.append(h['num_modules'])
        gains.append(h.get('gain', 0.0))
    fig, axs = plt.subplots(1,3, figsize=figsize)
    axs[0].plot(steps, Ls, marker='o'); axs[0].set_xlabel('merge step'); axs[0].set_ylabel('L (bits/step)'); axs[0].set_title('L vs step'); axs[0].grid(True)
    axs[1].plot(steps, nums, marker='o'); axs[1].set_xlabel('merge step'); axs[1].set_ylabel('#modules'); axs[1].set_title('#Modules vs step'); axs[1].grid(True)
    axs[2].bar(steps, gains); axs[2].set_xlabel('merge step'); axs[2].set_ylabel('gain'); axs[2].set_title('Gain per merge'); axs[2].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_sorted_adjacency(A, partition, vmax=None, figsize=(6,6), max_dense=800, save_path=None):
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]
    part = np.asarray(partition, dtype=int)
    order = np.argsort(part)
    if n <= max_dense:
        Ad = A.toarray()
        Ads = Ad[np.ix_(order, order)]
        fig = plt.figure(figsize=figsize)
        plt.imshow(Ads, aspect='auto', interpolation='nearest', cmap='viridis', vmax=vmax)
        cuts = np.where(np.diff(part[order]) != 0)[0]
        for c in cuts:
            plt.axhline(c + 0.5, color='w', linewidth=0.8)
            plt.axvline(c + 0.5, color='w', linewidth=0.8)
        plt.colorbar(label='edge weight')
        plt.title('Adjacency (nodes sorted by module)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()
    else:
        fig = plt.figure(figsize=figsize)
        Apr = A[order,:][:,order]
        plt.spy(Apr, markersize=1)
        plt.title('Adjacency (sparse spy) - nodes sorted by module')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()

def plot_module_sizes(partition, figsize=(6,3), save_path=None):
    part = np.asarray(partition, dtype=int)
    mods, counts = np.unique(part, return_counts=True)
    fig = plt.figure(figsize=figsize)
    plt.bar(mods.astype(str), counts)
    plt.xlabel('module'); plt.ylabel('size (nodes)'); plt.title('Module sizes')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()

# small helpers for map plotting
def _node_strength(G, weight="weight"):
    s = {n: 0.0 for n in G.nodes}
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight, 1.0))
        s[u] += w; s[v] += w
    return s

def _distinct_colors(k):
    pools = ["tab10","Set3","Set1","Set2","Dark2","Accent","Paired","tab20b","tab20c"]
    colors=[]
    for name in pools:
        cmap = plt.get_cmap(name)
        for i in range(cmap.N):
            colors.append(to_rgba(cmap(i)))
            if len(colors)>=k:
                return colors[:k]
    return [plt.cm.hsv(i/max(1,k)) for i in range(k)]

def plot_network_by_module(state_name_or_code,
                           G=None,
                           modules=None,
                           graphml_path=None,
                           with_labels=False,
                           figsize=(12,10),
                           dpi=300,
                           save_path=None,
                           node_scale=1.0,
                           verbose=False):
    """
    Geographic plot. Expects nodes to have 'lon' and 'lat' attributes in graphml.
    modules: list/array mapping node->module (in node order of G.nodes()) if provided.
    """
    if G is None and graphml_path is None:
        raise ValueError("Provide G or graphml_path.")
    if G is None:
        G = nx.read_graphml(graphml_path)
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes.")
    # nodes must have lon/lat
    n0 = next(iter(G.nodes()))
    if "lon" not in G.nodes[n0] or "lat" not in G.nodes[n0]:
        # fallback to spring layout if coords missing (but notebook expects coords)
        pos = nx.spring_layout(G)
    else:
        pos = {n: (float(G.nodes[n]["lon"]), float(G.nodes[n]["lat"])) for n in G.nodes}
    # modules: if not provided, fetch from node attributes 'module'
    if modules is None:
        modules = [int(G.nodes[n].get("module", -1)) for n in G.nodes]
    uniq_modules = sorted(set(modules))
    colors_pool = _distinct_colors(len(uniq_modules))
    mod_to_color = {m: colors_pool[i] for i,m in enumerate(uniq_modules)}
    node_colors = [mod_to_color[m] for m in modules]
    s_dict = _node_strength(G, weight="weight")
    s_vals = np.array([s_dict[n] for n in G.nodes], dtype=float)
    if np.all(s_vals == 0):
        s_vals = np.ones_like(s_vals)
    s_vals = s_vals / (s_vals.max() + 1e-12)
    node_sizes = node_scale * 300 * (0.25 + 1.75 * np.sqrt(s_vals))
    # edge widths
    if G.number_of_edges() > 0:
        w_raw = np.array([float(d.get("weight",1.0)) for _,_,d in G.edges(data=True)], dtype=float)
        w_max = float(np.max(w_raw)) if w_raw.size else 1.0
        edge_widths = (5.0 / (w_max + 1e-12)) * w_raw
    else:
        edge_widths = []
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # basemap if available
    if pygris is not None:
        try:
            counties = pygris.counties(state=state_name_or_code, cb=True, year=2018, cache=True).to_crs(4326)
            if str(state_name_or_code).lower() in ("alaska","ak","02"):
                counties = counties[counties["GEOID"] != "02016"]
            counties.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.3, alpha=1.0)
        except Exception as e:
            if verbose:
                warnings.warn(f"pygris failed: {e}")
    # draw edges (best effort using shapely/gpd)
    try:
        lines, widths = [], []
        for (u,v), lw in zip(G.edges(), edge_widths):
            if u in pos and v in pos:
                lines.append(LineString([pos[u], pos[v]])); widths.append(lw)
        if lines:
            edges_gdf = gpd.GeoDataFrame({"linewidth": widths}, geometry=lines, crs="EPSG:4326")
            edges_gdf.plot(ax=ax, linewidth=edges_gdf["linewidth"], alpha=1, color="cornflowerblue")
        else:
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="cornflowerblue", alpha=0.6, ax=ax)
    except Exception:
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="cornflowerblue", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, linewidths=0, ax=ax)
    if with_labels:
        labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    ax.set_title(f"SCI Network {state_name_or_code} (modules)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        if verbose:
            plt.show()
        else:
            plt.close(fig)

def _default_network_plot(A, partition, pi=None, node_list=None):
    # small fallback used in SA plotting
    if issparse(A):
        A = A.tocsr()
    G = nx.from_numpy_array(A.toarray() if issparse(A) else np.asarray(A))
    colors = ['C{}'.format(c % 10) for c in partition]
    plt.figure(figsize=(6,6)); nx.draw(G, node_color=colors, with_labels=False, node_size=30); plt.show()

def plot_sa_results(A, best_partition, sa_history, pi=None, start_partition=None, node_list=None,
                    use_existing_network_plot=True, rolling_window=None, topk=30, save_prefix=None):
    # Implementation reproduced from notebook; will save plots when save_prefix is provided
    if A is None or best_partition is None or sa_history is None:
        raise ValueError("A, best_partition and sa_history are required.")
    best_part = np.asarray(best_partition, dtype=int)
    history = list(sa_history)
    # 1) network final
    if use_existing_network_plot and save_prefix is not None:
        try:
            # try to use plot_network_by_module if we have graphml or G; fallback to default
            pass
        except Exception:
            _default_network_plot(A, best_part, pi=pi, node_list=node_list)
    else:
        _default_network_plot(A, best_part, pi=pi, node_list=node_list)
    # 2) L traces
    steps = [h.get('step', i) for i,h in enumerate(history)]
    currentLs = [h.get('current_L', np.nan) for h in history]
    bestLs = [h.get('best_L', np.nan) for h in history]
    accepted = [1 if h.get('accepted', False) else 0 for h in history]
    deltas = [h.get('delta', 0.0) for h in history]
    # plot traces
    fig = plt.figure(figsize=(10,4))
    plt.plot(steps, currentLs, label='current L', alpha=0.7); plt.plot(steps, bestLs, label='best L', alpha=0.9)
    plt.xlabel('proposal step'); plt.ylabel('L (bits/step)'); plt.title('SA: current and best L'); plt.legend(); plt.grid(True)
    if save_prefix:
        plt.savefig(f"{save_prefix}_Ltrace.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()
    # acceptance rolling
    if rolling_window is None:
        rolling_window = max(1, len(accepted)//20)
    if len(accepted) >= 5 and rolling_window > 0:
        acc = np.array(accepted, dtype=float)
        if len(acc) > rolling_window:
            acc_roll = np.convolve(acc, np.ones(rolling_window)/rolling_window, mode='valid')
            fig = plt.figure(figsize=(8,2)); plt.plot(range(len(acc_roll)), acc_roll); plt.title('Acceptance rate'); plt.grid(True)
            if save_prefix:
                plt.savefig(f"{save_prefix}_acceptance.png", dpi=200, bbox_inches='tight'); plt.close(fig)
            else:
                plt.show()
    # delta histogram
    fig = plt.figure(figsize=(8,3))
    _accepted_deltas = [d for d,a in zip(deltas, accepted) if a==1]
    _rejected_deltas = [d for d,a in zip(deltas, accepted) if a==0]
    if len(_rejected_deltas) > 0:
        plt.hist(_rejected_deltas, bins=40, alpha=0.6, label='rejected')
    if len(_accepted_deltas) > 0:
        plt.hist(_accepted_deltas, bins=40, alpha=0.8, label='accepted')
    plt.legend(); plt.title('ΔL histogram'); plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_delta_hist.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()
    # module sizes final
    final_mods, final_counts = np.unique(best_part, return_counts=True)
    fig = plt.figure(figsize=(6,3)); plt.bar([str(m) for m in final_mods], final_counts); plt.title('Module sizes after SA'); plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_modulesizes.png", dpi=200, bbox_inches='tight'); plt.close(fig)
    else:
        plt.show()
    # adjacency sorted
    n = A.shape[0]
    order = np.argsort(best_part)
    if n <= 800:
        Ad = A.toarray() if issparse(A) else np.asarray(A)
        Ads = Ad[np.ix_(order, order)]
        fig = plt.figure(figsize=(6,6)); plt.imshow(Ads, aspect='auto', interpolation='nearest', cmap='viridis')
        cuts = np.where(np.diff(best_part[order]) != 0)[0]
        for c in cuts:
            plt.axhline(c+0.5, color='w', linewidth=0.8); plt.axvline(c+0.5, color='w', linewidth=0.8)
        plt.colorbar(label='edge weight'); plt.title('Adjacency sorted by module'); plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_adj_sorted.png", dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()
    else:
        Apr = (A if not issparse(A) else A.tocsr())[order,:][:,order]
        fig = plt.figure(figsize=(6,6)); plt.spy(Apr, markersize=1); plt.title('Adjacency spy'); plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_adj_spy.png", dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()
    # top nodes by pi
    if pi is not None:
        pi_arr = np.asarray(pi); order_pi = np.argsort(pi_arr)[::-1]; k = min(topk, len(pi_arr))
        top_idx = order_pi[:k]; top_vals = pi_arr[top_idx]; top_mods = best_part[top_idx]
        cmap = plt.get_cmap('tab20'); colors = [cmap(int(m) % 20) for m in top_mods]
        fig = plt.figure(figsize=(max(8, k*0.25), 3)); plt.bar(range(k), top_vals, color=colors)
        labels = [(node_list[i] if node_list is not None else str(i)) for i in top_idx]
        plt.xticks(range(k), labels, rotation=45, ha='right'); plt.ylabel('π'); plt.title('Top nodes by π'); plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_top_pi.png", dpi=200, bbox_inches='tight'); plt.close(fig)
        else:
            plt.show()
    return {
        'final_module_counts': (final_mods.tolist(), final_counts.tolist()),
        'history_summary': {
            'steps': steps,
            'currentLs': currentLs,
            'bestLs': bestLs,
            'accepted': accepted,
            'deltas': deltas
        }
    }
