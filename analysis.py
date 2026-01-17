import numpy as np
import random
from scipy import sparse
from scipy.sparse import csr_matrix, issparse

import networkx as nx

import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Union

from itertools import combinations
import os

from shapely import LineString

from joblib import Parallel, delayed
_USE_JOBLIB = True

init_seed = 739


def build_adj_from_edge_list(n: int, edges, directed=True):
    """
    Build adjacency CSR using networkx. edges = list of (u,v,weight).
    Returns csr_matrix A with A[i,j] = weight of i -> j.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # add nodes (ensures nodes 0..n-1 exist)
    G.add_nodes_from(range(n))
    # add weighted edges
    # networkx expects weight argument name 'weight'
    G.add_weighted_edges_from(edges)

    # extract sparse adjacency in row-major (rows = source nodes)
    # nx.to_scipy_sparse_array (or nx.to_scipy_sparse_matrix in older nx) yields a matrix
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight='weight', format='csr')
    return A

def stochastic_block_model_directed_weighted(sizes,
                                                p_in: float,
                                                p_out: float,
                                                weight_in_scale: float = 1.0,
                                                weight_out_scale: float = 0.2,
                                                self_loops: bool = False,
                                                seed: int = None):
    """
    Build directed stochastic-block weighted adjacency using networkx.
    Returns (A_csr, blocks_array) where blocks_array[u] = block index of node u.
    """
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    # build probability matrix (block x block)
    k = len(sizes)
    p_matrix = [[(p_in if i == j else p_out) for j in range(k)] for i in range(k)]

    # nx.stochastic_block_model allows directed=True and selfloops flag
    G = nx.stochastic_block_model(sizes, p_matrix, directed=True, selfloops=self_loops, seed=seed)
    # nodes are labeled 0..n-1 in order, blocks can be reconstructed from sizes
    blocks = np.repeat(np.arange(k), sizes)

    # assign weights: for each edge (u,v) sample weight depending on same_block
    # use exponential as before
    for u, v in list(G.edges()):
        same = (blocks[u] == blocks[v])
        scale = weight_in_scale if same else weight_out_scale
        G[u][v]['weight'] = float(rng.exponential(scale=scale))

    # create CSR adjacency (A[i,j] = weight i->j)
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight='weight', format='csr')

    return A, blocks

def parse_partition(partition: Union[List[int], Dict[int, List[int]], List[List[int]]],
                    n: int) -> np.ndarray:
    """
    Accepts several partition forms and returns an array of length n mapping node -> module index (0..m-1):
      - array-like of length n with module indices
      - dict {module_label: [nodes,...]}
      - list of lists: [[nodes_of_module0], [nodes_of_module1], ...]
    """
    if partition is None:
        raise ValueError("partition cannot be None")

    # Case 1: array-like
    try:
        arr = np.asarray(partition)
        if arr.ndim == 1 and arr.size == n and np.issubdtype(arr.dtype, np.integer):
            return arr.astype(int)
    except Exception:
        pass

    # Case 2: dict or list-of-lists
    result = np.full(n, -1, dtype=int)
    if isinstance(partition, dict):
        items = partition.items()
    elif isinstance(partition, (list, tuple)):
        # assume list-of-lists: each item is a list of node ids (or integers)
        items = enumerate(partition)
    else:
        raise ValueError("Unsupported partition format")

    idx = 0
    for key, nodes in items:
        for u in nodes:
            result[int(u)] = idx
        idx += 1

    if (result < 0).any():
        missing = np.where(result < 0)[0]
        raise ValueError(f"Partition does not assign these nodes: {missing.tolist()}")
    return result

def row_stochastic_from_adj(A):
    """
    Convert adjacency matrix A (n x n) with A[i,j] = weight i -> j
    into a row-stochastic transition matrix T (csr_matrix).
    Dangling rows (rows with sum 0) remain zero-rows.
    Works for sparse or dense A.
    """
    if sparse.issparse(A):
        A = A.tocsr()
        row_sums = np.array(A.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sums, dtype=float)
        nz = row_sums > 0
        inv[nz] = 1.0 / row_sums[nz]
        Dinv = sparse.diags(inv)
        T = Dinv.dot(A)
    else:
        A = np.asarray(A, dtype=float)
        row_sums = A.sum(axis=1)
        T = np.zeros_like(A)
        nz = row_sums > 0
        T[nz] = (A[nz].T / row_sums[nz]).T
        T = csr_matrix(T)
    return T

def power_method_stationary(A, epsilon=0.15, tol=1e-12, maxiter=10000, verbose=False):
    """
    Compute stationary distribution pi as a numpy array using networkx.pagerank.
    - A: adjacency (scipy csr or networkx Graph/DiGraph). If scipy matrix is given, convert to DiGraph.
    - epsilon: teleportation probability (teleport prob). nx.pagerank uses damping alpha = 1 - epsilon.
    Returns pi (1d numpy array length n).
    """
    # If input is sparse adjacency, convert to networkx DiGraph
    if issparse(A):
        # convert sparse adjacency to DiGraph preserving weights
        # nx.from_scipy_sparse_array returns Graph (undirected) for symmetric; for adjacency it supports create_using
        G = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.DiGraph)
    elif isinstance(A, (nx.Graph, nx.DiGraph)):
        G = A
    else:
        # assume dense numpy array
        G = nx.from_numpy_array(np.asarray(A), create_using=nx.DiGraph)

    alpha = 1.0 - epsilon  # damping factor used by networkx pagerank
    # Use nx.pagerank (power iteration) — it respects 'weight' edge attribute
    pr = nx.pagerank(G, alpha=alpha, max_iter=maxiter, tol=tol, weight='weight')
    # pr is dict node->score; ensure ordering 0..n-1
    n = G.number_of_nodes()
    pi = np.array([pr[i] for i in range(n)], dtype=float)
    # normalize to be safe
    pi = pi / pi.sum()
    return pi


def entropy(probs, base=2.0):
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -np.sum(probs * np.log(probs) / np.log(base))

def map_equation_L(A,
                   partition,
                   pi: np.ndarray = None,
                   epsilon: float = 0.15,
                   return_components: bool = False):
    """
    Compute map equation L for adjacency A (weighted,directed) and a flat partition.

    partition: can be array-like (node->module), dict or list-of-lists (see parse_partition).
    pi: optional stationary distribution (if None, computed here).
    """
    if sparse.issparse(A):
        n = A.shape[0]
    else:
        A = np.asarray(A, dtype=float)
        n = A.shape[0]

    part_arr = parse_partition(partition, n)
    modules, inverse = np.unique(part_arr, return_inverse=True)
    m = modules.size

    T = row_stochastic_from_adj(A)
    if pi is None:
        pi = power_method_stationary(A, epsilon=0.15)

    # indicator matrix for modules (n x m)
    rows = np.arange(n)
    cols = inverse
    data = np.ones(n, dtype=float)
    indicator = sparse.csr_matrix((data, (rows, cols)), shape=(n, m))

    # T.dot(indicator) gives for each node a and module j: sum_{b in module j} T[a,b]
    T_to_modules = T.dot(indicator).toarray()  # shape (n,m)

    module_sizes = np.bincount(inverse, minlength=m)

    # P_to_modules (including teleportation)
    uniform_module_mass = module_sizes.astype(float) / float(n)
    P_to_modules = (1.0 - epsilon) * T_to_modules + epsilon * uniform_module_mass[np.newaxis, :]

    # q_i: exit probability per step from module i
    q_i = np.zeros(m, dtype=float)
    for a in range(n):
        mod_a = inverse[a]
        prob_leave_from_a = 1.0 - P_to_modules[a, mod_a]
        q_i[mod_a] += pi[a] * prob_leave_from_a

    q_total = q_i.sum()
    sum_pi_in_module = np.zeros(m, dtype=float)
    p_i = np.zeros(m, dtype=float)
    for j in range(m):
        mask = (inverse == j)
        sum_pi_in_module[j] = pi[mask].sum()
        p_i[j] = q_i[j] + sum_pi_in_module[j]

    H_Q = 0.0
    if q_total > 0:
        H_Q = entropy(q_i / q_total)

    H_Pi = np.zeros(m, dtype=float)
    for j in range(m):
        if p_i[j] <= 0:
            H_Pi[j] = 0.0
            continue
        probs = []
        if q_i[j] > 0:
            probs.append(q_i[j] / p_i[j])
        node_mask = (inverse == j)
        if node_mask.any():
            node_probs = pi[node_mask] / p_i[j]
            probs.extend(node_probs.tolist())
        probs = np.asarray(probs, dtype=float)
        H_Pi[j] = entropy(probs)

    L = q_total * H_Q + np.sum(p_i * H_Pi)

    if return_components:
        return {
            'L': L,
            'q_i': q_i,
            'q_total': q_total,
            'p_i': p_i,
            'sum_pi_in_module': sum_pi_in_module,
            'H_Q': H_Q,
            'H_Pi': H_Pi,
            'partition_mapping': part_arr
        }
    else:
        return L
    

def map_equation_two_level(A, top_partition, subpartitions, epsilon=0.15, return_components=False):
    """
    Evaluate map equation for a two-level partition:
      - top_partition: mapping node->top-module (array-like/dict/list)
      - subpartitions: dict mapping top_module_index -> partition-within-that-module
          each subpartition can be a list-of-lists of node indices (local to global node ids),
          or a dict or array-like mapping nodes->submodule ids (global indices).
    This constructs a flat partition that assigns a unique id for each submodule across top modules,
    then computes L using map_equation_L.
    Note: This is a convenience helper — the full hierarchical map equation (recursive codebooks)
          is more elaborate; here we simply evaluate the two-level partition as a flat partition
          of submodules (which is a common way to compare nested partitions).
    """
    if sparse.issparse(A):
        n = A.shape[0]
    else:
        n = np.asarray(A).shape[0]

    top_arr = parse_partition(top_partition, n)

    # Build global submodule assignment array:
    global_sub_assign = np.full(n, -1, dtype=int)
    next_id = 0
    # subpartitions should provide the internal partitioning for nodes in each top module
    for top_mod in np.unique(top_arr):
        if top_mod not in subpartitions:
            # treat whole top module as a single submodule
            nodes = np.where(top_arr == top_mod)[0].tolist()
            sub_list = [nodes]
        else:
            sub_list = subpartitions[top_mod]
            # allow dict or list-of-lists
            if isinstance(sub_list, dict):
                sub_list = list(sub_list.values())
        # assign ids
        for sub in sub_list:
            for u in sub:
                if top_arr[u] != top_mod:
                    raise ValueError(f"Node {u} assigned to submodule for top module {top_mod} but top partition says {top_arr[u]}")
                global_sub_assign[int(u)] = next_id
            next_id += 1

    if (global_sub_assign < 0).any():
        missing = np.where(global_sub_assign < 0)[0]
        raise ValueError(f"Some nodes were not assigned to any submodule: {missing.tolist()}")

    # Now compute map_equation on the flat partition given by global_sub_assign
    return map_equation_L(A, global_sub_assign, pi=None, epsilon=0.15, return_components=return_components)


def normalize_partition_labels(part_arr):
    """
    Map arbitrary integer labels to a compact 0..m-1 range.
    Input: array-like length n with integer module labels (may be non-consecutive).
    Returns: numpy array length n with labels remapped to 0..m-1.
    """
    arr = np.asarray(part_arr, dtype=int)
    unique = np.unique(arr)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[x] for x in arr], dtype=int)

def ensure_full_partition(partition_candidate, n):
    """
    Given a partition in one of accepted forms (array, dict, list-of-lists),
    guarantee it assigns every node in 0..n-1. If nodes are missing,
    assign missing nodes to existing modules (or create new module 0 if none exist).
    Returns a normalized array mapping node->module with labels 0..m-1.
    """
    # quick attempt to parse using your parse_partition if already valid
    try:
        arr = parse_partition(partition_candidate, n)
        # normalize labels to 0..m-1
        return normalize_partition_labels(arr)
    except Exception as e:
        # parse_partition failed → try to be forgiving and build from dict/list/array manually
        # If it's dict/list-of-lists build a dict mapping module->list(nodes)
        part_dict = {}
        if isinstance(partition_candidate, dict):
            part_dict = {int(k): [int(u) for u in v] for k,v in partition_candidate.items()}
        elif isinstance(partition_candidate, (list, tuple, np.ndarray)):
            # If it's a 1D array-like of length n with ints, use it.
            arr = np.asarray(partition_candidate)
            if arr.ndim == 1 and arr.size == n:
                return normalize_partition_labels(arr.astype(int))
            # else assume list-of-lists
            else:
                for i,sub in enumerate(partition_candidate):
                    part_dict[i] = [int(u) for u in sub]
        else:
            raise ValueError("Unsupported partition_candidate type in ensure_full_partition")

        # Build a node->module mapping and detect missing nodes
        node_to_mod = {}
        for mod, nodes in part_dict.items():
            for u in nodes:
                node_to_mod[int(u)] = int(mod)

        # Fill missing nodes: assign them to the module with largest size (or module 0 if none)
        missing = [u for u in range(n) if u not in node_to_mod]
        if len(part_dict) == 0:
            # No modules provided: put everyone in module 0
            return np.zeros(n, dtype=int)
        # choose target module for missing nodes: largest existing module
        sizes = {mod: len(nodes) for mod,nodes in part_dict.items()}
        if len(sizes) > 0:
            target_mod = max(sizes.items(), key=lambda x: x[1])[0]
        else:
            target_mod = 0
        for u in missing:
            node_to_mod[u] = int(target_mod)

        # produce array of node->module
        arr = np.array([node_to_mod[u] for u in range(n)], dtype=int)
        return normalize_partition_labels(arr)

def graphnx_to_csr_and_mapping(G: nx.Graph):
    """
    Convert a networkx Graph/DiGraph G to adjacency csr A and return:
      - A: scipy.sparse.csr_matrix (A[i,j] = weight i->j)
      - node_list: list mapping indices 0..n-1 -> original node labels
      - node_to_idx: dict mapping original label -> index
    Node order is deterministic: sorted(G.nodes()) if labels are comparable, otherwise list(G.nodes()).
    """
    # choose deterministic node ordering
    try:
        node_list = sorted(G.nodes())
    except Exception:
        node_list = list(G.nodes())
    node_to_idx = {node_list[i]: i for i in range(len(node_list))}
    # get adjacency in csr with weight attribute
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, weight='weight', format='csr')
    return A, node_list, node_to_idx

def load_graphnx_from_edgelist(path, delimiter=None, directed=True, weighted=True, nodetype=None):
    """
    Load an edge list to a networkx DiGraph or Graph.
    If weighted, tries to read a third column as weight (float).
    """
    if weighted:
        G = nx.read_weighted_edgelist(path, delimiter=delimiter, create_using=nx.DiGraph() if directed else nx.Graph(), nodetype=nodetype)
    else:
        G = nx.read_edgelist(path, delimiter=delimiter, create_using=nx.DiGraph() if directed else nx.Graph(), nodetype=nodetype)
    # ensure 'weight' attribute exists for every edge (default 1.0)
    for u,v,data in G.edges(data=True):
        if 'weight' not in data:
            G[u][v]['weight'] = 1.0
    return G

def load_graphnx_from_graphml(path):
    """Load a graph from GraphML. Networkx preserves node labels and weights if present."""
    G = nx.read_graphml(path)
    # ensure DiGraph if directed
    if not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        # keep same structure but make a DiGraph with same edges
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes(data=True))
        for u,v,data in G.edges(data=True):
            H.add_edge(u, v, **data)
        G = H
    # ensure weight attribute on edges
    for u,v,data in G.edges(data=True):
        if 'weight' not in data:
            G[u][v]['weight'] = 1.0
    return G

def csr_and_mapping_to_external_partition(partition_map, node_to_idx):
    """
    Convert an external partition mapping keyed by the original node labels (e.g. dict: label -> module)
    into a node-indexed partition array of length n (0..n-1), compatible with parse_partition.
    """
    # node_to_idx is a dict original_label -> index
    n = len(node_to_idx)
    arr = np.full(n, -1, dtype=int)
    for label, mod in partition_map.items():
        if label not in node_to_idx:
            raise KeyError(f"Label {label} not found in graph nodes")
        arr[node_to_idx[label]] = int(mod)
    if (arr < 0).any():
        missing = np.where(arr < 0)[0]
        raise ValueError(f"Not all nodes assigned in partition; missing indices: {missing.tolist()}")
    return arr

# -------------------------
# Plotting helpers
# -------------------------
def plot_network(A,
                 pi=None,
                 partition=None,
                 node_list=None,
                 figsize=(9,6),
                 layout_seed=init_seed,
                 max_label_chars=8,
                 edge_width_scale=5.0,
                 node_size_scale=3000):
    """
    Visualize the directed weighted network A with nodes colored by module.
    - If `partition` is None, will try to use global `ground_truth` or `partition_arr` if present.
    - If `partition` is given as a dict keyed by original labels, convert it using node_list -> index map.
    - A: scipy csr or numpy array, A[i,j] = weight i->j
    """
    # Ensure A is CSR for later operations
    if not issparse(A):
        A = csr_matrix(A)
    n = A.shape[0]


    # if partition is a dict keyed by original node labels and node_list is provided,
    # convert it to an index array
    if isinstance(partition, dict):
        if node_list is None:
            # try to use node_list from global mapping if available
            if 'node_list' in globals() and 'node_to_idx' in globals():
                nl = globals()['node_list']
                nt = globals()['node_to_idx']
                part_arr = csr_and_mapping_to_external_partition(partition, nt)
            else:
                raise ValueError("partition is a dict of external labels but node_list/node_to_idx not provided")
        else:
            # build node_to_idx from provided node_list
            node_to_idx = {node_list[i]: i for i in range(len(node_list))}
            part_arr = csr_and_mapping_to_external_partition(partition, node_to_idx)
    else:
        # assume array-like mapping node-index -> module
        part_arr = np.asarray(partition, dtype=int)
        if part_arr.size != n:
            raise ValueError(f"partition length {part_arr.size} != number of nodes {n}")

    # Build networkx DiGraph for plotting
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)

    # Node labels for display
    if node_list is None:
        node_list_plot = list(range(n))
    else:
        node_list_plot = node_list

    # layout from undirected version for stability
    pos = nx.spring_layout(G.to_undirected(), seed=layout_seed)

    # node sizes: based on pi if provided
    if pi is None:
        sizes = np.ones(n) * (node_size_scale / max(n,1))
    else:
        pi = np.asarray(pi, dtype=float)
        sizes = np.clip(pi / (pi.max() + 1e-16), 0.02, 1.0) * node_size_scale

    # node colors: by partition if provided, otherwise single color
    if part_arr is not None:
        # normalize labels to 0..m-1 for consistent colormap indexing
        unique_mods, inv = np.unique(part_arr, return_inverse=True)
        cmap = plt.get_cmap('tab20')
        node_color = [cmap(int(inv[i]) % 20) for i in range(n)]
    else:
        node_color = 'tab:blue'

    # edge widths proportional to weight
    weights = []
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        weights.append(w)
    if len(weights) > 0:
        weights = np.array(weights)
        if weights.max() > 0:
            ew = 0.2 + (weights - weights.min()) / (weights.max() - weights.min() + 1e-16) * (edge_width_scale - 0.2)
        else:
            ew = np.ones_like(weights) * 0.5
    else:
        ew = []

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=sizes,
                           node_color=node_color,
                           alpha=0.95)
    nx.draw_networkx_edges(G, pos, width=ew, arrowsize=10, alpha=0.6)

    # labels for a limited number of nodes
    show_labels = {i: (str(node_list_plot[i])[:max_label_chars]) for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=show_labels, font_size=8)

    plt.axis('off')
    plt.title("Network visualization (node size ~ stationary prob, node color ~ SBM module)")
    plt.show()


def plot_stationary(pi, node_list=None, sort_desc=True, top_k=None, figsize=(8,4)):
    """
    Plot the stationary distribution pi.
    - If node_list provided, x-axis labels use them (truncated).
    - top_k: if provided, only plot the top_k nodes by pi.
    """
    pi = np.asarray(pi, dtype=float)
    n = len(pi)
    indices = np.arange(n)
    if sort_desc:
        order = np.argsort(pi)[::-1]
    else:
        order = np.arange(n)
    if top_k is not None and top_k < n:
        order = order[:top_k]
    values = pi[order]
    labels = [str(node_list[i]) if node_list is not None else str(i) for i in order]
    labels = [lbl[:12] for lbl in labels]

    plt.figure(figsize=figsize)
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    plt.ylabel('Stationary probability (π)')
    plt.title('Stationary distribution (visit frequencies)')
    plt.tight_layout()
    plt.show()


def plot_module_stats(A, partition, pi=None, epsilon=0.15, figsize=(10,5), return_components=False):
    """
    Plot per-module statistics: q_i, p_i, H_Pi, per-module contribution to L.
    Uses map_equation_L(..., return_components=True).
    """
    comps = map_equation_L(A, partition, pi=pi, epsilon=0.15, return_components=True)
    q_i = comps['q_i']
    p_i = comps['p_i']
    H_Pi = comps['H_Pi']
    q_total = comps['q_total']
    m = len(q_i)

    # Per-module contribution to L:
    # global codebook term contribution per module:
    #   q_i * (-log2(q_i / q_total))  (if q_i>0)
    # local codebook term contribution per module:
    #   p_i * H_Pi
    global_term = np.zeros(m)
    for j in range(m):
        if q_i[j] > 0:
            global_term[j] = q_i[j] * (-np.log2(q_i[j] / (q_total + 1e-16)))
        else:
            global_term[j] = 0.0
    local_term = p_i * H_Pi
    L_i = global_term + local_term

    # Plot stacked bars: global_term and local_term
    ind = np.arange(m)
    plt.figure(figsize=figsize)
    plt.bar(ind, global_term, label='global codebook part (q_i * -log2(q_i/q_total))')
    plt.bar(ind, local_term, bottom=global_term, label='local codebook part (p_i * H_Pi)')
    plt.xticks(ind, [f'mod {i}' for i in range(m)])
    plt.ylabel('Contribution to L (bits per step)')
    plt.title('Per-module contributions to Map equation L')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Also plot q_i and p_i
    plt.figure(figsize=(8,4))
    width = 0.35
    plt.bar(ind - width/2, q_i, width=width, label='q_i (exit prob)')
    plt.bar(ind + width/2, p_i, width=width, label='p_i (codebook mass)')
    plt.xticks(ind, [f'mod {i}' for i in range(m)])
    plt.ylabel('Probability')
    plt.title('Module-level probabilities q_i and p_i')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if return_components:
        return comps
    return None


def plot_edge_weight_distribution(A, bins=40, figsize=(6,3)):
    """
    Plot histogram of non-zero edge weights.
    """
    if issparse(A):
        data = A.data
    else:
        data = np.asarray(A).ravel()
        data = data[data != 0]
    if data.size == 0:
        print("No edges to plot.")
        return
    plt.figure(figsize=figsize)
    # use log scale on x if weights vary hugely (but don't force)
    try:
        plt.hist(data, bins=bins)
    except Exception:
        plt.hist(data[data>0], bins=bins)
    plt.xlabel('Edge weight')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.title('Edge weight distribution (non-zero edges)')
    plt.tight_layout()
    plt.show()


# G = nx.read_graphml(f'state_networks_graphml/{STATENAME}.graphml')

# print(len(nx.nodes(G)))

# A = nx.adjacency_matrix(G)


def merge_labels_array(part_arr, a, b):
    """
    Return a new partition array where modules with label b are remapped to label a,
    and labels are then renormalized to 0..m-1 (compact).
    Accepts integer labels in part_arr.
    """
    arr = np.asarray(part_arr, dtype=int).copy()
    if a == b:
        return normalize_partition_labels(arr)
    # map label b -> a
    arr[arr == b] = a
    # If some labels > b, they remain same; now renormalize
    return normalize_partition_labels(arr)

def compute_all_pair_merges_L(A, current_part, pi, epsilon=0.15):
    """
    For a given partition (array), compute L for every possible unordered pair of distinct modules
    produced by merging them. Returns:
      - best_L_for_pair: dict keyed by (i,j) tuple (i<j) -> L_after_merge
    Note: this re-evaluates map_equation_L for each candidate merge (simple but expensive).
    """
    modules = np.unique(current_part)
    m = modules.size
    results = {}
    # iterate unordered pairs
    for i_idx in range(m):
        for j_idx in range(i_idx+1, m):
            i = modules[i_idx]
            j = modules[j_idx]
            new_part = merge_labels_array(current_part, i, j)
            L_new = map_equation_L(A, new_part, pi=pi, epsilon=0.15)
            results[(i,j)] = L_new
    return results

def greedy_agglomerative_map(A,
                             pi=None,
                             epsilon=0.15,
                             min_merge_gain=1e-12,
                             max_iters=None,
                             verbose=True):
    """
    Greedy agglomerative merging to minimize map-equation L.
    Parameters
    - A: adjacency (csr or ndarray)
    - pi: optional stationary distribution; if None, computed with power_method_stationary
    - epsilon: teleport probability for map_equation_L / power_method
    - min_merge_gain: require (L_before - L_after) > min_merge_gain to accept a merge
    - max_iters: maximum number of merges to perform (None -> up to m-1)
    Returns
    - final_partition_greedy: array length n mapping node->module
    - history: list of dicts for each step with keys:
        {'step', 'num_modules', 'L', 'merged_pair', 'L_before', 'L_after', 'gain', 'partition'}
      partition in history is a copy of the partition array after the merge
    """
    # ensure A shape and n
    if hasattr(A, "shape"):
        n = A.shape[0]
    else:
        A = np.asarray(A)
        n = A.shape[0]

    if pi is None:
        if verbose:
            print("Computing stationary distribution pi...")
        pi = power_method_stationary(A, epsilon=0.15)

    # start each node in its own module
    part = np.arange(n, dtype=int)
    part = normalize_partition_labels(part)  # would be same but keep stable ordering
    current_L = map_equation_L(A, part, pi=pi, epsilon=0.15)
    history = []
    step = 0
    if verbose:
        print(f"Initial L = {current_L:.6f}, n_modules = {np.unique(part).size}")

    # limit iterations to at most n-1 merges
    max_possible = n - 1
    if max_iters is None:
        max_iters = max_possible
    else:
        max_iters = min(max_iters, max_possible)

    # main loop
    while True:
        modules = np.unique(part)
        m = modules.size
        if verbose:
            print(f"\nStep {step}: trying merges among {m} modules")

        # stop condition: can't merge if only one module remains
        if m <= 1 or step >= max_iters:
            if verbose:
                print("Stopping: only one module left or reached max_iters.")
            break

        # compute L for all pairwise merges
        pair_Ls = compute_all_pair_merges_L(A, part, pi, epsilon=0.15)

        # find best (lowest L)
        best_pair, best_L = min(pair_Ls.items(), key=lambda kv: kv[1])
        best_gain = current_L - best_L

        if verbose:
            print(f"Best candidate merge {best_pair} -> L_after={best_L:.6f} (gain {best_gain:.6e})")

        # accept best merge only if it improves L by at least min_merge_gain
        if best_gain <= min_merge_gain:
            if verbose:
                print("No merge improves L by more than min_merge_gain. Stopping.")
            break

        # apply merge
        a, b = best_pair
        new_part = merge_labels_array(part, a, b)
        step += 1
        part = new_part
        L_before = current_L
        current_L = best_L

        history.append({
            'step': step,
            'num_modules': np.unique(part).size,
            'L': current_L,
            'merged_pair': (a,b),
            'L_before': L_before,
            'L_after': current_L,
            'gain': L_before - current_L,
            'partition': part.copy()
        })

        if verbose:
            print(f"Applied merge {a}<-{b}. New L = {current_L:.6f}, modules = {np.unique(part).size}")

    final_part = normalize_partition_labels(part)
    return final_part, history


def module_adjacency_pairs(A, part_arr):
    """
    Return sorted list of unordered module pairs (i,j) with at least one edge between
    module i and module j (either direction). This dramatically reduces candidate merges.
    """
    if hasattr(A, "tocoo"):
        Acoo = A.tocoo()
        rows = Acoo.row
        cols = Acoo.col
    else:
        Ai = np.asarray(A)
        rows, cols = np.nonzero(Ai)

    inv = np.asarray(part_arr, dtype=int)
    pairs = set()
    for u, v in zip(rows, cols):
        mu = int(inv[u]); mv = int(inv[v])
        if mu != mv:
            a, b = (mu, mv) if mu < mv else (mv, mu)
            pairs.add((a, b))
    return sorted(list(pairs))


def _evaluate_merge_pair_worker(A, part_arr, pair, pi, epsilon):
    """Worker that builds merged partition and returns L_after for pair (a,b)."""
    a, b = pair
    new_part = merge_labels_array(part_arr, a, b)
    Lnew = map_equation_L(A, new_part, pi=pi, epsilon=0.15)
    return (pair, float(Lnew))



def compute_candidate_pair_Ls_parallel(A, current_part, pi=None, epsilon=0.15, n_jobs=None):
    """
    Compute L for each candidate merge pair (neighbor-only) in parallel.
    Returns dict {(a,b): L_after}.
    """
    # choose neighbor-only candidate pairs to reduce work
    candidates = module_adjacency_pairs(A, current_part)
    if len(candidates) == 0:
        # fallback: all unordered pairs
        modules = np.unique(current_part)
        candidates = [(int(x), int(y)) for x,y in combinations(modules, 2)]

    if n_jobs is None:
        n_jobs = min(8, (os.cpu_count() or 1))

    if _USE_JOBLIB:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_merge_pair_worker)(A, current_part, pair, pi, epsilon) for pair in candidates
        )
    else:
        # multiprocessing fallback
        import multiprocessing as _mp
        with _mp.Pool(processes=n_jobs) as pool:
            args = [(A, current_part, pair, pi, epsilon) for pair in candidates]
            results = pool.starmap(_evaluate_merge_pair_worker, args)

    return {pair: L for (pair, L) in results}


def greedy_agglomerative_map_parallelized(A,
                                         pi=None,
                                         epsilon=0.15,
                                         min_merge_gain=1e-12,
                                         max_iters=None,
                                         n_jobs=None,
                                         verbose=True):
    """
    Parallelized greedy agglomerative merging (drop-in replacement).
    Same return: (final_partition_array, history_list)
    Key differences: candidate pairs restricted to adjacent-module pairs and evaluated in parallel.
    """
    if hasattr(A, "shape"):
        n = A.shape[0]
    else:
        A = np.asarray(A)
        n = A.shape[0]

    if pi is None:
        if verbose:
            print("Computing stationary distribution pi...")
        pi = power_method_stationary(A, epsilon=0.15)

    part = np.arange(n, dtype=int)
    part = normalize_partition_labels(part)
    current_L = map_equation_L(A, part, pi=pi, epsilon=0.15)
    history = []
    step = 0
    if verbose:
        print(f"Initial L = {current_L:.6f}, n_modules = {np.unique(part).size}")

    max_possible = n - 1
    if max_iters is None:
        max_iters = max_possible
    else:
        max_iters = min(max_iters, max_possible)

    while True:
        modules = np.unique(part)
        m = modules.size
        if verbose:
            print(f"\nStep {step}: modules={m}  computing candidate pairs...")

        if m <= 1 or step >= max_iters:
            if verbose:
                print("Stopping: one module left or reached max_iters.")
            break

        pair_Ls = compute_candidate_pair_Ls_parallel(A, part, pi=pi, epsilon=0.15, n_jobs=n_jobs)
        if not pair_Ls:
            if verbose:
                print("No candidate pairs found. Stopping.")
            break

        best_pair, best_L = min(pair_Ls.items(), key=lambda kv: kv[1])
        best_gain = current_L - best_L

        if verbose:
            print(f"Best candidate {best_pair} -> L_after={best_L:.6f} (gain {best_gain:.6e})")

        if best_gain <= min_merge_gain:
            if verbose:
                print("No merge improves L by more than min_merge_gain. Stopping.")
            break

        # apply best merge
        a, b = best_pair
        part = merge_labels_array(part, a, b)
        step += 1
        L_before = current_L
        current_L = best_L
        history.append({
            'step': step,
            'num_modules': np.unique(part).size,
            'L': current_L,
            'merged_pair': (a,b),
            'L_before': L_before,
            'L_after': current_L,
            'gain': L_before - current_L,
            'partition': part.copy()
        })
        if verbose:
            print(f"Applied merge {a}<-{b}. New L = {current_L:.6f}, modules = {np.unique(part).size}")

    final_part = normalize_partition_labels(part)
    return final_part, history


def plot_greedy_history(history, state_name, figsize=(10,6)):
    """
    Plots: L vs step, #modules vs step, gain per step.
    Expects `history` from greedy_agglomerative_map (list of dicts).
    """
    if not history:
        print("Empty history: no merges performed.")
        return

    steps = [0] + [h['step'] for h in history]
    # L at step 0 is initial L (we can reconstruct it)
    Ls = []
    nums = []
    gains = []
    # initial values
    # try to reconstruct initial L: if first history item has L_before use it
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

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].plot(steps, Ls, marker='o')
    axs[0].set_xlabel('merge step')
    axs[0].set_ylabel('L (bits/step)')
    axs[0].set_title('Description length L vs merge step')
    axs[0].grid(True)

    axs[1].plot(steps, nums, marker='o')
    axs[1].set_xlabel('merge step')
    axs[1].set_ylabel('number of modules')
    axs[1].set_title('#Modules vs merge step')
    axs[1].invert_xaxis()   # optional: show decreasing modules from left->right (uncomment if preferred)
    axs[1].grid(True)

    axs[2].bar(steps, gains)
    axs[2].set_xlabel('merge step')
    axs[2].set_ylabel('gain (L_before - L_after)')
    axs[2].set_title('Gain per merge')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{state_name}_GS_hist.png')
    plt.close(fig)


# print("Plotting greedy search history...")
# plot_greedy_history(history)

from matplotlib.colors import to_rgba
import pygris
import geopandas as gpd

def _node_strength(G, weight="weight"):
    s = {n: 0.0 for n in G.nodes}
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight, 1.0))
        s[u] += w; s[v] += w
    return s

def _distinct_colors(k):
    """
    Return k visually distinct RGBA colors.
    Strategy: concatenate several qualitative palettes, then (if needed)
    fall back to evenly spaced hues.
    """
    pools = ["tab10", "Set3", "Set1", "Set2", "Dark2", "Accent", "Paired", "tab20b", "tab20c"]
    colors = []
    for name in pools:
        cmap = plt.get_cmap(name)
        for i in range(cmap.N):
            colors.append(to_rgba(cmap(i)))
            if len(colors) >= k:
                return colors[:k]
    # Fallback: evenly spaced hues in HSV
    return [plt.cm.hsv(i / max(1, k)) for i in range(k)]


def plot_network_by_module(
    state_name,
    G=None,
    modules=None,
    graphml_path=None,
    with_labels=True,
    figsize=(12, 10),
    dpi=300,
    save_path=None,
    node_scale=1.0,
    verbose=True,
    method='GS'
):
    if G is None and graphml_path is None:
        raise ValueError("Provide either G or graphml_path.")
    if G is None:
        G = nx.read_graphml(graphml_path)

    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes.")
    n0 = next(iter(G.nodes))
    if "lon" not in G.nodes[n0] or "lat" not in G.nodes[n0]:
        raise ValueError("Nodes must have 'lon' and 'lat' attributes.")
    pos = {n: (float(G.nodes[n]["lon"]), float(G.nodes[n]["lat"])) for n in G.nodes}

    # Compute node strength
    s_dict = _node_strength(G, weight="weight")
    s_vals = np.array([s_dict[n] for n in G.nodes], dtype=float)
    if np.all(s_vals == 0):
        s_vals = np.ones_like(s_vals)
    s_vals = s_vals / (s_vals.max() + 1e-12)  # normalize to [0,1]

    node_sizes = node_scale * 300 * (0.25 + 1.75 * np.sqrt(s_vals))

    # Node colors by module
    # modules = [G.nodes[n].get("module", -1) for n in G.nodes]
    uniq_modules = sorted(set(modules))
    colors_pool = _distinct_colors(len(uniq_modules))
    mod_to_color = {m: colors_pool[i] for i, m in enumerate(uniq_modules)}
    node_colors = [mod_to_color[m] for m in modules]

    # Edge widths
    if G.number_of_edges() > 0:
        w_raw = np.array([float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)], dtype=float)
        w_max = float(np.max(w_raw)) if w_raw.size else 1.0
        edge_widths = (5.0 / (w_max + 1e-12)) * w_raw
    else:
        edge_widths = []

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Basemap
    if pygris is not None and gpd is not None:
        try:
            counties = pygris.counties(state=state_name, cb=True, year=2018, cache=True).to_crs(4326)
            if str(state_name).lower() in ("alaska", "ak", "02"):
                counties = counties[counties["GEOID"] != "02016"]
            counties.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.3, alpha=1.0)
        except Exception as e:
            print(f"Warning: Could not load basemap for {state_name}: {e}")
    # Draw edges
    if gpd is not None:
        try:
            lines, widths = [], []
            for (u, v), lw in zip(G.edges(), edge_widths):
                if u in pos and v in pos:
                    lines.append(LineString([pos[u], pos[v]])); widths.append(lw)
            if lines:
                edges_gdf = gpd.GeoDataFrame({"linewidth": widths}, geometry=lines, crs="EPSG:4326")
                edges_gdf.plot(ax=ax, linewidth=edges_gdf["linewidth"], alpha=1, color="cornflowerblue")
            else:
                nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="cornflowerblue", alpha=0.6, ax=ax)
        except Exception:
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="cornflowerblue", alpha=0.6, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="cornflowerblue", alpha=0.6, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, linewidths=0, ax=ax)

    # Draw labels
    if with_labels:
        labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    ax.set_title(f"SCI Network {state_name} (modules, node strength)")

    # plt.tight_layout()
    if method == 'GS':
        plt.savefig(f'plots/{state_name}_GS.png', dpi=dpi) #, bbox_inches="tight")
    else:
        plt.savefig(f'plots/{state_name}_SA.png', dpi=dpi) #, bbox_inches="tight")

    plt.close(fig)

def extract_map_codebooks(A, partition, pi, epsilon=0.15):
    """
    From adjacency A, partition (node->module array), and stationary pi,
    compute q_i, p_i and build the two-level codebooks implied by the map equation.
    Returns dict with:
      - 'q_i', 'p_i', 'q_total'
      - 'index_codebook': {module: {'prob':, 'len':}}
      - 'local_codebooks': {module: [{'token':, 'prob':, 'len':}, ...]}
      - 'L': total description length (bits/step)
    """
    comps = map_equation_L(A, partition, pi=pi, epsilon=0.15, return_components=True)
    q_i = comps['q_i']
    p_i = comps['p_i']
    q_total = comps['q_total']
    H_Q = comps['H_Q']
    H_Pi = comps['H_Pi']
    L = comps['L']
    part_arr = np.asarray(partition, dtype=int)
    modules = np.unique(part_arr)

    # index (module) codebook: probabilities for module names in the index codebook:
    index_codebook = {}
    if q_total > 0:
        for j in modules:
            prob = float(q_i[j] / q_total) if q_total > 0 else 0.0
            length = -np.log2(prob) if prob > 0 else np.inf
            index_codebook[int(j)] = {'prob': prob, 'len_bits': length}
    else:
        # degenerate: no exits at all, index codebook not used
        for j in modules:
            index_codebook[int(j)] = {'prob': 0.0, 'len_bits': 0.0}

    # local codebooks: for each module j build list: exit symbol (if q_i>0) then nodes
    local_codebooks = {}
    for j in modules:
        entries = []
        pj = float(p_i[j])
        # exit symbol
        if pj > 0 and q_i[j] > 0:
            prob_exit = float(q_i[j] / pj)
            len_exit = -np.log2(prob_exit)
            entries.append({'token': '<exit>', 'prob': prob_exit, 'len_bits': len_exit})
        elif pj > 0:
            # no exit mass
            entries.append({'token': '<exit>', 'prob': 0.0, 'len_bits': 0.0})

        # node symbols
        nodes_in_j = np.where(part_arr == j)[0]
        for a in nodes_in_j:
            prob_node = float(pi[a] / pj) if pj > 0 else 0.0
            len_node = -np.log2(prob_node) if prob_node > 0 else np.inf
            entries.append({'token': int(a), 'prob': prob_node, 'len_bits': len_node})
        local_codebooks[int(j)] = entries

    return {
        'q_i': q_i,
        'p_i': p_i,
        'q_total': q_total,
        'index_codebook': index_codebook,
        'local_codebooks': local_codebooks,
        'H_Q': H_Q,
        'H_Pi': H_Pi,
        'L': L
    }

def print_codebooks_summary(codebooks, top_nodes_per_module=5, show_all_nodes=False):
    """
    Pretty-print summary: index codebook and first few entries of each local codebook.
    """
    print(f"Total L = {codebooks['L']:.6f} bits/step  (q_total = {codebooks['q_total']:.6e})\n")
    print("Index codebook (module name probabilities and lengths):")
    for mod, info in sorted(codebooks['index_codebook'].items()):
        print(f"  module {mod:2d}: prob={info['prob']:.6f}, len={info['len_bits']:.3f} bits")
    print("\nPer-module local codebooks (exit first, then top nodes by local prob):")
    for mod, entries in sorted(codebooks['local_codebooks'].items()):
        print(f"\nModule {mod}: p_i={codebooks['p_i'][mod]:.6e}, q_i={codebooks['q_i'][mod]:.6e}")
        # show exit
        for e in entries[:1]:
            print(f"  token={e['token']:>6s} prob={e['prob']:.6f} len={e['len_bits']:.3f}")
        # show top nodes by prob
        node_entries = [e for e in entries if isinstance(e['token'], int)]
        node_entries_sorted = sorted(node_entries, key=lambda x: -x['prob'])

        for e in node_entries_sorted[:top_nodes_per_module]:
            print(f"  node {e['token']:3d}: prob_local={e['prob']:.6f}, len={e['len_bits']:.3f} bits")



def _candidate_targets_for_node(A, part, u, allow_new=True, neighbor_only=True):
    """Return candidate module labels for node u (exclude its current module)."""
    cur = int(part[u])
    if not neighbor_only:
        mods = list(np.unique(part))
        targets = [m for m in mods if m != cur]
        if allow_new:
            targets.append(max(mods) + 1)
        return targets

    # Ensure A is csr_matrix so getrow/getcol exist
    if issparse(A) and not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    if issparse(A):
        out_idx = set(A.getrow(u).indices.tolist())
        in_idx  = set(A.getcol(u).indices.tolist())
    else:
        row = np.asarray(A[u, :])
        col = np.asarray(A[:, u])
        out_idx = set(np.nonzero(row)[0].tolist())
        in_idx = set(np.nonzero(col)[0].tolist())

    neighbor_nodes = out_idx.union(in_idx) - {u}
    neighbor_mods = {int(part[v]) for v in neighbor_nodes if int(v) != u}
    neighbor_mods.discard(cur)
    targets = sorted(list(neighbor_mods))
    if allow_new:
        targets.append(int(max(np.unique(part)) + 1))
    return targets

def simulated_annealing_refine_compact(A,
                                       initial_partition,
                                       pi=None,
                                       epsilon=0.15,
                                       T0=0.2,
                                       cooling_rate=0.93,
                                       steps_per_T=None,
                                       min_T=1e-4,
                                       max_proposals=50000,
                                       neighbor_only=True,
                                       allow_new_module=True,
                                       rng_seed=None,
                                       verbose=True):
    """
    Compact simulated-annealing refinement for the map equation.
    Handles scipy.sparse csr_matrix and newer csr_array/sparray by coercion.
    Returns (best_partition, best_L, history_summary).
    """
    # Ensure sparse arrays become classic csr_matrix for indexing convenience
    if issparse(A) and not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    if hasattr(A, "shape"):
        n = A.shape[0]
    else:
        A = np.asarray(A); n = A.shape[0]

    rng = np.random.default_rng(rng_seed)

    if pi is None:
        if verbose: print("computing stationary pi...")
        pi = power_method_stationary(A, epsilon=0.15)

    current = normalize_partition_labels(np.asarray(initial_partition, dtype=int))
    current_L = map_equation_L(A, current, pi=pi, epsilon=0.15)
    best = current.copy()
    best_L = current_L

    if steps_per_T is None:
        steps_per_T = max(2*n, 100)

    T = float(T0)
    proposals = 0
    history = []

    if verbose:
        print(f"SA start: L={current_L:.6f}, T0={T0}, neighbor_only={neighbor_only}")

    while T > min_T and proposals < max_proposals:
        for _ in range(steps_per_T):
            if proposals >= max_proposals:
                break
            u = int(rng.integers(n))
            cur_mod = int(current[u])
            targets = _candidate_targets_for_node(A, current, u,
                                                 allow_new=allow_new_module,
                                                 neighbor_only=neighbor_only)
            if not targets:
                proposals += 1
                continue
            to_mod = int(targets[rng.integers(len(targets))])
            if to_mod == cur_mod:
                proposals += 1
                continue

            cand = current.copy()
            cand[u] = to_mod
            cand = normalize_partition_labels(cand)
            L_new = map_equation_L(A, cand, pi=pi, epsilon=0.15)
            delta = L_new - current_L

            accept = False
            if delta < 0:
                accept = True
            else:
                try:
                    if rng.random() < np.exp(-delta / T):
                        accept = True
                except OverflowError:
                    accept = False

            if accept:
                current = cand
                current_L = L_new
                if current_L < best_L - 1e-15:
                    best = current.copy()
                    best_L = current_L
                accepted = True
            else:
                accepted = False

            proposals += 1
            # compact history entries to keep memory small
            if proposals % max(1, steps_per_T//10) == 0:
                history.append({'step': proposals, 'T': T, 'current_L': float(current_L), 'best_L': float(best_L), 'accepted': accepted, 'modules': int(np.unique(current).size)})

        T *= cooling_rate
        if verbose and proposals % (steps_per_T*2) == 0:
            print(f"proposal {proposals}: T={T:.4g}, current_L={current_L:.6f}, best_L={best_L:.6f}, modules={np.unique(current).size}")

    best = normalize_partition_labels(best)
    if verbose:
        print(f"SA done: best_L={best_L:.6f}, proposals={proposals}, final modules={np.unique(best).size}")
    return best, best_L, history