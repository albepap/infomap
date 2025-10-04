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
        pi = power_method_stationary(A, epsilon=epsilon)

    current = normalize_partition_labels(np.asarray(initial_partition, dtype=int))
    current_L = map_equation_L(A, current, pi=pi, epsilon=epsilon)
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
            L_new = map_equation_L(A, cand, pi=pi, epsilon=epsilon)
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


best_part, best_L, sa_hist = simulated_annealing_refine_compact(A, final_partition, pi=pi,
                                                               T0=0.7, cooling_rate=0.94,
                                                               steps_per_T=2*A.shape[0], max_proposals=20000,
                                                               neighbor_only=True, rng_seed=123, verbose=True)