import json
import sys
from pathlib import Path

from analysis import *

# Get all state names from graphml files
#state_names = [f.stem for f in Path('state_networks_graphml').glob('*.graphml')]

state_names = ['Puerto_Rico','South_Carolina']

results = []
for state_name in state_names:
    print(f"\n{'='*60}")
    print(f"Processing: {state_name}")
    print(f"{'='*60}")
    
    try:
        # Load graph
        print(f"[1/5] Loading graph...")
        G = nx.read_graphml(f'state_networks_graphml/{state_name}.graphml')
        print(f"      Nodes: {len(nx.nodes(G))}")
        A = nx.adjacency_matrix(G)
        
        # Compute stationary distribution
        print(f"[2/5] Computing stationary distribution...")
        pi = power_method_stationary(A, epsilon=0.15)
        
        # Greedy search
        print(f"[3/5] Running greedy agglomerative search...")
        final_partition_greedy, history = greedy_agglomerative_map_parallelized(
            A, pi=pi, epsilon=0.15, min_merge_gain=1e-10, n_jobs=6, verbose=False
        )
        L_from_GS = map_equation_L(A, final_partition_greedy, pi=pi, epsilon=0.15)
        print(f"      GS complete: L={L_from_GS:.6f}, modules={np.unique(final_partition_greedy).size}")
        
        # Plot history
        plot_greedy_history(history, state_name)
        
        # Simulated annealing
        print(f"[4/5] Running simulated annealing refinement...")
        best_part_sa, best_L_sa, sa_hist = simulated_annealing_refine_compact(
            A, final_partition_greedy, pi=pi, 
            T0=0.05, 
            cooling_rate=0.98,
            min_T=0.005, 
            steps_per_T=2*A.shape[0], 
            max_proposals=15000,
            neighbor_only=True, rng_seed=123, verbose=True
        )
        print(f"      SA complete: L={best_L_sa:.6f}, modules={np.unique(best_part_sa).size}")

        history = sa_hist
        steps = [h.get('step', i) for i,h in enumerate(history)]
        currentLs = [h.get('current_L', np.nan) for h in history]
        bestLs = [h.get('best_L', np.nan) for h in history]
        accepted = [1 if h.get('accepted', False) else 0 for h in history]
        deltas = [h.get('delta', 0.0) for h in history]

        # L traces
        plt.figure(figsize=(10,4))
        plt.plot(steps, currentLs, label='current L', alpha=0.7)
        plt.plot(steps, bestLs, label='best L', alpha=0.9)
        plt.xlabel('proposal step'); plt.ylabel('L (bits/step)')
        plt.title('Simulated annealing: current and best L over steps')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(f'plots/{state_name}_SA_hist.png')
        
        # Plot network
        print(f"[5/5] Plotting network...")
        
        results.append({
            'state_name': state_name,
            'L_from_GS': float(L_from_GS),
            'best_L_sa': float(best_L_sa),
            'final_partition_greedy': final_partition_greedy.tolist(),
            'best_part_sa': best_part_sa.tolist()
        })
        print(f"✓ SUCCESS: {state_name}")

        if isinstance(state_name, str):
            state_name = state_name.replace("_", " ")
        plot_network_by_module(state_name, G, final_partition_greedy, f'{state_name}.graphml', method='GS')
        plot_network_by_module(state_name, G, best_part_sa, f'{state_name}.graphml', method='SA')
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        full_trace = traceback.format_exc()
        print(full_trace)
        results.append({'state_name': state_name, 'error': str(e), 'traceback': full_trace})
    
    # Save after each state
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results.json")

print(f"\n{'='*60}")
print(f"COMPLETE: {len(results)} states processed")
print(f"{'='*60}")