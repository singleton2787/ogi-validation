import numpy as np
import matplotlib.pyplot as plt

# --- Setup Parameters ---
K_FIXED = 5       # Fixed number of active modules (k)
N_MAX = 50        # Max number of modules (n)
N_values = np.arange(5, N_MAX + 1)

# Assume message dimension dm and attention dimension are fixed and absorbed into constants
DM = 100 # dm

# --- 1. Complexity Models based on Theorems ---

# Theorem 2: Full Mesh Complexity O(n² * dm)
# The cost scales quadratically with N. We use N*(N-1)*DM as a proxy.
def full_mesh_complexity(N, DM):
    return N * (N - 1) * DM 

# Theorem 3: Top-K Gating Complexity O(k² * dm + n * d_attention)
# The cost is dominated by N for large N (if K is fixed).
# We use K*(K-1)*DM + N*DM_ATTN as a proxy.
def top_k_complexity(N, K, DM):
    DM_ATTN = 50 # Proxy for attention computation cost (da * dc, etc.)
    # Communication cost + Attention cost
    return K * (K - 1) * DM + N * DM_ATTN

# --- 2. Calculate Complexities Across N ---
full_mesh_costs = [full_mesh_complexity(N, DM) for N in N_values]
top_k_costs = [top_k_complexity(N, K_FIXED, DM) for N in N_values]

# --- 3. Visualization (Log-Log Plot is best for complexity) ---
plt.figure(figsize=(8, 6))
plt.plot(N_values, full_mesh_costs, 'r-', label=r'Full Mesh (Theorem 2): $O(N^2 \cdot d_m)$')
plt.plot(N_values, top_k_costs, 'b-', label=r'Top-K Gating (Theorem 3): $O(K^2 \cdot d_m + N \cdot d_{attn})$')

plt.title('Complexity Reduction: Full Mesh vs. Executive Attention Gating', fontsize=14)
plt.xlabel('Number of Modules (N)', fontsize=12)
plt.ylabel('Simulated Computational Cost (Operations)', fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.annotate(f'K={K_FIXED} Fixed', xy=(20, 1e5), xytext=(30, 2e5), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
plt.savefig('visualization/complexity_plot.png')
plt.show()

print(f"Cost at N={N_MAX}:")
print(f"  Full Mesh (N^2): {full_mesh_costs[-1]}")
print(f"  Top-K (K^2+N):   {top_k_costs[-1]}")
print(f"Reduction Ratio: {full_mesh_costs[-1] / top_k_costs[-1]:.1f}x")
