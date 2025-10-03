import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dynamic_weighting import DynamicWeighting # Assume Phi is imported

# --- Setup Parameters ---
D_CONTEXT = 32
D_TASK = 16
N_MODULES = 10
N_TRIALS = 500

# Initialize the attention network
torch.manual_seed(42)
attention_net = DynamicWeighting(D_CONTEXT, D_TASK, N_MODULES)

# 1. Calculate Theoretical Lipschitz Constant L
Wc_norm = torch.linalg.norm(attention_net.Wc.weight, ord=2).item()
Wa_norm = torch.linalg.norm(attention_net.Wa.weight, ord=2).item()
L_theoretical = (Wa_norm * Wc_norm) / 2.0

# 2. Run Empirical Trials
input_diffs = []
output_diffs = []
et_fixed = torch.randn(1, D_TASK) # Fixed task embedding

for _ in range(N_TRIALS):
    # Generate base context c1
    c1 = torch.randn(1, D_CONTEXT) 
    
    # Generate perturbation magnitude epsilon (must be small)
    epsilon = np.random.uniform(0.001, 0.1) 
    
    # Create perturbation vector
    perturbation = torch.randn_like(c1)
    # Normalize perturbation and scale to exactly epsilon
    perturbation = epsilon * (perturbation / torch.linalg.norm(perturbation))
    c2 = c1 + perturbation

    # Calculate Phi for both contexts
    Phi1 = attention_net(c1, et_fixed)
    Phi2 = attention_net(c2, et_fixed)
    
    # Record L2-norms
    input_diff = torch.linalg.norm(c1 - c2).item()
    output_diff = torch.linalg.norm(Phi1 - Phi2).item()
    
    input_diffs.append(input_diff)
    output_diffs.append(output_diff)

# 3. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(input_diffs, output_diffs, label='Empirical Data Points (||Φ1-Φ2||)', s=10)

# Plot the theoretical stability boundary (y = L_theoretical * x)
x_line = np.linspace(0, max(input_diffs) * 1.1, 100)
y_line = L_theoretical * x_line
plt.plot(x_line, y_line, 'r--', label=f'Theoretical Boundary: y = {L_theoretical:.2f}x')

plt.title('Theorem 1: Stability Guarantee (Lipschitz Continuity)', fontsize=14)
plt.xlabel('Input Perturbation L2-Norm (||c1 - c2||₂)', fontsize=12)
plt.ylabel('Output Change L2-Norm (||Φ1 - Φ2||₂)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('visualization/stability_plot.png')
plt.show()

print(f"L_Theoretical (Predicted Max Ratio): {L_theoretical:.4f}")
print(f"L_Empirical (Max Observed Ratio): {np.max(np.array(output_diffs) / np.array(input_diffs)):.4f}")
