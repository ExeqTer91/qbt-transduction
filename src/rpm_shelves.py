# rpm_shelves.py
# Radical Pair Mechanism (RPM) Simulation for Shelf Detection
# Generates heatmaps of anisotropy vs magnetic field and kS/kT ratio

import numpy as np
from numpy import kron
from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

# --- Spin-1/2 operators ---
sx = np.array([[0,1],[1,0]], dtype=complex)/2
sy = np.array([[0,-1j],[1j,0]], dtype=complex)/2
sz = np.array([[1,0],[0,-1]], dtype=complex)/2
id2 = np.eye(2, dtype=complex)

def op_e1(a): return kron(kron(a, id2), id2)
def op_e2(a): return kron(kron(id2, a), id2)
def op_n(a):  return kron(kron(id2, id2), a)

S1x,S1y,S1z = op_e1(sx), op_e1(sy), op_e1(sz)
S2x,S2y,S2z = op_e2(sx), op_e2(sy), op_e2(sz)
Ix,Iy,Iz    = op_n(sx),  op_n(sy),  op_n(sz)

# --- Singlet / triplet projectors on electron subspace ---
up = np.array([1,0],complex); dn = np.array([0,1],complex)
S = (np.kron(up,dn) - np.kron(dn,up))/np.sqrt(2)
P_S_e = np.outer(S,S.conj())          # 4x4
P_T_e = np.eye(4, dtype=complex)-P_S_e

P_S = kron(P_S_e, id2)                # 8x8
P_T = kron(P_T_e, id2)

# initial: electron singlet, nuclear maximally mixed
rho0 = P_S / np.trace(P_S)

def commutator_super(H):
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return -1j*(np.kron(H, I) - np.kron(I, H.T))

def lindblad_super(L):
    d = L.shape[0]
    I = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5*np.kron(LdL, I) - 0.5*np.kron(I, LdL.T)

def singlet_yield(
    B_uT=50.0, theta=0.0,
    A=1.0, J=0.0,
    kS=1.0, kT=0.1,
    dephase=0.0,
    tmax=10.0, dt=0.01,
    B_scale_uT=50.0
):
    """
    Units: dimensionless. Map B_uT -> omega = B_uT/B_scale_uT.
    A,J,kS,kT,dephase are in same units.
    """
    d = 8
    omega = B_uT / B_scale_uT
    Bx, Bz = omega*np.sin(theta), omega*np.cos(theta)

    HZ  = Bx*(S1x+S2x) + Bz*(S1z+S2z)
    HHF = A*(S1x@Ix + S1y@Iy + S1z@Iz)
    HJ  = J*(S1x@S2x + S1y@S2y + S1z@S2z)
    H   = HZ + HHF + HJ

    L = commutator_super(H)

    I = np.eye(d, dtype=complex)
    # Haberkorn loss: -kS/2{P_S,rho} -kT/2{P_T,rho}
    loss = -(kS/2)*(np.kron(P_S, I)+np.kron(I, P_S.T)) -(kT/2)*(np.kron(P_T, I)+np.kron(I, P_T.T))
    Ltot = L + loss

    if dephase > 0:
        Ltot += dephase*lindblad_super(S1z)
        Ltot += dephase*lindblad_super(S2z)

    U = expm(Ltot*dt)
    rho = rho0.copy()
    y = 0.0
    steps = int(tmax/dt)

    for _ in range(steps):
        pS = float(np.real(np.trace(P_S @ rho)))
        y += kS * pS * dt
        rvec = rho.reshape(-1, order="F")
        rvec = U @ rvec
        rho = rvec.reshape((d,d), order="F")

    return y

def anisotropy(B_uT, **kwargs):
    y0  = singlet_yield(B_uT=B_uT, theta=0.0, **kwargs)
    y90 = singlet_yield(B_uT=B_uT, theta=np.pi/2, **kwargs)
    return y0 - y90

def _compute_anisotropy_point(args):
    """Worker function for parallel computation."""
    i, j, B, kS, A, J, kT, dephase, tmax, dt = args
    a = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT, dephase=dephase, tmax=tmax, dt=dt)
    return i, j, a

def run_heatmap(grid_size=31, save_data=True, parallel=True, 
                tmax=4.0, dt=0.05, A=1.0, J=0.5, kT=0.1, dephase=0.02, suffix=""):
    """
    Run heatmap sweep.
    For quick tests: grid_size=21, tmax=2.0, dt=0.1
    For publication: grid_size=81, tmax=8.0, dt=0.02
    """
    print(f"Running heatmap with {grid_size}x{grid_size} grid...")
    print(f"Parameters: A={A}, J={J}, kT={kT}, dephase={dephase}, tmax={tmax}, dt={dt}")
    start_time = time.time()
    
    Bs = np.linspace(0, 100, grid_size)
    ratios = np.logspace(0, 3, grid_size)

    Z = np.zeros((len(ratios), len(Bs)))
    
    if parallel:
        # Prepare all computation points
        args_list = []
        for i, r in enumerate(ratios):
            kS = r * kT
            for j, B in enumerate(Bs):
                args_list.append((i, j, B, kS, A, J, kT, dephase, tmax, dt))
        
        # Run in parallel
        n_workers = min(cpu_count(), 8)
        print(f"Using {n_workers} parallel workers...")
        
        with Pool(n_workers) as pool:
            results = pool.map(_compute_anisotropy_point, args_list)
        
        # Populate Z matrix
        for i, j, a in results:
            Z[i, j] = a
    else:
        # Sequential computation
        total = len(ratios) * len(Bs)
        count = 0
        for i, r in enumerate(ratios):
            kS = r * kT
            for j, B in enumerate(Bs):
                Z[i,j] = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT, dephase=dephase, tmax=tmax, dt=dt)
                count += 1
                if count % 50 == 0:
                    print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")

    elapsed = time.time() - start_time
    print(f"Computation completed in {elapsed:.1f} seconds")

    # Save raw data
    fname_base = f"heatmap{suffix}" if suffix else "heatmap_shelves"
    if save_data:
        np.savez(f"{fname_base}_data.npz", Z=Z, Bs=Bs, ratios=ratios, 
                 params={"kT": kT, "A": A, "J": J, "dephase": dephase, "tmax": tmax, "dt": dt})
        print(f"Saved: {fname_base}_data.npz")

    # Create figure
    plt.figure(figsize=(10, 8))
    im = plt.imshow(Z, aspect="auto", origin="lower",
                    extent=[Bs[0], Bs[-1], np.log10(ratios[0]), np.log10(ratios[-1])],
                    cmap='viridis')
    plt.xlabel("B (µT)", fontsize=12)
    plt.ylabel("log₁₀(kS/kT)", fontsize=12)
    plt.title(f"Anisotropy heatmap (J={J}, A={A}, dephase={dephase})", fontsize=14)
    plt.colorbar(im, label="Δ singlet yield")
    plt.tight_layout()
    plt.savefig(f"{fname_base}.png", dpi=200)
    print(f"Saved: {fname_base}.png")
    
    # Print summary statistics
    print(f"\nResults summary:")
    print(f"  Max anisotropy: {np.max(Z):.6f}")
    print(f"  Min anisotropy: {np.min(Z):.6f}")
    print(f"  Mean anisotropy: {np.mean(Z):.6f}")
    print(f"  Std anisotropy: {np.std(Z):.6f}")
    
    return Z, Bs, ratios

def run_robustness_test(n_points=10, tmax=4.0, dt=0.05):
    """
    Test shelf persistence across different dephasing rates.
    """
    print("Running robustness test...")
    
    dephase_values = np.logspace(-3, 0, n_points)
    B_test = 50.0
    kS, kT = 1.0, 0.1
    A, J = 1.0, 10.0
    
    aniso_vs_dephase = []
    for d in dephase_values:
        a = anisotropy(B_uT=B_test, A=A, J=J, kS=kS, kT=kT, dephase=d, tmax=tmax, dt=dt)
        aniso_vs_dephase.append(a)
        print(f"  dephase={d:.4f}: anisotropy={a:.6f}")
    
    plt.figure(figsize=(8, 6))
    plt.semilogx(dephase_values, aniso_vs_dephase, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Dephasing rate", fontsize=12)
    plt.ylabel("Anisotropy", fontsize=12)
    plt.title("Robustness: Shelf persistence vs dephasing", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("robustness_test.png", dpi=200)
    print("Saved: robustness_test.png")
    
    return dephase_values, aniso_vs_dephase

def run_ablation_test(n_points=21, tmax=4.0, dt=0.05):
    """
    Ablation: check effect of exchange coupling J on anisotropy.
    Tests J ∈ {0, 0.1, 0.5, 1, 5, 10} as recommended.
    """
    print("Running ablation test (multi-J)...")
    
    Bs = np.linspace(0, 100, n_points)
    J_values = [0, 0.1, 0.5, 1.0, 5.0, 10.0]
    colors = ['gray', 'green', 'blue', 'orange', 'red', 'purple']
    results = {}
    
    for J, color in zip(J_values, colors):
        print(f"  Computing with J={J}...")
        aniso = [anisotropy(B_uT=B, A=1.0, J=J, kS=1.0, kT=0.1, dephase=0.02, tmax=tmax, dt=dt) for B in Bs]
        results[J] = aniso
    
    plt.figure(figsize=(10, 6))
    for J, color in zip(J_values, colors):
        style = '-' if J <= 1 else '--'
        lw = 2.5 if J == 0.5 else 1.5
        plt.plot(Bs, results[J], style, color=color, linewidth=lw, label=f'J={J}')
    
    plt.xlabel("B (µT)", fontsize=12)
    plt.ylabel("Anisotropy (Δ singlet yield)", fontsize=12)
    plt.title("Ablation: Effect of exchange coupling J", fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_test.png", dpi=200)
    print("Saved: ablation_test.png")
    
    return Bs, results

def run_dual_heatmap(grid_size=21, tmax=4.0, dt=0.05):
    """
    Nature Figure 1: Dual heatmap comparison J=0.5 (optimal) vs J=10 (strong exchange).
    Shows that J_small has visible shelves, J_large is flat/dead.
    """
    print("=" * 60)
    print("Running DUAL HEATMAP (Figure 1 candidate)")
    print("=" * 60)
    
    # Run heatmap A: J=0.5 (optimal regime)
    print("\n[A] J=0.5 (optimal, expect shelves)...")
    Z_A, Bs, ratios = run_heatmap(grid_size=grid_size, J=0.5, tmax=tmax, dt=dt, suffix="_J05")
    
    # Run heatmap B: J=10 (strong exchange, expect flat)
    print("\n[B] J=10 (strong exchange, expect flat)...")
    Z_B, _, _ = run_heatmap(grid_size=grid_size, J=10.0, tmax=tmax, dt=dt, suffix="_J10")
    
    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shared colorbar range for fair comparison
    vmin = min(np.min(Z_A), np.min(Z_B))
    vmax = max(np.max(Z_A), np.max(Z_B))
    
    im0 = axes[0].imshow(Z_A, aspect="auto", origin="lower",
                         extent=[Bs[0], Bs[-1], np.log10(ratios[0]), np.log10(ratios[-1])],
                         cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel("B (µT)", fontsize=12)
    axes[0].set_ylabel("log₁₀(kS/kT)", fontsize=12)
    axes[0].set_title("A: J=0.5 (optimal regime)", fontsize=14)
    
    im1 = axes[1].imshow(Z_B, aspect="auto", origin="lower",
                         extent=[Bs[0], Bs[-1], np.log10(ratios[0]), np.log10(ratios[-1])],
                         cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel("B (µT)", fontsize=12)
    axes[1].set_ylabel("log₁₀(kS/kT)", fontsize=12)
    axes[1].set_title("B: J=10 (strong exchange)", fontsize=14)
    
    # Shared colorbar
    fig.colorbar(im1, ax=axes, label="Δ singlet yield", shrink=0.8)
    
    plt.suptitle("Dual Heatmap: Optimal vs Strong Exchange", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("dual_heatmap_figure1.png", dpi=200, bbox_inches='tight')
    print("\nSaved: dual_heatmap_figure1.png")
    
    # Print comparison
    print(f"\nComparison:")
    print(f"  J=0.5: max={np.max(Z_A):.6f}, mean={np.mean(Z_A):.6f}")
    print(f"  J=10:  max={np.max(Z_B):.6f}, mean={np.mean(Z_B):.6f}")
    print(f"  Ratio (max): {np.max(Z_A)/np.max(Z_B):.1f}x")
    
    return Z_A, Z_B, Bs, ratios

def run_robustness_shelf(B_uT=50.0, kS=1.0, kT=0.1, n_points=15, tmax=4.0, dt=0.05):
    """
    Robustness test: scan dephasing to find stability boundary.
    Expect plateau (stable) then collapse past threshold.
    """
    print("Running ROBUSTNESS SHELF test...")
    print(f"Fixed point: B={B_uT}µT, kS/kT={kS/kT}")
    
    dephase_values = np.logspace(-3, 0, n_points)
    A, J = 1.0, 0.5  # Optimal regime
    
    results_J05 = []
    results_J10 = []
    
    for d in dephase_values:
        a05 = anisotropy(B_uT=B_uT, A=A, J=0.5, kS=kS, kT=kT, dephase=d, tmax=tmax, dt=dt)
        a10 = anisotropy(B_uT=B_uT, A=A, J=10.0, kS=kS, kT=kT, dephase=d, tmax=tmax, dt=dt)
        results_J05.append(a05)
        results_J10.append(a10)
        print(f"  dephase={d:.4f}: J=0.5→{a05:.6f}, J=10→{a10:.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(dephase_values, results_J05, 'b-o', linewidth=2, markersize=8, label='J=0.5 (optimal)')
    plt.semilogx(dephase_values, results_J10, 'r--s', linewidth=2, markersize=6, label='J=10 (strong)')
    
    # Mark potential shelf region
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel("Dephasing rate", fontsize=12)
    plt.ylabel("Anisotropy (Δ singlet yield)", fontsize=12)
    plt.title(f"Robustness: Stability boundary (B={B_uT}µT)", fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("robustness_shelf.png", dpi=200)
    print("Saved: robustness_shelf.png")
    
    # Find approximate threshold (where signal drops to 50% of max)
    max_signal = max(results_J05)
    threshold_idx = next((i for i, v in enumerate(results_J05) if v < max_signal * 0.5), len(results_J05)-1)
    print(f"\nStability analysis (J=0.5):")
    print(f"  Max signal: {max_signal:.6f}")
    print(f"  50% threshold at dephase ≈ {dephase_values[threshold_idx]:.4f}")
    
    return dephase_values, results_J05, results_J10

def run_parameter_scan():
    """
    Scan different J values to find regime with visible shelves.
    """
    print("Running parameter scan for J...")
    
    J_values = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
    B = 50.0
    results = []
    
    for J in J_values:
        a = anisotropy(B_uT=B, A=1.0, J=J, kS=1.0, kT=0.1, dephase=0.02, tmax=4.0, dt=0.05)
        results.append((J, a))
        print(f"  J={J}: anisotropy={a:.6f}")
    
    plt.figure(figsize=(8, 6))
    Js, As = zip(*results)
    plt.semilogx(Js, As, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Exchange coupling J", fontsize=12)
    plt.ylabel("Anisotropy at B=50µT", fontsize=12)
    plt.title("Parameter scan: Anisotropy vs J", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("parameter_scan_J.png", dpi=200)
    print("Saved: parameter_scan_J.png")
    
    return results

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("RPM Shelves Simulation")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "test":
            # Ultra-quick test
            run_heatmap(grid_size=11, tmax=2.0, dt=0.1)
        elif cmd == "quick":
            run_heatmap(grid_size=21, tmax=4.0, dt=0.05)
        elif cmd == "medium":
            run_heatmap(grid_size=41, tmax=6.0, dt=0.03)
        elif cmd == "full":
            run_heatmap(grid_size=81, tmax=8.0, dt=0.02)
        elif cmd == "publication":
            run_heatmap(grid_size=101, tmax=10.0, dt=0.02)
        elif cmd == "robustness":
            run_robustness_test()
        elif cmd == "ablation":
            run_ablation_test()
        elif cmd == "scan":
            run_parameter_scan()
        elif cmd == "dual":
            # Nature Figure 1: J=0.5 vs J=10 comparison
            run_dual_heatmap(grid_size=21, tmax=4.0, dt=0.05)
        elif cmd == "shelf":
            # Robustness shelf test
            run_robustness_shelf()
        elif cmd == "nature":
            # All Nature-style figures
            print("\n>>> Running all Nature-style tests <<<\n")
            run_dual_heatmap(grid_size=21, tmax=4.0, dt=0.05)
            run_robustness_shelf()
            run_ablation_test()
        elif cmd == "all":
            run_heatmap(grid_size=41, tmax=6.0, dt=0.03)
            run_robustness_test()
            run_ablation_test()
            run_parameter_scan()
        else:
            print(f"Unknown command: {cmd}")
            print("Available: test, quick, medium, full, publication, robustness, ablation, scan, dual, shelf, nature, all")
    else:
        # Default: run test
        run_heatmap(grid_size=21, tmax=4.0, dt=0.05)
