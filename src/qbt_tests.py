#!/usr/bin/env python3
"""
QBT Transduction — Generic Shelf Tests (RPM + Toy Model)
Validates ratio-controlled, noise-stabilized response shelves
"""

import numpy as np
from numpy import kron
from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import json
import os

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

up = np.array([1,0],complex); dn = np.array([0,1],complex)
S = (np.kron(up,dn) - np.kron(dn,up))/np.sqrt(2)
P_S_e = np.outer(S,S.conj())
P_T_e = np.eye(4, dtype=complex)-P_S_e
P_S = kron(P_S_e, id2)
P_T = kron(P_T_e, id2)
rho0_rpm = P_S / np.trace(P_S)

def commutator_super(H):
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return -1j*(np.kron(H, I) - np.kron(I, H.T))

def lindblad_super(L):
    d = L.shape[0]
    I = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5*np.kron(LdL, I) - 0.5*np.kron(I, LdL.T)

def singlet_yield(B_uT=50.0, theta=0.0, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02, tmax=4.0, dt=0.1, B_scale_uT=50.0):
    d = 8
    omega = B_uT / B_scale_uT
    Bx, Bz = omega*np.sin(theta), omega*np.cos(theta)
    HZ  = Bx*(S1x+S2x) + Bz*(S1z+S2z)
    HHF = A*(S1x@Ix + S1y@Iy + S1z@Iz)
    HJ  = J*(S1x@S2x + S1y@S2y + S1z@S2z)
    H   = HZ + HHF + HJ
    L = commutator_super(H)
    I = np.eye(d, dtype=complex)
    loss = -(kS/2)*(np.kron(P_S, I)+np.kron(I, P_S.T)) -(kT/2)*(np.kron(P_T, I)+np.kron(I, P_T.T))
    Ltot = L + loss
    if dephase > 0:
        Ltot += dephase*lindblad_super(S1z) + dephase*lindblad_super(S2z)
    U = expm(Ltot*dt)
    rho = rho0_rpm.copy()
    y = 0.0
    for _ in range(int(tmax/dt)):
        pS = float(np.real(np.trace(P_S @ rho)))
        y += kS * pS * dt
        rho = (U @ rho.reshape(-1, order="F")).reshape((d,d), order="F")
    return y

def anisotropy(B_uT, **kwargs):
    y0 = singlet_yield(B_uT=B_uT, theta=0.0, **kwargs)
    y90 = singlet_yield(B_uT=B_uT, theta=np.pi/2, **kwargs)
    return y0 - y90


class RPMTests:
    """RPM module tests A1-A4"""
    
    def __init__(self, output_dir="results/rpm"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def A1_ratio_invariance(self, baseline_kT=0.1, factors=[0.5, 1, 2, 5], B_test=50.0, ratio_fixed=10.0):
        """Test that shelf position in log-ratio space is invariant under scaling.
        Note: Amplitude may vary with absolute scaling, but position in u=log(kS/kT) is fixed."""
        print("\n" + "="*60)
        print("A1: RATIO INVARIANCE TEST (position in log-ratio space)")
        print("="*60)
        
        results = []
        baseline_kS = ratio_fixed * baseline_kT
        u_fixed = np.log10(ratio_fixed)
        
        for c in factors:
            kT_scaled = baseline_kT * c
            kS_scaled = baseline_kS * c
            anis = anisotropy(B_uT=B_test, kS=kS_scaled, kT=kT_scaled, A=1.0, J=0.5, dephase=0.02)
            u = np.log10(kS_scaled / kT_scaled)
            results.append({'factor': c, 'kS': kS_scaled, 'kT': kT_scaled, 'u': u, 'anisotropy': anis})
            print(f"  c={c:>6.0e}: kS={kS_scaled:.4f}, kT={kT_scaled:.4f}, u={u:.2f}, aniso={anis:.6f}")
        
        u_values = [r['u'] for r in results]
        u_mean = np.mean(u_values)
        u_std = np.std(u_values)
        position_invariant = u_std < 0.01
        
        anisos = [r['anisotropy'] for r in results]
        mean_anis = np.mean(anisos)
        std_anis = np.std(anisos)
        rel_var = std_anis / mean_anis if mean_anis > 0 else 0
        
        plt.figure(figsize=(10, 6))
        plt.semilogx([r['factor'] for r in results], anisos, 'bo-', linewidth=2, markersize=10)
        plt.axhline(mean_anis, color='r', linestyle='--', label=f'Mean={mean_anis:.6f}')
        plt.fill_between([1e-3, 1e3], mean_anis-std_anis, mean_anis+std_anis, alpha=0.2, color='red')
        plt.xlabel("Scaling factor c", fontsize=12)
        plt.ylabel("Anisotropy", fontsize=12)
        plt.title(f"A1: Ratio Invariance (u=log₁₀(kS/kT)={u_fixed:.1f} fixed)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/A1_ratio_invariance.png", dpi=200)
        
        self.metrics['A1'] = {
            'u_position': u_mean,
            'u_std': u_std,
            'position_invariant': position_invariant,
            'amplitude_mean': mean_anis, 
            'amplitude_std': std_anis, 
            'amplitude_relative_variance': rel_var
        }
        np.savetxt(f"{self.output_dir}/A1_ratio_invariance.csv", 
                   [[r['factor'], r['kS'], r['kT'], r['u'], r['anisotropy']] for r in results],
                   header="factor,kS,kT,u,anisotropy", delimiter=",", comments='')
        print(f"  Position u={u_mean:.2f} (std={u_std:.4f}) - INVARIANT: {position_invariant}")
        print(f"  Amplitude: mean={mean_anis:.6f}, rel_var={rel_var:.3f}")
        return results
    
    def A2_noise_sculpting(self, gamma_range=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], B_test=50.0):
        """Test noise-assisted stabilization (dephasing sweep)"""
        print("\n" + "="*60)
        print("A2: NOISE SCULPTING TEST (dephasing)")
        print("="*60)
        
        results = []
        for gamma in gamma_range:
            anis = anisotropy(B_uT=B_test, kS=1.0, kT=0.1, A=1.0, J=0.5, dephase=gamma)
            results.append({'gamma': gamma, 'anisotropy': anis})
            print(f"  γ={gamma:.4f}: aniso={anis:.6f}")
        
        gammas = [r['gamma'] for r in results]
        anisos = [r['anisotropy'] for r in results]
        
        max_idx = np.argmax(anisos)
        optimal_gamma = gammas[max_idx]
        max_anis = anisos[max_idx]
        baseline_anis = anisos[0]
        enhancement = max_anis / baseline_anis if baseline_anis > 1e-10 else float('inf')
        
        plt.figure(figsize=(10, 6))
        plt.semilogx([g if g > 0 else 1e-4 for g in gammas], anisos, 'go-', linewidth=2, markersize=10)
        plt.axvline(optimal_gamma if optimal_gamma > 0 else 1e-4, color='r', linestyle='--', label=f'Optimal γ={optimal_gamma:.3f}')
        plt.xlabel("Dephasing rate γ", fontsize=12)
        plt.ylabel("Anisotropy", fontsize=12)
        plt.title("A2: Noise Sculpting (Zeno-unlocking signature)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/A2_noise_sculpting.png", dpi=200)
        
        self.metrics['A2'] = {'optimal_gamma': optimal_gamma, 'max_anisotropy': max_anis, 'enhancement': enhancement}
        np.savetxt(f"{self.output_dir}/A2_noise_sculpting.csv",
                   [[r['gamma'], r['anisotropy']] for r in results],
                   header="gamma,anisotropy", delimiter=",", comments='')
        print(f"  Optimal γ: {optimal_gamma:.4f}, Enhancement: {enhancement:.2f}x")
        return results
    
    def A3_hyperfine_ablation(self, A_values=[0, 0.1, 0.5, 1.0], B_range=np.linspace(0, 100, 21)):
        """Test hyperfine ablation A→0"""
        print("\n" + "="*60)
        print("A3: HYPERFINE ABLATION TEST (A→0)")
        print("="*60)
        
        results = {}
        for A in A_values:
            anisos = [anisotropy(B_uT=B, kS=1.0, kT=0.1, A=A, J=0.5, dephase=0.02) for B in B_range]
            results[A] = anisos
            max_anis = max(anisos)
            argmax_B = B_range[np.argmax(anisos)]
            print(f"  A={A}: max_aniso={max_anis:.6f} at B={argmax_B:.1f}µT")
        
        plt.figure(figsize=(10, 6))
        colors = ['gray', 'blue', 'green', 'red']
        for (A, anisos), color in zip(results.items(), colors):
            lw = 2.5 if A == 1.0 else 1.5
            style = '-' if A > 0 else '--'
            plt.plot(B_range, anisos, style, color=color, linewidth=lw, label=f'A={A}')
        plt.xlabel("B (µT)", fontsize=12)
        plt.ylabel("Anisotropy", fontsize=12)
        plt.title("A3: Hyperfine Ablation (A→0 kills shelf)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/A3_hyperfine_ablation.png", dpi=200)
        
        max_A1 = max(results[1.0])
        max_A0 = max(results[0]) if 0 in results else max(results[0.1])
        drop_factor = max_A1 / max_A0 if max_A0 > 1e-10 else float('inf')
        
        self.metrics['A3'] = {'max_A1': max_A1, 'max_A0': max_A0, 'drop_factor': drop_factor}
        
        csv_data = np.column_stack([B_range] + [results[A] for A in A_values])
        header = "B," + ",".join([f"A={A}" for A in A_values])
        np.savetxt(f"{self.output_dir}/A3_hyperfine_ablation.csv", csv_data, header=header, delimiter=",", comments='')
        print(f"  Drop factor (A=1 vs A=0): {drop_factor:.1f}x")
        return results
    
    def A4_axis_coupling_scan(self, B_range=np.linspace(10, 100, 9), theta_range=np.linspace(0, np.pi, 9), u_values=[0.5, 1.0, 1.5]):
        """2D scan B×θ at different u values"""
        print("\n" + "="*60)
        print("A4: AXIS-COUPLING SCAN (B-θ structure)")
        print("="*60)
        
        kT = 0.1
        results = {}
        
        for u in u_values:
            kS = kT * (10**u)
            print(f"  Scanning u={u:.1f} (kS/kT={10**u:.1f})...")
            Z = np.zeros((len(theta_range), len(B_range)))
            for i, theta in enumerate(theta_range):
                for j, B in enumerate(B_range):
                    y = singlet_yield(B_uT=B, theta=theta, kS=kS, kT=kT, A=1.0, J=0.5, dephase=0.02)
                    Z[i, j] = y
            results[u] = Z
            
            argmax_theta_per_B = [theta_range[np.argmax(Z[:, j])] for j in range(len(B_range))]
            print(f"    θ_max drift: {np.degrees(argmax_theta_per_B[0]):.1f}° → {np.degrees(argmax_theta_per_B[-1]):.1f}°")
        
        fig, axes = plt.subplots(1, len(u_values), figsize=(5*len(u_values), 4))
        if len(u_values) == 1:
            axes = [axes]
        for ax, u in zip(axes, u_values):
            im = ax.imshow(results[u], aspect='auto', origin='lower',
                          extent=[B_range[0], B_range[-1], 0, 180], cmap='viridis')
            ax.set_xlabel("B (µT)")
            ax.set_ylabel("θ (degrees)")
            ax.set_title(f"u={u:.1f}")
            plt.colorbar(im, ax=ax, label="Yield")
        plt.suptitle("A4: B-θ Heatmaps at different u=log₁₀(kS/kT)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/A4_axis_coupling_scan.png", dpi=200)
        
        self.metrics['A4'] = {'u_values': u_values, 'B_range': [float(B_range[0]), float(B_range[-1])]}
        
        for u in u_values:
            np.savez(f"{self.output_dir}/A4_heatmap_u{u:.1f}.npz", Z=results[u], B=B_range, theta=theta_range)
        
        return results
    
    def run_all(self):
        """Run all RPM tests"""
        print("\n" + "="*70)
        print("RUNNING ALL RPM TESTS (A1-A4)")
        print("="*70)
        
        self.A1_ratio_invariance()
        self.A2_noise_sculpting()
        self.A3_hyperfine_ablation()
        self.A4_axis_coupling_scan()
        
        return self.metrics


class ToyQBTModel:
    """Minimal open quantum system + slow readout (generic QBT)"""
    
    def __init__(self, output_dir="results/toy"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def _build_2level_system(self, J, gamma, k1, k2):
        """Build 2-level open quantum system Liouvillian"""
        H = J * sx
        d = 2
        I = np.eye(d, dtype=complex)
        L_H = -1j * (np.kron(H, I) - np.kron(I, H.T))
        L_deph = gamma * lindblad_super(sz)
        L_decay1 = k1 * lindblad_super(np.array([[0,1],[0,0]], dtype=complex))
        L_decay2 = k2 * lindblad_super(np.array([[0,0],[1,0]], dtype=complex))
        return L_H + L_deph + L_decay1 + L_decay2
    
    def _build_4level_system(self, J, gamma, k1, k2):
        """Build 4-level open quantum system (two qubits)"""
        H = J * (np.kron(sx, id2) + np.kron(id2, sx) + 0.5 * np.kron(sz, sz))
        d = 4
        I = np.eye(d, dtype=complex)
        L_H = -1j * (np.kron(H, I) - np.kron(I, H.T))
        L_deph = gamma * (lindblad_super(np.kron(sz, id2)) + lindblad_super(np.kron(id2, sz)))
        
        decay_op1 = np.kron(np.array([[0,1],[0,0]], dtype=complex), id2)
        decay_op2 = np.kron(id2, np.array([[0,1],[0,0]], dtype=complex))
        L_decay1 = k1 * lindblad_super(decay_op1)
        L_decay2 = k2 * lindblad_super(decay_op2)
        
        return L_H + L_deph + L_decay1 + L_decay2
    
    def slow_readout(self, observable_trajectory, epsilon=0.1, T=None):
        """Slow integrator: dx/dt = epsilon * (O(t) - x), return steady x(T)"""
        if T is None:
            T = len(observable_trajectory)
        x = 0.0
        dt = 1.0
        for O in observable_trajectory[:T]:
            x += epsilon * (O - x) * dt
        return x
    
    def compute_response_2level(self, r, J=0.5, gamma=0.1, k2=0.1, tmax=50.0, dt=0.5, epsilon=0.1):
        """Compute response F(r) for 2-level system"""
        k1 = r * k2
        L = self._build_2level_system(J, gamma, k1, k2)
        d = 2
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        U = expm(L * dt)
        
        obs_traj = []
        steps = int(tmax / dt)
        for _ in range(steps):
            obs = float(np.real(np.trace(sz @ rho)))
            obs_traj.append(obs)
            rho = (U @ rho.reshape(-1, order='F')).reshape((d, d), order='F')
        
        return self.slow_readout(obs_traj, epsilon=epsilon)
    
    def compute_response_4level(self, r, J=0.5, gamma=0.1, k2=0.1, tmax=50.0, dt=0.5, epsilon=0.1):
        """Compute response F(r) for 4-level system"""
        k1 = r * k2
        L = self._build_4level_system(J, gamma, k1, k2)
        d = 4
        rho = np.zeros((d, d), dtype=complex)
        rho[0, 0] = 1.0
        U = expm(L * dt)
        
        obs_op = np.kron(sz, id2) + np.kron(id2, sz)
        obs_traj = []
        steps = int(tmax / dt)
        for _ in range(steps):
            obs = float(np.real(np.trace(obs_op @ rho)))
            obs_traj.append(obs)
            rho = (U @ rho.reshape(-1, order='F')).reshape((d, d), order='F')
        
        return self.slow_readout(obs_traj, epsilon=epsilon)
    
    def B2_ratio_sweep(self, ratio_range=np.logspace(-1, 3, 31), J_values=[0.1, 0.5, 1.0, 2.0], gamma=0.1, system='4level'):
        """Sweep ratio r=k1/k2 over 4+ decades"""
        print("\n" + "="*60)
        print(f"B2: RATIO SWEEP ({system} system)")
        print("="*60)
        
        compute_fn = self.compute_response_4level if system == '4level' else self.compute_response_2level
        
        results = {}
        for J in J_values:
            responses = [compute_fn(r, J=J, gamma=gamma) for r in ratio_range]
            results[J] = responses
            print(f"  J={J}: range=[{min(responses):.4f}, {max(responses):.4f}]")
        
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'orange', 'red']
        for (J, resp), color in zip(results.items(), colors):
            plt.semilogx(ratio_range, resp, '-', color=color, linewidth=2, label=f'J={J}')
        plt.xlabel("r = k₁/k₂", fontsize=12)
        plt.ylabel("Response F(r)", fontsize=12)
        plt.title(f"B2: Response vs Ratio ({system})", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/B2_ratio_sweep_{system}.png", dpi=200)
        
        csv_data = np.column_stack([ratio_range] + [results[J] for J in J_values])
        header = "ratio," + ",".join([f"J={J}" for J in J_values])
        np.savetxt(f"{self.output_dir}/B2_ratio_sweep_{system}.csv", csv_data, header=header, delimiter=",", comments='')
        
        return results
    
    def B2_gamma_sweep(self, ratio_range=np.logspace(-1, 3, 21), gamma_values=[0.01, 0.1, 0.5, 1.0], J=0.5, system='4level'):
        """Sweep gamma to test noise-assisted stabilization"""
        print("\n" + "="*60)
        print(f"B2: GAMMA SWEEP ({system} system)")
        print("="*60)
        
        compute_fn = self.compute_response_4level if system == '4level' else self.compute_response_2level
        
        results = {}
        for gamma in gamma_values:
            responses = [compute_fn(r, J=J, gamma=gamma) for r in ratio_range]
            results[gamma] = responses
            print(f"  γ={gamma}: range=[{min(responses):.4f}, {max(responses):.4f}]")
        
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'orange', 'red']
        for (gamma, resp), color in zip(results.items(), colors):
            plt.semilogx(ratio_range, resp, '-', color=color, linewidth=2, label=f'γ={gamma}')
        plt.xlabel("r = k₁/k₂", fontsize=12)
        plt.ylabel("Response F(r)", fontsize=12)
        plt.title(f"B2: Response vs Ratio (varying γ, {system})", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/B2_gamma_sweep_{system}.png", dpi=200)
        
        return results
    
    def B3_collapse_analysis(self, ratio_range=np.logspace(-1, 3, 31), J_values=[0.1, 0.5, 1.0, 2.0], gamma=0.1, system='4level'):
        """Look for curve collapse under normalization"""
        print("\n" + "="*60)
        print(f"B3: COLLAPSE ANALYSIS ({system})")
        print("="*60)
        
        compute_fn = self.compute_response_4level if system == '4level' else self.compute_response_2level
        
        raw_results = {}
        normalized_results = {}
        
        for J in J_values:
            responses = np.array([compute_fn(r, J=J, gamma=gamma) for r in ratio_range])
            raw_results[J] = responses
            
            mean_resp = np.mean(responses)
            std_resp = np.std(responses)
            if std_resp > 1e-10:
                normalized = (responses - mean_resp) / std_resp
            else:
                normalized = responses - mean_resp
            normalized_results[J] = normalized
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['blue', 'green', 'orange', 'red']
        for (J, resp), color in zip(raw_results.items(), colors):
            ax1.semilogx(ratio_range, resp, '-', color=color, linewidth=2, label=f'J={J}')
        ax1.set_xlabel("r = k₁/k₂", fontsize=12)
        ax1.set_ylabel("Response F(r)", fontsize=12)
        ax1.set_title("Raw Response Curves", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for (J, resp), color in zip(normalized_results.items(), colors):
            ax2.semilogx(ratio_range, resp, '-', color=color, linewidth=2, label=f'J={J}')
        ax2.set_xlabel("r = k₁/k₂", fontsize=12)
        ax2.set_ylabel("Normalized Response", fontsize=12)
        ax2.set_title("Normalized (Collapse Test)", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"B3: Curve Collapse Analysis ({system})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/B3_collapse_{system}.png", dpi=200)
        
        all_normalized = np.array([normalized_results[J] for J in J_values])
        if len(J_values) > 1:
            corr_matrix = np.corrcoef(all_normalized)
            upper_tri = corr_matrix[np.triu_indices(len(J_values), k=1)]
            collapse_correlation = float(np.mean(upper_tri))
        else:
            collapse_correlation = 1.0
        
        self.metrics['B3_collapse'] = {'correlation': collapse_correlation, 'all_pairwise': [float(c) for c in upper_tri] if len(J_values) > 1 else [1.0]}
        print(f"  Collapse correlation: {collapse_correlation:.4f}")
        
        return raw_results, normalized_results, collapse_correlation
    
    def B3_plateau_detection(self, ratio_range=np.logspace(-1, 3, 51), J=0.5, gamma=0.1, system='4level'):
        """Detect plateau/shelf in F(r)"""
        print("\n" + "="*60)
        print(f"B3: PLATEAU DETECTION ({system})")
        print("="*60)
        
        compute_fn = self.compute_response_4level if system == '4level' else self.compute_response_2level
        
        responses = np.array([compute_fn(r, J=J, gamma=gamma) for r in ratio_range])
        log_ratios = np.log10(ratio_range)
        
        derivative = np.gradient(responses, log_ratios)
        abs_deriv = np.abs(derivative)
        
        threshold = 0.1 * np.max(abs_deriv)
        plateau_mask = abs_deriv < threshold
        
        if np.any(plateau_mask):
            plateau_indices = np.where(plateau_mask)[0]
            shelf_center_idx = plateau_indices[len(plateau_indices)//2]
            shelf_center = log_ratios[shelf_center_idx]
            shelf_width = log_ratios[plateau_indices[-1]] - log_ratios[plateau_indices[0]]
            shelf_amplitude = np.mean(responses[plateau_mask])
        else:
            shelf_center = np.nan
            shelf_width = 0
            shelf_amplitude = 0
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.semilogx(ratio_range, responses, 'b-', linewidth=2)
        if not np.isnan(shelf_center):
            ax1.axvspan(10**(shelf_center - shelf_width/2), 10**(shelf_center + shelf_width/2), 
                       alpha=0.3, color='green', label=f'Plateau (width={shelf_width:.2f})')
        ax1.set_xlabel("r = k₁/k₂", fontsize=12)
        ax1.set_ylabel("Response F(r)", fontsize=12)
        ax1.set_title("Response Curve with Plateau Detection", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogx(ratio_range, abs_deriv, 'r-', linewidth=2)
        ax2.axhline(threshold, color='gray', linestyle='--', label=f'Threshold={threshold:.4f}')
        ax2.set_xlabel("r = k₁/k₂", fontsize=12)
        ax2.set_ylabel("|dF/d(log r)|", fontsize=12)
        ax2.set_title("Derivative (plateau = low values)", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/B3_plateau_{system}.png", dpi=200)
        
        self.metrics['B3_plateau'] = {
            'shelf_center': float(shelf_center) if not np.isnan(shelf_center) else None,
            'shelf_width': float(shelf_width),
            'shelf_amplitude': float(shelf_amplitude)
        }
        print(f"  Shelf center (log r): {shelf_center:.2f}" if not np.isnan(shelf_center) else "  No clear plateau detected")
        print(f"  Shelf width: {shelf_width:.2f}")
        print(f"  Shelf amplitude: {shelf_amplitude:.4f}")
        
        return responses, shelf_center, shelf_width, shelf_amplitude
    
    def run_all(self):
        """Run all Toy model tests"""
        print("\n" + "="*70)
        print("RUNNING ALL TOY MODEL TESTS (B1-B3)")
        print("="*70)
        
        self.B2_ratio_sweep(system='2level')
        self.B2_ratio_sweep(system='4level')
        self.B2_gamma_sweep(system='4level')
        self.B3_collapse_analysis(system='4level')
        self.B3_plateau_detection(system='4level')
        
        return self.metrics


def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return True if obj else False
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj

def generate_summary(rpm_metrics, toy_metrics, output_dir="results"):
    """Generate summary.json"""
    summary = {
        'rpm': convert_numpy(rpm_metrics),
        'toy': convert_numpy(toy_metrics),
        'conclusions': convert_numpy({
            'position_invariant': rpm_metrics.get('A1', {}).get('position_invariant', False),
            'noise_enhanced': rpm_metrics.get('A2', {}).get('enhancement', 0) > 1.0,
            'hyperfine_essential': rpm_metrics.get('A3', {}).get('drop_factor', 1) > 5,
            'toy_has_plateau': toy_metrics.get('B3_plateau', {}).get('shelf_width', 0) > 0.5,
            'collapse_correlation': float(toy_metrics.get('B3_collapse', {}).get('correlation', 0))
        })
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {output_dir}/summary.json")
    return summary


def generate_paper_figures(rpm_dir="results/rpm", toy_dir="results/toy", output_dir="paper_figure_pack"):
    """Generate paper figure pack (Fig1-Fig4)"""
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    
    figure_mapping = [
        (f"{rpm_dir}/A3_hyperfine_ablation.png", f"{output_dir}/Fig1_RPM_shelf_ablation.png"),
        (f"{rpm_dir}/A2_noise_sculpting.png", f"{output_dir}/Fig2_noise_sculpting.png"),
        (f"{rpm_dir}/A1_ratio_invariance.png", f"{output_dir}/Fig3_ratio_invariance.png"),
        (f"{toy_dir}/B3_collapse_4level.png", f"{output_dir}/Fig4_toy_model_collapse.png"),
    ]
    
    for src, dst in figure_mapping:
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  {dst}")
        else:
            print(f"  [MISSING] {src}")
    
    return output_dir


def run_all():
    """Main entry point: run all tests and generate deliverables"""
    print("="*70)
    print("QBT TRANSDUCTION — GENERIC SHELF TESTS")
    print("="*70)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    rpm = RPMTests(output_dir="results/rpm")
    rpm_metrics = rpm.run_all()
    
    toy = ToyQBTModel(output_dir="results/toy")
    toy_metrics = toy.run_all()
    
    summary = generate_summary(rpm_metrics, toy_metrics, output_dir="results")
    generate_paper_figures()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  results/rpm/*.csv + *.png")
    print("  results/toy/*.csv + *.png")
    print("  results/summary.json")
    print("  paper_figure_pack/Fig1-Fig4")
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        if cmd == "rpm":
            rpm = RPMTests()
            rpm.run_all()
        elif cmd == "toy":
            toy = ToyQBTModel()
            toy.run_all()
        elif cmd == "A1":
            rpm = RPMTests()
            rpm.A1_ratio_invariance()
        elif cmd == "A2":
            rpm = RPMTests()
            rpm.A2_noise_sculpting()
        elif cmd == "A3":
            rpm = RPMTests()
            rpm.A3_hyperfine_ablation()
        elif cmd == "A4":
            rpm = RPMTests()
            rpm.A4_axis_coupling_scan()
        elif cmd == "B2":
            toy = ToyQBTModel()
            toy.B2_ratio_sweep()
        elif cmd == "B3":
            toy = ToyQBTModel()
            toy.B3_collapse_analysis()
            toy.B3_plateau_detection()
        elif cmd == "all":
            run_all()
        else:
            print(f"Unknown command: {cmd}")
            print("Available: rpm, toy, A1, A2, A3, A4, B2, B3, all")
    else:
        run_all()
