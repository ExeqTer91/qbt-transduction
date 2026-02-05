"""
Shelf + Drift + Spectral Mechanism + Initial-State Suites
Implements Suites A-E for comprehensive shelf mechanism analysis

Uses the proven superoperator formalism from rpm_shelves.py
"""

import numpy as np
from numpy import kron
from scipy.linalg import expm, eig
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import json

# --- Spin-1/2 operators ---
sx = np.array([[0,1],[1,0]], dtype=complex)/2
sy = np.array([[0,-1j],[1j,0]], dtype=complex)/2
sz = np.array([[1,0],[0,-1]], dtype=complex)/2
id2 = np.eye(2, dtype=complex)

def op_e1(a): return kron(kron(a, id2), id2)
def op_e2(a): return kron(kron(id2, a), id2)
def op_n(a):  return kron(kron(id2, id2), a)

S1x, S1y, S1z = op_e1(sx), op_e1(sy), op_e1(sz)
S2x, S2y, S2z = op_e2(sx), op_e2(sy), op_e2(sz)
Ix, Iy, Iz = op_n(sx), op_n(sy), op_n(sz)

# --- Singlet / triplet projectors on electron subspace ---
up = np.array([1, 0], complex)
dn = np.array([0, 1], complex)
S_state = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)
P_S_e = np.outer(S_state, S_state.conj())
P_T_e = np.eye(4, dtype=complex) - P_S_e

P_S = kron(P_S_e, id2)
P_T = kron(P_T_e, id2)

# Initial state: electron singlet, nuclear maximally mixed
rho0_default = P_S / np.trace(P_S)

def commutator_super(H):
    """Superoperator for -i[H, rho]"""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return -1j * (np.kron(H, I) - np.kron(I, H.T))

def lindblad_super(L):
    """Superoperator for L @ rho @ L.H - 0.5{L.H @ L, rho}"""
    d = L.shape[0]
    I = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T)

def singlet_yield(B_uT=50.0, theta=0.0, A=1.0, J=0.5, kS=1.0, kT=0.1,
                  dephase=0.02, tmax=6.0, dt=0.03, B_scale_uT=50.0, rho0=None):
    """Compute singlet yield using superoperator formalism"""
    d = 8
    omega = B_uT / B_scale_uT
    Bx, Bz = omega * np.sin(theta), omega * np.cos(theta)
    
    HZ = Bx * (S1x + S2x) + Bz * (S1z + S2z)
    HHF = A * (S1x @ Ix + S1y @ Iy + S1z @ Iz)
    HJ = J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    H = HZ + HHF + HJ
    
    L = commutator_super(H)
    
    I = np.eye(d, dtype=complex)
    loss = -(kS/2) * (np.kron(P_S, I) + np.kron(I, P_S.T)) - (kT/2) * (np.kron(P_T, I) + np.kron(I, P_T.T))
    Ltot = L + loss
    
    if dephase > 0:
        Ltot += dephase * lindblad_super(S1z)
        Ltot += dephase * lindblad_super(S2z)
    
    U = expm(Ltot * dt)
    
    if rho0 is None:
        rho = rho0_default.copy()
    else:
        rho = rho0.copy()
    
    y = 0.0
    steps = int(tmax / dt)
    
    for _ in range(steps):
        pS = float(np.real(np.trace(P_S @ rho)))
        y += kS * pS * dt
        rvec = rho.reshape(-1, order="F")
        rvec = U @ rvec
        rho = rvec.reshape((d, d), order="F")
    
    return y

def anisotropy(B_uT=50.0, n_theta=7, **kwargs):
    """Compute anisotropy as max - min singlet yield over theta angles"""
    thetas = np.linspace(0, np.pi, n_theta)
    yields = [singlet_yield(B_uT=B_uT, theta=t, **kwargs) for t in thetas]
    return max(yields) - min(yields), yields, thetas

def build_liouvillian(B_uT=50.0, theta=0.0, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02, B_scale_uT=50.0):
    """Build full Liouvillian superoperator"""
    d = 8
    omega = B_uT / B_scale_uT
    Bx, Bz = omega * np.sin(theta), omega * np.cos(theta)
    
    HZ = Bx * (S1x + S2x) + Bz * (S1z + S2z)
    HHF = A * (S1x @ Ix + S1y @ Iy + S1z @ Iz)
    HJ = J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    H = HZ + HHF + HJ
    
    L = commutator_super(H)
    
    I = np.eye(d, dtype=complex)
    loss = -(kS/2) * (np.kron(P_S, I) + np.kron(I, P_S.T)) - (kT/2) * (np.kron(P_T, I) + np.kron(I, P_T.T))
    Ltot = L + loss
    
    if dephase > 0:
        Ltot += dephase * lindblad_super(S1z)
        Ltot += dephase * lindblad_super(S2z)
    
    return Ltot


class SuiteA:
    """Suite A — Shelf core (u-scan)"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def run(self, u_range=None, B=50.0, J=0.5, A=1.0, dephase=0.02, kT_base=0.1):
        """u-scan to find shelf center, width, amplitude, collapse score"""
        print("\n" + "=" * 60)
        print("Suite A: SHELF CORE (u-scan)")
        print("=" * 60)
        
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 16)
        
        anisotropies = []
        for u in u_range:
            ratio = 10 ** u
            kS = kT_base * ratio
            aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
            anisotropies.append(aniso)
            print(f"  u={u:.2f} (kS/kT={ratio:.1f}): aniso={aniso:.6f}")
        
        anisotropies = np.array(anisotropies)
        
        max_idx = np.argmax(anisotropies)
        u_star = u_range[max_idx]
        amplitude = anisotropies[max_idx]
        
        half_max = amplitude / 2
        above_half = anisotropies >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            width = u_range[indices[-1]] - u_range[indices[0]]
        else:
            width = 0.0
        
        smoothed = gaussian_filter1d(anisotropies, sigma=2)
        if np.std(anisotropies) > 0:
            collapse_score = np.corrcoef(anisotropies, smoothed)[0, 1]
        else:
            collapse_score = 1.0
        
        self.metrics = {
            'u_star': float(u_star),
            'width': float(width),
            'amplitude': float(amplitude),
            'collapse_score': float(collapse_score)
        }
        
        print(f"\n  Results:")
        print(f"    u* (center) = {u_star:.2f}")
        print(f"    Width (FWHM) = {width:.2f}")
        print(f"    Amplitude = {amplitude:.6f}")
        print(f"    Collapse score = {collapse_score:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(u_range, anisotropies, 'b-', lw=2, label='Anisotropy')
        ax.axvline(u_star, color='r', ls='--', label=f'u* = {u_star:.2f}')
        ax.axhline(half_max, color='g', ls=':', alpha=0.5, label='Half-max')
        ax.set_xlabel(r'u = log$_{10}$(kS/kT)', fontsize=12)
        ax.set_ylabel('Anisotropy', fontsize=12)
        ax.set_title(f'Suite A: Shelf Core\nu*={u_star:.2f}, width={width:.2f}, amp={amplitude:.4f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/A_shelf_core.png", dpi=150)
        plt.close()
        
        np.savetxt(f"{self.output_dir}/A_shelf_core.csv",
                   np.column_stack([u_range, anisotropies]),
                   delimiter=',', header='u,anisotropy', comments='')
        
        return self.metrics


class SuiteB:
    """Suite B — Drift map (B×θ at fixed u)"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def run(self, u_values=None, B_range=None, A_values=None, J=0.5, dephase=0.02, kT_base=0.1):
        """Drift map: argmax θ*(B|u) and flip detection 0↔180"""
        print("\n" + "=" * 60)
        print("Suite B: DRIFT MAP (B×θ at fixed u)")
        print("=" * 60)
        
        if u_values is None:
            u_values = [0.5, 1.0, 1.5]
        if B_range is None:
            B_range = np.linspace(10, 100, 5)
        if A_values is None:
            A_values = [1.0, 0.0]
        
        n_theta = 13
        thetas = np.linspace(0, np.pi, n_theta)
        
        results = {}
        
        for A in A_values:
            A_label = f"A={A:.1f}"
            results[A_label] = {}
            print(f"\n  {A_label}:")
            
            for u in u_values:
                ratio = 10 ** u
                kS = kT_base * ratio
                
                theta_max_list = []
                flip_detected = False
                prev_theta_max = None
                
                for B in B_range:
                    yields = []
                    for theta in thetas:
                        Y = singlet_yield(B_uT=B, theta=theta, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
                        yields.append(Y)
                    
                    theta_max_idx = np.argmax(yields)
                    theta_max = thetas[theta_max_idx]
                    theta_max_deg = np.degrees(theta_max)
                    theta_max_list.append(theta_max_deg)
                    
                    if prev_theta_max is not None:
                        if abs(theta_max_deg - prev_theta_max) > 90:
                            flip_detected = True
                    prev_theta_max = theta_max_deg
                
                results[A_label][f"u={u}"] = {
                    'theta_max': theta_max_list,
                    'B_range': B_range.tolist(),
                    'flip_detected': flip_detected
                }
                
                print(f"    u={u}: θ* range [{min(theta_max_list):.0f}°, {max(theta_max_list):.0f}°], flip={flip_detected}")
        
        self.metrics = results
        
        fig, axes = plt.subplots(1, len(A_values), figsize=(6 * len(A_values), 5))
        if len(A_values) == 1:
            axes = [axes]
        
        for ax, A in zip(axes, A_values):
            A_label = f"A={A:.1f}"
            for u in u_values:
                data = results[A_label][f"u={u}"]
                ax.plot(data['B_range'], data['theta_max'], 'o-', label=f'u={u}', lw=2, markersize=4)
            
            ax.set_xlabel('B (µT)', fontsize=12)
            ax.set_ylabel('θ* (degrees)', fontsize=12)
            ax.set_title(f'Drift Map: {A_label}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-5, 185)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/B_drift_map.png", dpi=150)
        plt.close()
        
        return self.metrics


class SuiteC:
    """Suite C — Noise/disorder agitation"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def run(self, dephase_range=None, u_test=1.0, B=50.0, J=0.5, A=1.0, kT_base=0.1):
        """Sweep γ (dephasing) and compute shelf integrity metric"""
        print("\n" + "=" * 60)
        print("Suite C: NOISE/DISORDER AGITATION")
        print("=" * 60)
        
        if dephase_range is None:
            dephase_range = np.array([0.001, 0.01, 0.1, 0.5, 1.0])
        
        u_scan = np.linspace(0, 2, 7)
        
        integrity_scores = []
        collapse_scores = []
        continuity_scores = []
        amplitudes = []
        
        baseline_curves = None
        
        for dephase in dephase_range:
            curve = []
            for u in u_scan:
                ratio = 10 ** u
                kS = kT_base * ratio
                aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
                curve.append(aniso)
            
            curve = np.array(curve)
            
            smoothed = gaussian_filter1d(curve, sigma=1)
            if np.std(curve) > 0:
                collapse_score = np.corrcoef(curve, smoothed)[0, 1]
            else:
                collapse_score = 1.0
            
            if baseline_curves is None:
                baseline_curves = curve
                continuity_score = 1.0
            else:
                if np.std(curve) > 0 and np.std(baseline_curves) > 0:
                    continuity_score = np.corrcoef(curve, baseline_curves)[0, 1]
                else:
                    continuity_score = 1.0
            
            integrity = 0.5 * (collapse_score + continuity_score)
            
            collapse_scores.append(float(collapse_score))
            continuity_scores.append(float(continuity_score))
            integrity_scores.append(float(integrity))
            amplitudes.append(float(np.max(curve)))
            
            print(f"  γ={dephase:.3f}: collapse={collapse_score:.3f}, cont={continuity_score:.3f}, integrity={integrity:.3f}, amp={np.max(curve):.4f}")
        
        stab_idx = np.argmax(integrity_scores)
        stabilizing = dephase_range[stab_idx] > dephase_range[0]
        fracturing = integrity_scores[-1] < integrity_scores[0]
        
        self.metrics = {
            'dephase_values': dephase_range.tolist(),
            'integrity_scores': integrity_scores,
            'collapse_scores': collapse_scores,
            'continuity_scores': continuity_scores,
            'amplitudes': amplitudes,
            'optimal_dephase': float(dephase_range[stab_idx]),
            'stabilizing': bool(stabilizing),
            'fracturing': bool(fracturing)
        }
        
        print(f"\n  Optimal γ: {dephase_range[stab_idx]:.3f}")
        print(f"  γ stabilizes: {stabilizing}")
        print(f"  γ fractures (eventually): {fracturing}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax = axes[0]
        ax.semilogx(dephase_range, integrity_scores, 'bo-', lw=2, label='Integrity')
        ax.semilogx(dephase_range, collapse_scores, 'g^--', lw=1.5, label='Collapse')
        ax.semilogx(dephase_range, continuity_scores, 'rs--', lw=1.5, label='Continuity')
        ax.set_xlabel('γ (dephasing)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Suite C: Shelf Integrity vs Dephasing', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.semilogx(dephase_range, amplitudes, 'mo-', lw=2)
        ax.set_xlabel('γ (dephasing)', fontsize=12)
        ax.set_ylabel('Max Anisotropy', fontsize=12)
        ax.set_title('Amplitude vs Dephasing', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/C_noise_agitation.png", dpi=150)
        plt.close()
        
        np.savetxt(f"{self.output_dir}/C_noise_agitation.csv",
                   np.column_stack([dephase_range, integrity_scores, collapse_scores,
                                   continuity_scores, amplitudes]),
                   delimiter=',',
                   header='dephase,integrity,collapse,continuity,amplitude',
                   comments='')
        
        return self.metrics


class SuiteD:
    """Suite D — Spectral mechanism (Liouvillian spectrum)"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def run(self, u_range=None, B=50.0, theta=0.0, J=0.5, A=1.0, dephase=0.02, kT_base=0.1):
        """Compute Liouvillian spectrum vs u"""
        print("\n" + "=" * 60)
        print("Suite D: SPECTRAL MECHANISM (Liouvillian)")
        print("=" * 60)
        
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 13)
        
        n_modes = 10
        
        all_eigenvalues = []
        gaps = []
        anisotropies = []
        
        for u in u_range:
            ratio = 10 ** u
            kS = kT_base * ratio
            
            L = build_liouvillian(B_uT=B, theta=theta, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
            
            eigenvalues, _ = eig(L)
            sorted_idx = np.argsort(eigenvalues.real)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            
            decay_rates = -eigenvalues.real
            decay_rates = np.sort(decay_rates)
            if len(decay_rates) >= 2:
                gap = decay_rates[1] - decay_rates[0]
            else:
                gap = 0.0
            
            all_eigenvalues.append(eigenvalues[:n_modes])
            gaps.append(gap.real)
            
            aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
            anisotropies.append(aniso)
            
            print(f"  u={u:.2f}: gap={gap.real:.4f}, aniso={aniso:.6f}")
        
        all_eigenvalues = np.array(all_eigenvalues)
        gaps = np.array(gaps)
        anisotropies = np.array(anisotropies)
        
        gap_min_idx = np.argmin(np.abs(gaps))
        gap_min_u = u_range[gap_min_idx]
        
        aniso_max_idx = np.argmax(anisotropies)
        aniso_max_u = u_range[aniso_max_idx]
        
        gap_aniso_corr = np.corrcoef(gaps, anisotropies)[0, 1]
        
        avoided_crossings = []
        for i in range(len(u_range) - 1):
            for m in range(min(n_modes - 1, all_eigenvalues.shape[1] - 1)):
                diff_curr = abs(all_eigenvalues[i, m] - all_eigenvalues[i, m + 1])
                diff_next = abs(all_eigenvalues[i + 1, m] - all_eigenvalues[i + 1, m + 1])
                if diff_curr < 0.1 and diff_next < 0.1 and diff_curr > 0.01:
                    avoided_crossings.append({
                        'u': float(u_range[i]),
                        'modes': (m, m + 1),
                        'gap': float(diff_curr)
                    })
        
        self.metrics = {
            'gap_min_u': float(gap_min_u),
            'gap_min_value': float(gaps[gap_min_idx]),
            'aniso_max_u': float(aniso_max_u),
            'gap_aniso_correlation': float(gap_aniso_corr),
            'n_avoided_crossings': len(avoided_crossings),
            'avoided_crossings': avoided_crossings[:5]
        }
        
        print(f"\n  Results:")
        print(f"    Gap minimum at u = {gap_min_u:.2f}")
        print(f"    Aniso maximum at u = {aniso_max_u:.2f}")
        print(f"    Gap-aniso correlation: {gap_aniso_corr:.3f}")
        print(f"    Avoided crossings detected: {len(avoided_crossings)}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        for m in range(min(5, all_eigenvalues.shape[1])):
            ax.plot(u_range, all_eigenvalues[:, m], '-', lw=1.5, label=f'Mode {m}')
        ax.set_xlabel(r'u = log$_{10}$(kS/kT)', fontsize=11)
        ax.set_ylabel('Re(λ)', fontsize=11)
        ax.set_title('Liouvillian Spectrum', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(u_range, gaps, 'b-', lw=2)
        ax.axvline(gap_min_u, color='r', ls='--', label=f'Gap min @ u={gap_min_u:.2f}')
        ax.set_xlabel(r'u = log$_{10}$(kS/kT)', fontsize=11)
        ax.set_ylabel('Spectral Gap', fontsize=11)
        ax.set_title('Gap vs u', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(u_range, anisotropies, 'g-', lw=2)
        ax.axvline(aniso_max_u, color='r', ls='--', label=f'Max @ u={aniso_max_u:.2f}')
        ax.set_xlabel(r'u = log$_{10}$(kS/kT)', fontsize=11)
        ax.set_ylabel('Anisotropy', fontsize=11)
        ax.set_title('Anisotropy vs u', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        sc = ax.scatter(gaps, anisotropies, c=u_range, cmap='viridis', s=40)
        plt.colorbar(sc, ax=ax, label='u')
        ax.set_xlabel('Spectral Gap', fontsize=11)
        ax.set_ylabel('Anisotropy', fontsize=11)
        ax.set_title(f'Gap-Aniso Correlation: {gap_aniso_corr:.3f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/D_spectral_mechanism.png", dpi=150)
        plt.close()
        
        np.savetxt(f"{self.output_dir}/D_spectral_mechanism.csv",
                   np.column_stack([u_range, gaps, anisotropies]),
                   delimiter=',', header='u,gap,anisotropy', comments='')
        
        return self.metrics


class SuiteE:
    """Suite E — Initial state (purity/mixing)"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
    
    def run(self, p_range=None, u_test=1.0, B=50.0, J=0.5, A=1.0, dephase=0.02, kT_base=0.1):
        """Parameterize initialization: p (pure singlet → mixed)"""
        print("\n" + "=" * 60)
        print("Suite E: INITIAL STATE (purity/mixing)")
        print("=" * 60)
        
        if p_range is None:
            p_range = np.linspace(0, 1, 11)
        
        d = 8
        I_mixed = np.eye(d, dtype=complex) / d
        PS_norm = P_S / np.trace(P_S).real
        
        ratio = 10 ** u_test
        kS = kT_base * ratio
        
        anisotropies = []
        yields_0 = []
        yields_90 = []
        
        for p in p_range:
            rho0 = p * PS_norm + (1 - p) * I_mixed
            
            Y_0 = singlet_yield(B_uT=B, theta=0, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase, rho0=rho0)
            Y_90 = singlet_yield(B_uT=B, theta=np.pi/2, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase, rho0=rho0)
            
            aniso = abs(Y_0 - Y_90)
            
            anisotropies.append(aniso)
            yields_0.append(Y_0)
            yields_90.append(Y_90)
            
            print(f"  p={p:.2f}: Y(0°)={Y_0:.4f}, Y(90°)={Y_90:.4f}, aniso={aniso:.6f}")
        
        anisotropies = np.array(anisotropies)
        
        if anisotropies[0] > 1e-10:
            relative_change = (anisotropies[-1] - anisotropies[0]) / anisotropies[0]
        else:
            relative_change = float('inf') if anisotropies[-1] > 0 else 0.0
        
        threshold_p = None
        for i, a in enumerate(anisotropies):
            if a < 0.1 * anisotropies[-1]:
                threshold_p = p_range[i]
                break
        
        self.metrics = {
            'p_range': p_range.tolist(),
            'anisotropies': [float(a) for a in anisotropies],
            'yields_0': [float(y) for y in yields_0],
            'yields_90': [float(y) for y in yields_90],
            'relative_change': float(relative_change) if np.isfinite(relative_change) else None,
            'threshold_p': float(threshold_p) if threshold_p else None,
            'pure_singlet_aniso': float(anisotropies[-1]),
            'mixed_aniso': float(anisotropies[0])
        }
        
        print(f"\n  Results:")
        print(f"    Pure singlet (p=1) aniso: {anisotropies[-1]:.6f}")
        print(f"    Fully mixed (p=0) aniso: {anisotropies[0]:.6f}")
        if np.isfinite(relative_change):
            print(f"    Relative change: {relative_change:.2%}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax = axes[0]
        ax.plot(p_range, anisotropies, 'bo-', lw=2, markersize=8)
        ax.set_xlabel('p (purity: 0=mixed, 1=singlet)', fontsize=12)
        ax.set_ylabel('Anisotropy', fontsize=12)
        ax.set_title(f'Suite E: Shelf Dependence on Initial State\nu={u_test}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(p_range, yields_0, 'r^-', lw=2, label='θ=0°')
        ax.plot(p_range, yields_90, 'gs-', lw=2, label='θ=90°')
        ax.set_xlabel('p (purity)', fontsize=12)
        ax.set_ylabel('Singlet Yield', fontsize=12)
        ax.set_title('Yield at θ=0° and θ=90°', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/E_initial_state.png", dpi=150)
        plt.close()
        
        np.savetxt(f"{self.output_dir}/E_initial_state.csv",
                   np.column_stack([p_range, anisotropies, yields_0, yields_90]),
                   delimiter=',', header='p,anisotropy,yield_0,yield_90', comments='')
        
        return self.metrics


def convert_numpy(obj):
    """Convert numpy types for JSON"""
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


def run_all_suites(output_dir="results"):
    """Run all suites and generate deliverables"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("figpack", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RUNNING ALL MECHANISM SUITES (A-E)")
    print("=" * 70)
    
    suite_a = SuiteA(output_dir)
    metrics_a = suite_a.run()
    
    suite_b = SuiteB(output_dir)
    metrics_b = suite_b.run()
    
    suite_c = SuiteC(output_dir)
    metrics_c = suite_c.run()
    
    suite_d = SuiteD(output_dir)
    metrics_d = suite_d.run()
    
    suite_e = SuiteE(output_dir)
    metrics_e = suite_e.run()
    
    existing_summary = {}
    if os.path.exists(f"{output_dir}/summary.json"):
        try:
            with open(f"{output_dir}/summary.json", 'r') as f:
                existing_summary = json.load(f)
        except:
            pass
    
    mechanism_results = {
        'suite_a': convert_numpy(metrics_a),
        'suite_b': convert_numpy(metrics_b),
        'suite_c': convert_numpy(metrics_c),
        'suite_d': convert_numpy(metrics_d),
        'suite_e': convert_numpy(metrics_e),
        'mechanism_conclusions': convert_numpy({
            'shelf_found': metrics_a['amplitude'] > 0.001,
            'shelf_width': metrics_a['width'],
            'shelf_center': metrics_a['u_star'],
            'drift_detected': any(
                metrics_b.get(f'A={A}', {}).get(f'u={u}', {}).get('flip_detected', False)
                for A in [1.0, 0.0] for u in [0.5, 1.0, 1.5]
            ),
            'noise_stabilizes': metrics_c.get('stabilizing', False),
            'noise_fractures': metrics_c.get('fracturing', False),
            'spectral_gap_correlated': abs(metrics_d.get('gap_aniso_correlation', 0)) > 0.3,
            'purity_sensitive': metrics_e.get('pure_singlet_aniso', 0) > metrics_e.get('mixed_aniso', 0) * 2
        })
    }
    
    summary = {**existing_summary, **mechanism_results}
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    import shutil
    figure_mapping = [
        (f"{output_dir}/A_shelf_core.png", "figpack/Fig1_shelf_core.png"),
        (f"{output_dir}/B_drift_map.png", "figpack/Fig2_drift_map.png"),
        (f"{output_dir}/C_noise_agitation.png", "figpack/Fig3_noise_agitation.png"),
        (f"{output_dir}/D_spectral_mechanism.png", "figpack/Fig4_spectral_mechanism.png"),
        (f"{output_dir}/E_initial_state.png", "figpack/Fig5_initial_state.png"),
    ]
    
    for src, dst in figure_mapping:
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  {dst}")
    
    print(f"\nSaved: {output_dir}/summary.json")
    print("\n" + "=" * 70)
    print("ALL SUITES COMPLETE")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    run_all_suites()
