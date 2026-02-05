"""
F-Tests: Nature-grade validation suite for RPM shelf phenomena
F1 - Spectral mechanism (Liouvillian)
F2 - Initial-state dependence  
F3 - Readout timescale separation
F4 - Detrending robustness
F5 - Non-Markovian noise (optional)
"""

import numpy as np
from numpy import kron
import os
import json
import csv
from scipy.linalg import expm, eig
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from mechanism_suites import anisotropy, singlet_yield, commutator_super, lindblad_super
from mechanism_suites import sx, sy, sz, id2, op_e1, op_e2, op_n
from mechanism_suites import S1x, S1y, S1z, S2x, S2y, S2z, Ix, Iy, Iz
from mechanism_suites import P_S, P_T

def build_H(B_T, theta, A, J, B_scale_uT=50.0):
    """Build Hamiltonian"""
    B_uT = B_T * 1e6
    omega = B_uT / B_scale_uT
    Bx, Bz = omega * np.sin(theta), omega * np.cos(theta)
    HZ = Bx * (S1x + S2x) + Bz * (S1z + S2z)
    HHF = A * (S1x @ Ix + S1y @ Iy + S1z @ Iz)
    HJ = J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    return HZ + HHF + HJ

def build_Ltot(H, kS, kT, dephase):
    """Build total Liouvillian"""
    d = 8
    I = np.eye(d, dtype=complex)
    
    L = commutator_super(H)
    
    sqrt_kS = np.sqrt(kS)
    for i in range(d):
        for j in range(d):
            if P_S[i, j] != 0:
                Lop = np.zeros((d, d), dtype=complex)
                Lop[i, j] = sqrt_kS * P_S[i, j]
                if np.abs(Lop).max() > 1e-10:
                    L += lindblad_super(Lop)
    
    sqrt_kT = np.sqrt(kT)
    for i in range(d):
        for j in range(d):
            if P_T[i, j] != 0:
                Lop = np.zeros((d, d), dtype=complex)
                Lop[i, j] = sqrt_kT * P_T[i, j]
                if np.abs(Lop).max() > 1e-10:
                    L += lindblad_super(Lop)
    
    if dephase > 0:
        Lz1 = np.sqrt(dephase) * S1z
        Lz2 = np.sqrt(dephase) * S2z
        L += lindblad_super(Lz1) + lindblad_super(Lz2)
    
    return L

def singlet_projector():
    """Return singlet projector"""
    return P_S

def convert_numpy(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


class TestF1_Spectral:
    """F1: Spectral mechanism - Liouvillian eigenvalue analysis"""
    
    def __init__(self, output_dir='results/F1'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/../figpack/F1', exist_ok=True)
        self.metrics = {}
    
    def run(self, u_points=None, B_values=None, gamma_values=None, J=0.5, A=1.0, kT_base=0.1):
        """Run spectral analysis at representative parameter points"""
        print("\n" + "=" * 60)
        print("F1: SPECTRAL MECHANISM (Liouvillian)")
        print("=" * 60)
        
        if u_points is None:
            u_points = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        if B_values is None:
            B_values = np.array([50.0, 100.0])
        if gamma_values is None:
            gamma_values = np.array([0.0, 0.1])
        
        n_modes = 8
        all_eigvals = []
        all_gaps = []
        mode_swap_scores = []
        
        prev_eigvals = None
        
        for u in u_points:
            ratio = 10 ** u
            kS = kT_base * ratio
            
            B = B_values[0]
            gamma = gamma_values[-1]
            
            B_T = B * 1e-6
            theta = 0
            H = build_H(B_T, theta, A, J)
            L = build_Ltot(H, kS, kT_base, gamma)
            
            eigvals, eigvecs = eig(L)
            idx = np.argsort(-eigvals.real)
            eigvals = eigvals[idx][:n_modes]
            
            all_eigvals.append(eigvals)
            
            decay_rates = -eigvals.real
            decay_rates_sorted = np.sort(decay_rates[decay_rates > 1e-10])
            if len(decay_rates_sorted) >= 2:
                gap = decay_rates_sorted[1] - decay_rates_sorted[0]
            else:
                gap = 0.0
            all_gaps.append(float(gap.real) if hasattr(gap, 'real') else float(gap))
            
            if prev_eigvals is not None:
                order_change = 0
                for i in range(min(4, len(eigvals), len(prev_eigvals))):
                    if i < len(eigvals) - 1:
                        curr_diff = abs(eigvals[i].real - eigvals[i+1].real)
                        prev_diff = abs(prev_eigvals[i].real - prev_eigvals[i+1].real)
                        if prev_diff > 0 and curr_diff / (prev_diff + 1e-10) < 0.5:
                            order_change += 1
                mode_swap_scores.append(order_change)
            else:
                mode_swap_scores.append(0)
            
            prev_eigvals = eigvals
            print(f"  u={u:.2f}: gap={all_gaps[-1]:.6f}, swap_score={mode_swap_scores[-1]}")
        
        with open(f'{self.output_dir}/eigvals_vs_u.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['u'] + [f'eigval_{i}_real' for i in range(n_modes)] + [f'eigval_{i}_imag' for i in range(n_modes)]
            writer.writerow(header)
            for i, u in enumerate(u_points):
                row = [u] + [e.real for e in all_eigvals[i]] + [e.imag for e in all_eigvals[i]]
                writer.writerow(row)
        
        with open(f'{self.output_dir}/gap_min_vs_u.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['u', 'spectral_gap'])
            for u, gap in zip(u_points, all_gaps):
                writer.writerow([u, gap])
        
        with open(f'{self.output_dir}/mode_swap_score_vs_u.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['u', 'mode_swap_score'])
            for u, score in zip(u_points, mode_swap_scores):
                writer.writerow([u, score])
        
        gap_min_idx = np.argmin(all_gaps)
        gap_min_u = float(u_points[gap_min_idx])
        
        anisotropies = []
        for u in u_points:
            ratio = 10 ** u
            kS = kT_base * ratio
            aniso, _, _ = anisotropy(B_uT=B_values[0], A=A, J=J, kS=kS, kT=kT_base, dephase=gamma_values[-1])
            anisotropies.append(aniso)
        
        aniso_max_idx = np.argmax(anisotropies)
        aniso_max_u = float(u_points[aniso_max_idx])
        
        u_star_aligned = abs(gap_min_u - aniso_max_u) < 0.5 or max(mode_swap_scores) > 0
        
        alignment_metric = 1.0 / (1.0 + abs(gap_min_u - aniso_max_u))
        
        self.metrics = {
            'u_points': list(u_points),
            'gaps': all_gaps,
            'mode_swap_scores': mode_swap_scores,
            'gap_min_u': gap_min_u,
            'aniso_max_u': aniso_max_u,
            'spectral_alignment': bool(u_star_aligned),
            'alignment_metric': float(alignment_metric),
            'max_swap_score': max(mode_swap_scores)
        }
        
        print(f"\n  Gap minimum at u = {gap_min_u:.2f}")
        print(f"  Aniso maximum at u = {aniso_max_u:.2f}")
        print(f"  Spectral alignment: {u_star_aligned}")
        print(f"  Alignment metric: {alignment_metric:.3f}")
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        eigvals_real = np.array([[e.real for e in ev] for ev in all_eigvals])
        for i in range(min(4, n_modes)):
            axes[0].plot(u_points, eigvals_real[:, i], 'o-', label=f'Mode {i}')
        axes[0].set_xlabel('u = log₁₀(kS/kT)')
        axes[0].set_ylabel('Re(eigenvalue)')
        axes[0].set_title('Top Liouvillian Eigenvalues')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(u_points, all_gaps, 'rs-', markersize=8)
        axes[1].axvline(gap_min_u, color='r', linestyle='--', alpha=0.5, label=f'Gap min: u={gap_min_u:.2f}')
        axes[1].axvline(aniso_max_u, color='b', linestyle='--', alpha=0.5, label=f'Aniso max: u={aniso_max_u:.2f}')
        axes[1].set_xlabel('u = log₁₀(kS/kT)')
        axes[1].set_ylabel('Spectral Gap')
        axes[1].set_title('Spectral Gap vs u')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].bar(u_points, mode_swap_scores, width=0.3, color='purple', alpha=0.7)
        axes[2].set_xlabel('u = log₁₀(kS/kT)')
        axes[2].set_ylabel('Mode Swap Score')
        axes[2].set_title('Mode Ordering Changes')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/../figpack/F1/spectral_mechanism.png', dpi=150)
        plt.close()
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        return self.metrics


class TestF2_InitialState:
    """F2: Initial-state dependence - shelf robustness across purity"""
    
    def __init__(self, output_dir='results/F2'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/../figpack/F2', exist_ok=True)
        self.metrics = {}
    
    def run(self, p_values=None, u_range=None, B=50.0, J=0.5, A=1.0, kT_base=0.1, dephase=0.02):
        """Sweep initial state purity and measure shelf properties"""
        print("\n" + "=" * 60)
        print("F2: INITIAL-STATE DEPENDENCE")
        print("=" * 60)
        
        if p_values is None:
            p_values = np.array([1.0, 0.8, 0.5, 0.2])
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 13)
        
        results = []
        all_curves = []
        
        for p in p_values:
            curve = []
            for u in u_range:
                ratio = 10 ** u
                kS = kT_base * ratio
                
                B_T = B * 1e-6
                theta = 0
                H = build_H(B_T, theta, A, J)
                L = build_Ltot(H, kS, kT_base, dephase)
                
                Ps = singlet_projector()
                d = Ps.shape[0]
                rho0_singlet = Ps / np.trace(Ps)
                rho0_mixed = np.eye(d) / float(d)
                rho0 = p * rho0_singlet + (1 - p) * rho0_mixed
                
                tmax, dt = 6.0, 0.03
                n_steps = int(tmax / dt)
                prop = expm(L * dt)
                
                rho_vec = rho0.flatten()
                Y_accum = 0.0
                for _ in range(n_steps):
                    rho_vec = prop @ rho_vec
                    rho = rho_vec.reshape(d, d)
                    Y_accum += np.real(np.trace(Ps @ rho)) * dt
                
                Y_0 = Y_accum
                
                theta = np.pi / 2
                H = build_H(B_T, theta, A, J)
                L = build_Ltot(H, kS, kT_base, dephase)
                prop = expm(L * dt)
                
                rho_vec = rho0.flatten()
                Y_accum = 0.0
                for _ in range(n_steps):
                    rho_vec = prop @ rho_vec
                    rho = rho_vec.reshape(d, d)
                    Y_accum += np.real(np.trace(Ps @ rho)) * dt
                
                Y_90 = Y_accum
                
                aniso = abs(Y_0 - Y_90)
                curve.append(aniso)
            
            curve = np.array(curve)
            all_curves.append(curve)
            
            peak_idx = np.argmax(curve)
            u_star = float(u_range[peak_idx])
            amplitude = float(curve[peak_idx])
            
            half_max = amplitude / 2
            above_half = np.where(curve > half_max)[0]
            if len(above_half) > 0:
                width = float(u_range[above_half[-1]] - u_range[above_half[0]])
            else:
                width = 0.0
            
            results.append({
                'p_singlet': float(p),
                'u_star': u_star,
                'width': width,
                'amplitude': amplitude
            })
            
            print(f"  p={p:.2f}: u*={u_star:.2f}, width={width:.2f}, amp={amplitude:.6f}")
        
        u_stars = [r['u_star'] for r in results]
        u_star_std = np.std(u_stars)
        initial_state_invariant = u_star_std < 0.3
        
        all_curves = np.array(all_curves)
        if all_curves.shape[0] > 1:
            correlations = []
            for i in range(len(p_values) - 1):
                if np.std(all_curves[i]) > 0 and np.std(all_curves[i+1]) > 0:
                    corr = np.corrcoef(all_curves[i], all_curves[i+1])[0, 1]
                    correlations.append(corr)
            collapse_correlation = float(np.mean(correlations)) if correlations else 1.0
        else:
            collapse_correlation = 1.0
        
        self.metrics = {
            'p_values': list(p_values),
            'results': results,
            'u_star_mean': float(np.mean(u_stars)),
            'u_star_std': float(u_star_std),
            'initial_state_invariant': bool(initial_state_invariant),
            'collapse_correlation': collapse_correlation
        }
        
        print(f"\n  u* mean: {np.mean(u_stars):.2f} ± {u_star_std:.2f}")
        print(f"  Initial-state invariant: {initial_state_invariant}")
        print(f"  Collapse correlation: {collapse_correlation:.3f}")
        
        with open(f'{self.output_dir}/metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['p_singlet', 'u_star', 'width', 'amplitude'])
            for r in results:
                writer.writerow([r['p_singlet'], r['u_star'], r['width'], r['amplitude']])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, p in enumerate(p_values):
            axes[0].plot(u_range, all_curves[i], 'o-', label=f'p={p:.1f}', markersize=4)
        axes[0].set_xlabel('u = log₁₀(kS/kT)')
        axes[0].set_ylabel('Anisotropy')
        axes[0].set_title('Shelf vs Initial State Purity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].errorbar(p_values, [r['u_star'] for r in results], 
                        yerr=[r['width']/4 for r in results], 
                        fmt='o-', capsize=5, markersize=8)
        axes[1].axhline(np.mean(u_stars), color='r', linestyle='--', alpha=0.5, 
                       label=f'Mean u*={np.mean(u_stars):.2f}')
        axes[1].set_xlabel('p (singlet fraction)')
        axes[1].set_ylabel('u* (shelf center)')
        axes[1].set_title(f'Shelf Position Stability (std={u_star_std:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/../figpack/F2/initial_state.png', dpi=150)
        plt.close()
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        return self.metrics


class TestF3_Readout:
    """F3: Readout timescale separation"""
    
    def __init__(self, output_dir='results/F3'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/../figpack/F3', exist_ok=True)
        self.metrics = {}
    
    def run(self, tmax_values=None, u_range=None, B=50.0, J=0.5, A=1.0, kT_base=0.1, dephase=0.02):
        """Sweep readout integration time and measure shelf integrity"""
        print("\n" + "=" * 60)
        print("F3: READOUT TIMESCALE SEPARATION")
        print("=" * 60)
        
        if tmax_values is None:
            tmax_values = np.array([1.0, 3.0, 6.0, 12.0])
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 13)
        
        results = []
        all_curves = []
        
        for tmax in tmax_values:
            dt = min(0.03, tmax / 100)
            
            curve = []
            for u in u_range:
                ratio = 10 ** u
                kS = kT_base * ratio
                aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, 
                                         dephase=dephase, tmax=tmax, dt=dt)
                curve.append(aniso)
            
            curve = np.array(curve)
            all_curves.append(curve)
            
            peak_idx = np.argmax(curve)
            amplitude = float(curve[peak_idx])
            
            smoothed = gaussian_filter1d(curve, sigma=1)
            if np.std(curve) > 0:
                collapse_score = np.corrcoef(curve, smoothed)[0, 1]
            else:
                collapse_score = 1.0
            
            half_max = amplitude / 2
            above_half = np.where(curve > half_max)[0]
            if len(above_half) > 0:
                width = float(u_range[above_half[-1]] - u_range[above_half[0]])
            else:
                width = 0.0
            
            integrity = collapse_score * (1 - np.std(curve) / (np.mean(curve) + 1e-10))
            
            results.append({
                'tmax': float(tmax),
                'amplitude': amplitude,
                'width': width,
                'collapse_score': float(collapse_score),
                'integrity': float(integrity)
            })
            
            print(f"  tmax={tmax:.1f}: amp={amplitude:.6f}, width={width:.2f}, integrity={integrity:.3f}")
        
        integrities = [r['integrity'] for r in results]
        readout_required = integrities[-1] > integrities[0] * 1.1
        
        self.metrics = {
            'tmax_values': list(tmax_values),
            'results': results,
            'readout_required': bool(readout_required),
            'integrity_trend': 'increasing' if readout_required else 'stable'
        }
        
        print(f"\n  Readout required for clean shelf: {readout_required}")
        
        with open(f'{self.output_dir}/metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tmax', 'amplitude', 'width', 'collapse_score', 'integrity'])
            for r in results:
                writer.writerow([r['tmax'], r['amplitude'], r['width'], r['collapse_score'], r['integrity']])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, tmax in enumerate(tmax_values):
            label = ['fast', 'medium', 'slow', 'very slow'][i] if i < 4 else f't={tmax}'
            axes[0].plot(u_range, all_curves[i], 'o-', label=f'{label} (t={tmax})', markersize=4)
        axes[0].set_xlabel('u = log₁₀(kS/kT)')
        axes[0].set_ylabel('Anisotropy')
        axes[0].set_title('Shelf vs Readout Timescale')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(tmax_values, [r['integrity'] for r in results], 'o-', 
                    markersize=10, linewidth=2, color='green')
        axes[1].set_xlabel('Readout time (tmax)')
        axes[1].set_ylabel('Shelf Integrity')
        axes[1].set_title(f'Integrity vs Readout Time (trend: {self.metrics["integrity_trend"]})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/../figpack/F3/readout_timescale.png', dpi=150)
        plt.close()
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        return self.metrics


class TestF4_Detrend:
    """F4: Detrending robustness"""
    
    def __init__(self, output_dir='results/F4'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/../figpack/F4', exist_ok=True)
        self.metrics = {}
    
    def run(self, u_range=None, B=50.0, J=0.5, A=1.0, kT_base=0.1, dephase=0.02):
        """Apply detrending and measure shelf robustness"""
        print("\n" + "=" * 60)
        print("F4: DETRENDING ROBUSTNESS")
        print("=" * 60)
        
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 21)
        
        raw_curve = []
        for u in u_range:
            ratio = 10 ** u
            kS = kT_base * ratio
            aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase)
            raw_curve.append(aniso)
        raw_curve = np.array(raw_curve)
        
        def analyze_curve(curve, name):
            peaks, _ = find_peaks(curve, height=np.max(curve) * 0.3)
            n_peaks = len(peaks)
            
            if len(peaks) > 0:
                main_peak = peaks[np.argmax(curve[peaks])]
                u_star = float(u_range[main_peak])
                amplitude = float(curve[main_peak])
            else:
                main_peak = np.argmax(curve)
                u_star = float(u_range[main_peak])
                amplitude = float(curve[main_peak])
            
            half_max = amplitude / 2
            above_half = np.where(curve > half_max)[0]
            if len(above_half) > 0:
                width = float(u_range[above_half[-1]] - u_range[above_half[0]])
            else:
                width = 0.0
            
            return {
                'name': name,
                'u_star': u_star,
                'width': width,
                'amplitude': amplitude,
                'n_peaks': n_peaks
            }
        
        linear_trend = np.polyfit(u_range, raw_curve, 1)
        linear_detrended = raw_curve - np.polyval(linear_trend, u_range)
        linear_detrended = np.maximum(linear_detrended, 0)
        
        poly_trend = np.polyfit(u_range, raw_curve, 2)
        poly_detrended = raw_curve - np.polyval(poly_trend, u_range)
        poly_detrended = np.maximum(poly_detrended, 0)
        
        results = {
            'no_detrend': analyze_curve(raw_curve, 'no_detrend'),
            'linear_detrend': analyze_curve(linear_detrended, 'linear'),
            'poly_detrend': analyze_curve(poly_detrended, 'poly')
        }
        
        ghost_score = results['no_detrend']['n_peaks'] - results['linear_detrend']['n_peaks']
        
        u_stars = [results[k]['u_star'] for k in results]
        detrend_robust = np.std(u_stars) < 0.3 and results['linear_detrend']['amplitude'] > 0
        
        self.metrics = {
            'results': results,
            'ghost_score': ghost_score,
            'detrend_robust': bool(detrend_robust),
            'u_star_consistency': float(np.std(u_stars))
        }
        
        for name, r in results.items():
            print(f"  {name}: u*={r['u_star']:.2f}, width={r['width']:.2f}, amp={r['amplitude']:.6f}, peaks={r['n_peaks']}")
        print(f"\n  Ghost score: {ghost_score}")
        print(f"  Detrend robust: {detrend_robust}")
        
        with open(f'{self.output_dir}/metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'u_star', 'width', 'amplitude', 'n_peaks'])
            for r in results.values():
                writer.writerow([r['name'], r['u_star'], r['width'], r['amplitude'], r['n_peaks']])
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        axes[0].plot(u_range, raw_curve, 'b-', linewidth=2, label='Raw')
        axes[0].plot(u_range, np.polyval(linear_trend, u_range), 'r--', label='Linear trend')
        axes[0].set_xlabel('u = log₁₀(kS/kT)')
        axes[0].set_ylabel('Anisotropy')
        axes[0].set_title('Raw Curve + Trend')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(u_range, raw_curve, 'b-', linewidth=2, alpha=0.5, label='Raw')
        axes[1].plot(u_range, linear_detrended, 'g-', linewidth=2, label='Linear detrend')
        axes[1].plot(u_range, poly_detrended, 'm-', linewidth=2, label='Poly detrend')
        axes[1].set_xlabel('u = log₁₀(kS/kT)')
        axes[1].set_ylabel('Anisotropy')
        axes[1].set_title('Detrended Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        methods = ['no_detrend', 'linear', 'poly']
        u_stars = [results[k]['u_star'] for k in ['no_detrend', 'linear_detrend', 'poly_detrend']]
        amplitudes = [results[k]['amplitude'] for k in ['no_detrend', 'linear_detrend', 'poly_detrend']]
        
        x = np.arange(3)
        axes[2].bar(x - 0.2, u_stars, 0.4, label='u*', color='blue', alpha=0.7)
        ax2 = axes[2].twinx()
        ax2.bar(x + 0.2, amplitudes, 0.4, label='Amplitude', color='orange', alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(methods)
        axes[2].set_ylabel('u* (blue)')
        ax2.set_ylabel('Amplitude (orange)')
        axes[2].set_title(f'Shelf Consistency (ghost={ghost_score})')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/../figpack/F4/detrending.png', dpi=150)
        plt.close()
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        return self.metrics


class TestF5_NonMarkov:
    """F5: Non-Markovian / colored noise (optional)"""
    
    def __init__(self, output_dir='results/F5'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/../figpack/F5', exist_ok=True)
        self.metrics = {}
    
    def run(self, u_range=None, B=50.0, J=0.5, A=1.0, kT_base=0.1, base_dephase=0.02):
        """Compare Markovian vs colored (telegraph) noise - simplified approximation"""
        print("\n" + "=" * 60)
        print("F5: NON-MARKOVIAN NOISE (colored/telegraph)")
        print("=" * 60)
        
        if u_range is None:
            u_range = np.linspace(-0.5, 2.5, 7)
        
        markov_curve = []
        for u in u_range:
            ratio = 10 ** u
            kS = kT_base * ratio
            aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=base_dephase)
            markov_curve.append(aniso)
        markov_curve = np.array(markov_curve)
        
        telegraph_curve = []
        dephase_high = base_dephase * 1.5
        dephase_low = base_dephase * 0.5
        
        for u in u_range:
            ratio = 10 ** u
            kS = kT_base * ratio
            aniso_high, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase_high)
            aniso_low, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=dephase_low)
            telegraph_curve.append((aniso_high + aniso_low) / 2)
        
        telegraph_curve = np.array(telegraph_curve)
        
        def get_shelf_props(curve):
            peak_idx = np.argmax(curve)
            u_star = float(u_range[peak_idx])
            amplitude = float(curve[peak_idx])
            half_max = amplitude / 2
            above_half = np.where(curve > half_max)[0]
            width = float(u_range[above_half[-1]] - u_range[above_half[0]]) if len(above_half) > 0 else 0.0
            return u_star, width, amplitude
        
        markov_props = get_shelf_props(markov_curve)
        telegraph_props = get_shelf_props(telegraph_curve)
        
        u_star_diff = abs(markov_props[0] - telegraph_props[0])
        nonmarkov_robust = u_star_diff < 0.5 and telegraph_props[2] > 0
        
        self.metrics = {
            'markov': {
                'u_star': markov_props[0],
                'width': markov_props[1],
                'amplitude': markov_props[2]
            },
            'telegraph': {
                'u_star': telegraph_props[0],
                'width': telegraph_props[1],
                'amplitude': telegraph_props[2]
            },
            'u_star_difference': float(u_star_diff),
            'nonmarkov_robust': bool(nonmarkov_robust)
        }
        
        print(f"  Markov: u*={markov_props[0]:.2f}, width={markov_props[1]:.2f}, amp={markov_props[2]:.6f}")
        print(f"  Telegraph: u*={telegraph_props[0]:.2f}, width={telegraph_props[1]:.2f}, amp={telegraph_props[2]:.6f}")
        print(f"\n  Non-Markov robust: {nonmarkov_robust}")
        
        with open(f'{self.output_dir}/metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['noise_type', 'u_star', 'width', 'amplitude'])
            writer.writerow(['markov', markov_props[0], markov_props[1], markov_props[2]])
            writer.writerow(['telegraph', telegraph_props[0], telegraph_props[1], telegraph_props[2]])
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        ax.plot(u_range, markov_curve, 'b-o', linewidth=2, markersize=6, label='Markov (white noise)')
        ax.plot(u_range, telegraph_curve, 'r-s', linewidth=2, markersize=6, label='Telegraph (colored)')
        ax.axvline(markov_props[0], color='b', linestyle='--', alpha=0.5)
        ax.axvline(telegraph_props[0], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('u = log₁₀(kS/kT)')
        ax.set_ylabel('Anisotropy')
        ax.set_title(f'Markov vs Non-Markov Noise (Δu*={u_star_diff:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/../figpack/F5/nonmarkov.png', dpi=150)
        plt.close()
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        return self.metrics


def run_all_f_tests():
    """Run all F-tests and update top-level summary"""
    print("\n" + "=" * 70)
    print("RUNNING ALL F-TESTS (Nature-grade validation)")
    print("=" * 70)
    
    f1 = TestF1_Spectral()
    m1 = f1.run()
    
    f2 = TestF2_InitialState()
    m2 = f2.run()
    
    f3 = TestF3_Readout()
    m3 = f3.run()
    
    f4 = TestF4_Detrend()
    m4 = f4.run()
    
    f5 = TestF5_NonMarkov()
    m5 = f5.run()
    
    with open('results/summary.json', 'r') as f:
        summary = json.load(f)
    
    summary['F_tests'] = {
        'F1_spectral': convert_numpy(m1),
        'F2_initial_state': convert_numpy(m2),
        'F3_readout': convert_numpy(m3),
        'F4_detrend': convert_numpy(m4),
        'F5_nonmarkov': convert_numpy(m5)
    }
    
    summary['spectral_alignment'] = m1['spectral_alignment']
    summary['initial_state_invariant'] = m2['initial_state_invariant']
    summary['readout_required'] = m3['readout_required']
    summary['detrend_robust'] = m4['detrend_robust']
    summary['nonmarkov_robust'] = m5['nonmarkov_robust']
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("F-TESTS SUMMARY")
    print("=" * 70)
    print(f"  spectral_alignment: {m1['spectral_alignment']}")
    print(f"  initial_state_invariant: {m2['initial_state_invariant']}")
    print(f"  readout_required: {m3['readout_required']}")
    print(f"  detrend_robust: {m4['detrend_robust']}")
    print(f"  nonmarkov_robust: {m5['nonmarkov_robust']}")
    
    return summary


if __name__ == '__main__':
    run_all_f_tests()
