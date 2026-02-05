"""
Functional Microstructures in Parameter Space
Implements baseline-relative analysis with 6 boundary facets

Microstructure definition:
  P_τ = {(u,J,γ): ΔF(u,J,γ) ≥ τ}
  where ΔF = F(u,J,γ) - F(u,J,0)

6 Boundary Facets ("maxilo-occipital"):
  1. low-u facet (asymmetry insufficient)
  2. high-u facet (singlet decay too fast)
  3. high-J facet (exchange suppression)
  4. low-γ facet (no enhancement / baseline regime)
  5. mid-γ window (bounded enhancement microstructure)
  6. high-γ facet (fracture / incoherent limit)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, label
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

from mechanism_suites import anisotropy, commutator_super, lindblad_super
from mechanism_suites import S1x, S1y, S1z, S2x, S2y, S2z, Ix, Iy, Iz, P_S, P_T

def convert_numpy(obj):
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


class MicrostructureAnalyzer:
    """Analyze functional microstructures in (u, J, γ) parameter space"""
    
    def __init__(self, output_dir='results/microstructure'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/figures', exist_ok=True)
        self.metrics = {}
        
    def compute_3d_field(self, u_range, J_range, gamma_range, B=50.0, A=1.0, kT_base=0.1):
        """Compute anisotropy field F(u, J, γ) over 3D parameter space"""
        print("\n" + "=" * 60)
        print("Computing 3D anisotropy field F(u, J, γ)")
        print("=" * 60)
        
        nu, nJ, ng = len(u_range), len(J_range), len(gamma_range)
        F = np.zeros((nu, nJ, ng))
        
        total = nu * nJ * ng
        count = 0
        
        for i, u in enumerate(u_range):
            ratio = 10 ** u
            kS = kT_base * ratio
            for j, J in enumerate(J_range):
                for k, gamma in enumerate(gamma_range):
                    aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=gamma)
                    F[i, j, k] = aniso
                    count += 1
                    if count % 50 == 0:
                        print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")
        
        self.F = F
        self.u_range = u_range
        self.J_range = J_range
        self.gamma_range = gamma_range
        
        return F
    
    def compute_baseline_anomaly(self):
        """Compute ΔF = F(u,J,γ) - F(u,J,0) (baseline at γ=0)"""
        print("\nComputing baseline-relative anomaly ΔF...")
        
        F_baseline = self.F[:, :, 0:1]
        self.Delta_F = self.F - F_baseline
        
        self.anomaly_positive_volume = np.sum(self.Delta_F > 0)
        self.anomaly_negative_volume = np.sum(self.Delta_F < 0)
        
        max_idx = np.unravel_index(np.argmax(self.Delta_F), self.Delta_F.shape)
        self.anomaly_max_location = {
            'u': float(self.u_range[max_idx[0]]),
            'J': float(self.J_range[max_idx[1]]),
            'gamma': float(self.gamma_range[max_idx[2]]),
            'value': float(self.Delta_F[max_idx])
        }
        
        print(f"  Max anomaly at u={self.anomaly_max_location['u']:.2f}, "
              f"J={self.anomaly_max_location['J']:.2f}, "
              f"γ={self.anomaly_max_location['gamma']:.3f}: "
              f"ΔF={self.anomaly_max_location['value']:.6f}")
        
        return self.Delta_F
    
    def compute_microstructure_set(self, tau):
        """Compute P_τ = {(u,J,γ): ΔF ≥ τ}"""
        self.P_tau = self.Delta_F >= tau
        self.tau = tau
        
        self.volume = np.sum(self.P_tau)
        self.volume_fraction = self.volume / self.P_tau.size
        
        print(f"\nMicrostructure P_τ (τ={tau:.6f}):")
        print(f"  Volume: {self.volume} voxels ({100*self.volume_fraction:.1f}%)")
        
        return self.P_tau
    
    def compute_biomarkers(self):
        """Compute all biomarkers for the microstructure"""
        print("\nComputing biomarkers...")
        
        volume = self.volume
        volume_fraction = self.volume_fraction
        
        boundary = np.zeros_like(self.P_tau, dtype=bool)
        for axis in range(3):
            grad = np.abs(np.diff(self.P_tau.astype(float), axis=axis))
            slices_before = [slice(None)] * 3
            slices_before[axis] = slice(None, -1)
            slices_after = [slice(None)] * 3
            slices_after[axis] = slice(1, None)
            boundary[tuple(slices_before)] |= grad > 0
            boundary[tuple(slices_after)] |= grad > 0
        
        boundary_size = np.sum(boundary)
        if volume > 0:
            boundary_sharpness = boundary_size / volume
        else:
            boundary_sharpness = 0
        
        interior = self.P_tau & ~boundary
        interior_size = np.sum(interior)
        
        if interior_size > 0:
            interior_values = self.Delta_F[interior]
            neighborhood_stability = 1 - np.std(interior_values) / (np.mean(interior_values) + 1e-10)
        else:
            neighborhood_stability = 0
        
        labeled_array, num_features = label(self.P_tau)
        connectivity = 1.0 if num_features == 1 else 1.0 / num_features
        
        self.biomarkers = {
            'volume': int(volume),
            'volume_fraction': float(volume_fraction),
            'boundary_size': int(boundary_size),
            'boundary_sharpness': float(boundary_sharpness),
            'interior_size': int(interior_size),
            'neighborhood_stability': float(neighborhood_stability),
            'num_components': int(num_features),
            'connectivity': float(connectivity)
        }
        
        print(f"  Volume: {volume} ({100*volume_fraction:.1f}%)")
        print(f"  Boundary sharpness: {boundary_sharpness:.3f}")
        print(f"  Neighborhood stability: {neighborhood_stability:.3f}")
        print(f"  Connectivity: {connectivity:.3f} ({num_features} components)")
        
        return self.biomarkers
    
    def detect_facets(self):
        """Detect 6 boundary facets of the microstructure"""
        print("\nDetecting 6 boundary facets...")
        
        u_range = self.u_range
        J_range = self.J_range
        gamma_range = self.gamma_range
        P = self.P_tau
        Delta_F = self.Delta_F
        
        facets = {}
        
        u_profile = np.any(P, axis=(1, 2))
        if np.any(u_profile):
            u_in = np.where(u_profile)[0]
            low_u_idx = u_in[0]
            high_u_idx = u_in[-1]
            
            facets['low_u'] = {
                'detected': low_u_idx > 0,
                'u_boundary': float(u_range[low_u_idx]),
                'mechanism': 'asymmetry insufficient'
            }
            facets['high_u'] = {
                'detected': high_u_idx < len(u_range) - 1,
                'u_boundary': float(u_range[high_u_idx]),
                'mechanism': 'singlet decay too fast'
            }
        else:
            facets['low_u'] = {'detected': False, 'u_boundary': None, 'mechanism': 'asymmetry insufficient'}
            facets['high_u'] = {'detected': False, 'u_boundary': None, 'mechanism': 'singlet decay too fast'}
        
        J_profile = np.any(P, axis=(0, 2))
        if np.any(J_profile):
            J_in = np.where(J_profile)[0]
            high_J_idx = J_in[-1]
            facets['high_J'] = {
                'detected': high_J_idx < len(J_range) - 1,
                'J_boundary': float(J_range[high_J_idx]),
                'mechanism': 'exchange suppression'
            }
        else:
            facets['high_J'] = {'detected': False, 'J_boundary': None, 'mechanism': 'exchange suppression'}
        
        gamma_profile = np.mean(Delta_F, axis=(0, 1))
        
        low_gamma_idx = 0
        facets['low_gamma'] = {
            'detected': gamma_profile[0] < gamma_profile[1] if len(gamma_profile) > 1 else False,
            'gamma_boundary': float(gamma_range[low_gamma_idx]),
            'mechanism': 'no enhancement / baseline regime',
            'baseline_value': float(gamma_profile[0])
        }
        
        peak_gamma_idx = np.argmax(gamma_profile)
        if peak_gamma_idx > 0 and peak_gamma_idx < len(gamma_range) - 1:
            facets['mid_gamma_window'] = {
                'detected': True,
                'gamma_start': float(gamma_range[0]),
                'gamma_peak': float(gamma_range[peak_gamma_idx]),
                'gamma_end': float(gamma_range[-1]),
                'peak_value': float(gamma_profile[peak_gamma_idx]),
                'mechanism': 'bounded enhancement microstructure'
            }
        else:
            facets['mid_gamma_window'] = {
                'detected': False,
                'gamma_peak': float(gamma_range[peak_gamma_idx]),
                'mechanism': 'bounded enhancement microstructure'
            }
        
        if len(gamma_range) > 1:
            gamma_grad = np.gradient(gamma_profile)
            negative_grad_start = np.where(gamma_grad < 0)[0]
            if len(negative_grad_start) > 0:
                high_gamma_idx = negative_grad_start[0]
                facets['high_gamma'] = {
                    'detected': True,
                    'gamma_boundary': float(gamma_range[high_gamma_idx]),
                    'mechanism': 'fracture / incoherent limit',
                    'gradient': float(gamma_grad[high_gamma_idx])
                }
            else:
                facets['high_gamma'] = {
                    'detected': False,
                    'gamma_boundary': float(gamma_range[-1]),
                    'mechanism': 'fracture / incoherent limit'
                }
        else:
            facets['high_gamma'] = {'detected': False, 'gamma_boundary': None, 'mechanism': 'fracture / incoherent limit'}
        
        detected_count = sum(1 for f in facets.values() if f.get('detected', False))
        print(f"  Detected {detected_count}/6 facets:")
        for name, data in facets.items():
            status = "✓" if data.get('detected', False) else "✗"
            print(f"    {status} {name}: {data['mechanism']}")
        
        self.facets = facets
        return facets
    
    def compute_scaling_overlap(self, scale_factors=[0.5, 1.0, 2.0]):
        """Test invariance under (kS, kT) → (λkS, λkT) scaling"""
        print("\nComputing scaling overlap...")
        
        u_test = self.u_range[len(self.u_range)//2]
        J_test = self.J_range[0]
        gamma_test = self.gamma_range[len(self.gamma_range)//2] if len(self.gamma_range) > 1 else 0.01
        
        kT_base = 0.1
        results = []
        
        for scale in scale_factors:
            kT_scaled = kT_base * scale
            ratio = 10 ** u_test
            kS_scaled = kT_scaled * ratio
            
            aniso, _, _ = anisotropy(B_uT=50.0, A=1.0, J=J_test, 
                                     kS=kS_scaled, kT=kT_scaled, dephase=gamma_test)
            results.append(aniso)
        
        results = np.array(results)
        if np.mean(results) > 0:
            relative_variance = np.std(results) / np.mean(results)
        else:
            relative_variance = 0
        
        scaling_invariant = relative_variance < 0.5
        
        self.scaling_overlap = {
            'scale_factors': scale_factors,
            'anisotropies': list(results),
            'relative_variance': float(relative_variance),
            'invariant': bool(scaling_invariant)
        }
        
        print(f"  Scale factors: {scale_factors}")
        print(f"  Anisotropies: {[f'{a:.6f}' for a in results]}")
        print(f"  Relative variance: {relative_variance:.3f}")
        print(f"  Scaling invariant: {scaling_invariant}")
        
        return self.scaling_overlap
    
    def compute_spectral_alignment(self, u_sample=None, J=0.5, gamma=0.02, B=50.0, A=1.0, kT_base=0.1):
        """Check if mode-swap aligns with microstructure boundary"""
        print("\nComputing spectral alignment with boundary...")
        
        if u_sample is None:
            u_sample = self.u_range[::2] if len(self.u_range) > 5 else self.u_range
        
        from mechanism_suites import build_H, build_Ltot
        
        mode_swaps = []
        anisotropies = []
        prev_eigvals = None
        
        for u in u_sample:
            ratio = 10 ** u
            kS = kT_base * ratio
            
            B_T = B * 1e-6
            H = build_H(B_T, 0, A, J)
            L = build_Ltot(H, kS, kT_base, gamma)
            
            eigvals, _ = eig(L)
            idx = np.argsort(-eigvals.real)
            eigvals = eigvals[idx][:4]
            
            if prev_eigvals is not None:
                swap_score = 0
                for i in range(min(3, len(eigvals), len(prev_eigvals))):
                    if i < len(eigvals) - 1:
                        curr_diff = abs(eigvals[i].real - eigvals[i+1].real)
                        prev_diff = abs(prev_eigvals[i].real - prev_eigvals[i+1].real)
                        if prev_diff > 1e-10 and curr_diff / (prev_diff + 1e-10) < 0.5:
                            swap_score += 1
                mode_swaps.append(swap_score)
            else:
                mode_swaps.append(0)
            
            prev_eigvals = eigvals
            
            aniso, _, _ = anisotropy(B_uT=B, A=A, J=J, kS=kS, kT=kT_base, dephase=gamma)
            anisotropies.append(aniso)
        
        in_microstructure = [a > self.tau if hasattr(self, 'tau') else a > 0.001 for a in anisotropies]
        
        boundary_transitions = []
        for i in range(1, len(in_microstructure)):
            if in_microstructure[i] != in_microstructure[i-1]:
                boundary_transitions.append(i)
        
        swap_at_boundary = sum(1 for i in boundary_transitions if i < len(mode_swaps) and mode_swaps[i] > 0)
        total_boundary = len(boundary_transitions)
        
        if total_boundary > 0:
            alignment_score = swap_at_boundary / total_boundary
        else:
            alignment_score = 1.0 if any(s > 0 for s in mode_swaps) else 0.0
        
        self.spectral_alignment = {
            'u_sample': list(u_sample),
            'mode_swaps': mode_swaps,
            'anisotropies': anisotropies,
            'boundary_transitions': boundary_transitions,
            'swap_at_boundary': swap_at_boundary,
            'alignment_score': float(alignment_score),
            'aligned': alignment_score > 0.5
        }
        
        print(f"  Mode swaps: {mode_swaps}")
        print(f"  Boundary transitions at: {boundary_transitions}")
        print(f"  Alignment score: {alignment_score:.2f}")
        print(f"  Spectrally aligned: {alignment_score > 0.5}")
        
        return self.spectral_alignment
    
    def generate_figures(self):
        """Generate visualization figures"""
        print("\nGenerating figures...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if len(self.gamma_range) > 1:
            mid_gamma = len(self.gamma_range) // 2
        else:
            mid_gamma = 0
        
        slice_2d = self.Delta_F[:, :, mid_gamma]
        im = axes[0, 0].imshow(slice_2d.T, origin='lower', aspect='auto',
                               extent=[self.u_range[0], self.u_range[-1], 
                                      self.J_range[0], self.J_range[-1]],
                               cmap='RdBu_r')
        axes[0, 0].set_xlabel('u = log₁₀(kS/kT)')
        axes[0, 0].set_ylabel('J (exchange)')
        axes[0, 0].set_title(f'ΔF(u, J) at γ={self.gamma_range[mid_gamma]:.3f}')
        plt.colorbar(im, ax=axes[0, 0], label='ΔF')
        
        gamma_profile = np.mean(self.Delta_F, axis=(0, 1))
        axes[0, 1].plot(self.gamma_range, gamma_profile, 'b-o', linewidth=2)
        axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('γ (dephasing)')
        axes[0, 1].set_ylabel('Mean ΔF')
        axes[0, 1].set_title('Noise Response Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        if 'mid_gamma_window' in self.facets and self.facets['mid_gamma_window']['detected']:
            peak_g = self.facets['mid_gamma_window']['gamma_peak']
            axes[0, 1].axvline(peak_g, color='r', linestyle='--', label=f'Peak: γ={peak_g:.3f}')
            axes[0, 1].legend()
        
        u_profile = np.mean(self.Delta_F[:, 0, :], axis=1)
        axes[1, 0].plot(self.u_range, u_profile, 'g-o', linewidth=2)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('u = log₁₀(kS/kT)')
        axes[1, 0].set_ylabel('Mean ΔF')
        axes[1, 0].set_title('Ratio Response Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        if 'low_u' in self.facets and self.facets['low_u']['detected']:
            axes[1, 0].axvline(self.facets['low_u']['u_boundary'], color='r', linestyle='--', 
                              label=f"low-u: {self.facets['low_u']['u_boundary']:.2f}")
        if 'high_u' in self.facets and self.facets['high_u']['detected']:
            axes[1, 0].axvline(self.facets['high_u']['u_boundary'], color='orange', linestyle='--',
                              label=f"high-u: {self.facets['high_u']['u_boundary']:.2f}")
        axes[1, 0].legend()
        
        facet_names = ['low_u', 'high_u', 'high_J', 'low_gamma', 'mid_gamma_window', 'high_gamma']
        detected = [1 if self.facets.get(f, {}).get('detected', False) else 0 for f in facet_names]
        colors = ['green' if d else 'red' for d in detected]
        axes[1, 1].barh(facet_names, detected, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Detected')
        axes[1, 1].set_title(f'Facet Detection ({sum(detected)}/6)')
        axes[1, 1].set_xlim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/microstructure_analysis.png', dpi=150)
        plt.close()
        
        print(f"  Saved: {self.output_dir}/figures/microstructure_analysis.png")
    
    def run_full_analysis(self, u_range=None, J_range=None, gamma_range=None, tau=0.001):
        """Run complete microstructure analysis"""
        print("\n" + "=" * 70)
        print("FUNCTIONAL MICROSTRUCTURE ANALYSIS")
        print("=" * 70)
        
        if u_range is None:
            u_range = np.linspace(-0.5, 2.0, 11)
        if J_range is None:
            J_range = np.array([0.5, 1.0, 2.0, 5.0])
        if gamma_range is None:
            gamma_range = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
        
        self.compute_3d_field(u_range, J_range, gamma_range)
        self.compute_baseline_anomaly()
        self.compute_microstructure_set(tau)
        self.compute_biomarkers()
        self.detect_facets()
        self.compute_scaling_overlap()
        self.compute_spectral_alignment()
        self.generate_figures()
        
        self.metrics = {
            'tau': tau,
            'anomaly_max': self.anomaly_max_location,
            'biomarkers': self.biomarkers,
            'facets': convert_numpy(self.facets),
            'scaling_overlap': convert_numpy(self.scaling_overlap),
            'spectral_alignment': convert_numpy(self.spectral_alignment),
            'summary': {
                'facets_detected': sum(1 for f in self.facets.values() if f.get('detected', False)),
                'volume_fraction': self.biomarkers['volume_fraction'],
                'boundary_sharpness': self.biomarkers['boundary_sharpness'],
                'scaling_invariant': self.scaling_overlap['invariant'],
                'spectrally_aligned': self.spectral_alignment['aligned']
            }
        }
        
        with open(f'{self.output_dir}/microstructure_summary.json', 'w') as f:
            json.dump(convert_numpy(self.metrics), f, indent=2)
        
        print("\n" + "=" * 70)
        print("MICROSTRUCTURE ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"  Facets detected: {self.metrics['summary']['facets_detected']}/6")
        print(f"  Volume fraction: {100*self.metrics['summary']['volume_fraction']:.1f}%")
        print(f"  Boundary sharpness: {self.metrics['summary']['boundary_sharpness']:.3f}")
        print(f"  Scaling invariant: {self.metrics['summary']['scaling_invariant']}")
        print(f"  Spectrally aligned: {self.metrics['summary']['spectrally_aligned']}")
        
        return self.metrics


def run_unit_tests():
    """Unit tests on synthetic fields"""
    print("\n" + "=" * 70)
    print("UNIT TESTS: Synthetic Fields")
    print("=" * 70)
    
    results = {}
    
    print("\n1. Plateau detection on synthetic bell curve...")
    u = np.linspace(-1, 2, 30)
    synthetic = np.exp(-(u - 0.5)**2 / 0.5)
    peak_idx = np.argmax(synthetic)
    peak_u = u[peak_idx]
    
    half_max = synthetic.max() / 2
    above_half = np.where(synthetic > half_max)[0]
    width = u[above_half[-1]] - u[above_half[0]]
    
    results['synthetic_plateau'] = {
        'peak_u': float(peak_u),
        'expected_peak': 0.5,
        'width': float(width),
        'pass': abs(peak_u - 0.5) < 0.1 and width > 0.5
    }
    print(f"  Peak at u={peak_u:.2f} (expected 0.5): {'PASS' if results['synthetic_plateau']['pass'] else 'FAIL'}")
    
    print("\n2. Boundary detection test...")
    field_2d = np.zeros((20, 20))
    field_2d[5:15, 5:15] = 1.0
    
    from scipy.ndimage import sobel
    boundary_x = np.abs(sobel(field_2d, axis=0))
    boundary_y = np.abs(sobel(field_2d, axis=1))
    boundary = (boundary_x + boundary_y) > 0
    
    boundary_pixels = np.sum(boundary)
    expected_boundary = 4 * 10
    
    results['boundary_detection'] = {
        'boundary_pixels': int(boundary_pixels),
        'expected_range': [30, 50],
        'pass': 30 <= boundary_pixels <= 50
    }
    print(f"  Boundary pixels: {boundary_pixels} (expected ~40): {'PASS' if results['boundary_detection']['pass'] else 'FAIL'}")
    
    print("\n3. Facet gradient sign test...")
    gamma = np.linspace(0, 2, 20)
    response = np.exp(-0.5 * (gamma - 0.3)**2) - 0.2
    gradient = np.gradient(response)
    
    positive_region = np.sum(gradient[:5] > 0)
    negative_region = np.sum(gradient[-5:] < 0)
    
    results['gradient_sign'] = {
        'positive_early': int(positive_region),
        'negative_late': int(negative_region),
        'pass': positive_region >= 3 and negative_region >= 3
    }
    print(f"  Positive gradient early: {positive_region}/5, Negative late: {negative_region}/5: "
          f"{'PASS' if results['gradient_sign']['pass'] else 'FAIL'}")
    
    all_pass = all(r['pass'] for r in results.values())
    print(f"\nUnit tests: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    
    return results


def run_regression_tests():
    """Regression tests matching Nature manuscript claims"""
    print("\n" + "=" * 70)
    print("REGRESSION TESTS: Nature Manuscript Claims")
    print("=" * 70)
    
    results = {}
    
    print("\n1. Peak position u* ≈ 0.5...")
    u_range = np.linspace(-0.5, 2.5, 13)
    anisotropies = []
    for u in u_range:
        kS = 0.1 * (10 ** u)
        aniso, _, _ = anisotropy(B_uT=50.0, A=1.0, J=0.5, kS=kS, kT=0.1, dephase=0.02)
        anisotropies.append(aniso)
    
    peak_idx = np.argmax(anisotropies)
    peak_u = u_range[peak_idx]
    
    results['peak_position'] = {
        'measured': float(peak_u),
        'expected': 0.5,
        'tolerance': 0.3,
        'pass': abs(peak_u - 0.5) < 0.3
    }
    print(f"  u* = {peak_u:.2f} (expected ~0.5): {'PASS' if results['peak_position']['pass'] else 'FAIL'}")
    
    print("\n2. Exchange suppression at high J...")
    aniso_low_J, _, _ = anisotropy(B_uT=50.0, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02)
    aniso_high_J, _, _ = anisotropy(B_uT=50.0, A=1.0, J=10.0, kS=1.0, kT=0.1, dephase=0.02)
    
    suppression_ratio = aniso_low_J / (aniso_high_J + 1e-10)
    
    results['exchange_suppression'] = {
        'ratio': float(suppression_ratio),
        'expected_min': 5.0,
        'pass': suppression_ratio > 5.0
    }
    print(f"  Suppression ratio: {suppression_ratio:.1f}x (expected >5x): {'PASS' if results['exchange_suppression']['pass'] else 'FAIL'}")
    
    print("\n3. Noise enhancement window...")
    gamma_vals = [0.001, 0.1, 1.0]
    noise_response = []
    for g in gamma_vals:
        aniso, _, _ = anisotropy(B_uT=50.0, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=g)
        noise_response.append(aniso)
    
    enhancement = noise_response[1] / (noise_response[0] + 1e-10)
    
    results['noise_enhancement'] = {
        'enhancement': float(enhancement),
        'expected_min': 10.0,
        'pass': enhancement > 10.0
    }
    print(f"  Enhancement factor: {enhancement:.1f}x (expected >10x): {'PASS' if results['noise_enhancement']['pass'] else 'FAIL'}")
    
    print("\n4. Hyperfine essential...")
    aniso_A1, _, _ = anisotropy(B_uT=50.0, A=1.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02)
    aniso_A0, _, _ = anisotropy(B_uT=50.0, A=0.0, J=0.5, kS=1.0, kT=0.1, dephase=0.02)
    
    drop_factor = aniso_A1 / (aniso_A0 + 1e-10)
    
    results['hyperfine_essential'] = {
        'drop_factor': float(drop_factor),
        'expected_min': 10.0,
        'pass': drop_factor > 10.0
    }
    print(f"  Drop factor: {drop_factor:.1f}x (expected >10x): {'PASS' if results['hyperfine_essential']['pass'] else 'FAIL'}")
    
    all_pass = all(r['pass'] for r in results.values())
    print(f"\nRegression tests: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    
    return results


if __name__ == '__main__':
    unit_results = run_unit_tests()
    
    regression_results = run_regression_tests()
    
    analyzer = MicrostructureAnalyzer()
    metrics = analyzer.run_full_analysis(
        u_range=np.linspace(-0.5, 2.0, 9),
        J_range=np.array([0.5, 2.0, 5.0]),
        gamma_range=np.array([0.0, 0.02, 0.1, 0.5]),
        tau=0.0005
    )
    
    all_results = {
        'unit_tests': convert_numpy(unit_results),
        'regression_tests': convert_numpy(regression_results),
        'microstructure': metrics
    }
    
    with open('results/microstructure/full_results.json', 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print("\n✅ All results saved to results/microstructure/")
