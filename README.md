# QBT Transduction: Ratio-Controlled Regimes in Quantum Biology

Simulation code and results for the paper:

**A General Theory of Quantum–Biological Transduction: Ratio-Controlled Regimes, Noise-Bounded Robustness, and Universal Response Plateaus**

*Andrei-Sebastian Ursachi*

---

## Key Results

### Universal Response Plateau

| u = log₁₀(kS/kT) | kS/kT | Anisotropy (×10⁻³) |
|------------------|-------|---------------------|
| -0.50 | 0.32 | 1.10 |
| 0.00 | 1.0 | 2.84 |
| **0.50** | **3.2** | **4.92** (peak) |
| 1.00 | 10.0 | 3.40 |
| 1.50 | 31.6 | 0.75 |
| 2.00 | 100 | 0.09 |

- **Peak position:** u* = 0.50 (kS/kT ≈ 3.2)
- **Plateau width:** 1.0 decade
- **Collapse score:** 0.948

### Noise Sculpting

| γ | Enhancement |
|---|-------------|
| 0.001 | 1× |
| 0.1 | 93× |
| 2.0 | **903×** |

### Spectral Mechanism

| u | Mode Swap Score |
|-----|-----------------|
| -0.5 | 0 (off-plateau) |
| 0.0–2.0 | **2** (on-plateau) |

Gap-anisotropy correlation: r = -0.15 (weak)

### Nature-Grade Validation (F-Tests)

| Test | Result |
|------|--------|
| F1: Spectral alignment | ✅ TRUE |
| F2: Initial-state invariant | ✅ TRUE |
| F3: Readout required | FALSE |
| F4: Detrend robust | ✅ TRUE |
| F5: Non-Markov robust | ✅ TRUE |

---

## Repository Structure

```
src/
├── rpm_shelves.py          # Core RPM simulation
├── mechanism_suites.py     # Suites A-E
├── f_tests.py              # F1-F5 validation
└── qbt_tests.py            # RPM + Toy model tests

results/
├── summary.json            # All metrics
├── A_shelf_core.csv        # Table 1
├── C_noise_agitation.csv
├── F1-F5/                  # Nature-grade outputs

figures/
├── Fig1_shelf_core.png
├── Fig2_drift_map.png
├── Fig3_noise_agitation.png
├── Fig4_spectral_mechanism.png
├── Fig5_initial_state.png
└── F1-F5/                  # Validation figures
```

---

## Usage

```bash
# Run all mechanism suites
python src/mechanism_suites.py

# Run Nature-grade tests
python src/f_tests.py

# Run RPM/Toy tests
python src/qbt_tests.py
```

---

## Dependencies

- Python 3.11+
- NumPy, SciPy, Matplotlib

---

## License

MIT
