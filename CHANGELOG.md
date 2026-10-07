# Changelog

All notable changes to OperatorForge are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/) and the project adheres
to semantic versioning.

## [0.1.0] - 2026-10-07

### Fixed
- **Darcy solver right-hand side double `1/h**2` (Defect A).** In
  `data/pde.py` the interior forcing term was divided by `h_squared` a second
  time even though the assembled stiffness matrix already carries `1/h_squared`
  on every interior entry. The manufactured-solution test now recovers the
  analytic field with `relative_error <= 1e-6` and `relative_residual <= 1e-10`.
  Boundary-flux compensation terms are intentionally left untouched.
- **OpFuse independent component switches (Defect B).** `OpFuseOperator` now
  accepts `no_gate`, `no_markov`, and `no_highfreq` boolean flags in addition to
  the `ablation=` preset. With all three set the operator is numerically
  identical to `FourierNeuralOperator` (relative difference <= 1e-10), making the
  "all-off" configuration expressible through the public API.
- **Missing optional backend (Defect C).** `BenchmarkPipeline` now marks a run
  `status="skipped"` with a human-readable `reason` when an operator backend is
  unavailable, instead of writing `NaN`/fabricated zeros. Aggregation excludes
  skipped entries so no `Mean of empty slice` warning is produced.
- **Spectral tail error blow-up (Defect D2).** `eval.metrics.spectral_tail_error`
  now normalises by the *total* ground-truth spectral energy rather than by the
  near-zero high-frequency tail energy, keeping the metric finite and bounded for
  smooth fields. Undefined ratios return `NaN` (excluded from aggregation) with an
  explicit `isfinite` guard applied before any threshold comparison.

### Changed
- **Super-resolution gate G2 (Defect D1).** The off-grid super-resolution path of
  OpFuse is the identical FNO diagonal-spectral operator (same seed, ridge, and
  feature map), so its super-resolution error equals FNO's bit-for-bit. G2 now
  reports this structural identity honestly (`structural_identity=True`) and
  exposes the in-grid (`opfuse_ingrid_mean` vs `fno_ingrid_mean`) comparison,
  which is where the architectures genuinely differ.
- **Flagship performance claim (Defect D4).** OpFuse's advantage over the default
  FNO is substantially capacity-driven: an equal-capacity FNO (widened to
  width=20, depth=3) closes most of the gap on burgers1d (0.346 vs 0.326). The
  gated / Markov / high-frequency closure components are net-negative in this
  finite-data 1-D regime (removing any of them improves the error). OpFuse's real
  differentiator vs FNO is its richer nonlinear feature lift, not the three named
  closure components.

### Documented
- **DeepONet on burgers1d (Defect D3).** The cheap closed-form PCA+ridge DeepONet
  baseline scores `relative_l2 = 1.57`, worse than the constant predictor
  (~0.95) under this training budget. This is a documented capacity/architecture
  limitation of the baseline, not a correctness bug; it is reported faithfully and
  not tuned away.

### Added
- Delivery artifacts: `README.md`, `CHANGELOG.md`, `LICENSE` (MIT),
  `requirements.txt`, `requirements.lock.txt`, `Dockerfile`, `Makefile`,
  `.gitignore`, `.gitattributes`.

[0.1.0]: https://example.com/operatorforge/releases/tag/0.1.0
