# Changelog

All notable changes to ChiSao are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/).

## [0.1.1] — 2026-06-23

### Added
- `cuda11` / `cuda12` / `cuda13` install extras for the matching CuPy wheels.
- Working-GPU probe (`chisao._gpu`, `chisao.GPU_OK`): detects a CuPy that imports
  but whose CUDA libraries are broken/mismatched and falls back to CPU instead of
  crashing. Force CPU with `CHISAO_FORCE_CPU=1`.

### Changed
- `chisao.__version__` now reports the distribution version; the vendored
  optimizer lineage is available as `chisao.__core_version__`.
- Full README (the 0.1.0 release shipped a placeholder).

### Fixed
- Recovery tests are GPU-safe (host-convert CuPy arrays via `.get()`).

## [0.1.0] — unreleased

### Added
- First standalone release. Extracted from the SunBURST package
  (`sunburst.utils.chisao` + `sunburst.utils.single_whip`) into an independent
  `chisao` package and repository.
- `chisao.seeding`: decoupled population seeders.
  - `carry_tiger_seed` / `carry_tiger_rays` — structured ray-based seeding
    (vertex-to-vertex, vertex-to-edge, wall-to-wall, QR-orthonormal sunburst),
    lifted from `CarryTigerToMountain` without the evidence machinery.
  - `random_seed` — uniform baseline seeder.
- `optimize(func, bounds, seeder=...)` — high-level wrapper that seeds a
  population and runs `sticky_hands`, mirroring the paper's two-seeder comparison.
- src-layout packaging, MIT license, pytest suite (CPU), GitHub Actions CI.

### Notes
- Core optimizer (`sticky_hands`) carries internal version `3.2.0`, unchanged
  from the SunBURST source.
