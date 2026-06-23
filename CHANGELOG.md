# Changelog

All notable changes to ChiSao are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/).

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
