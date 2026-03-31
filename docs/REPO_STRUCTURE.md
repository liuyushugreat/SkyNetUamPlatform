## Target Repository Structure (Phase 1)

Goal: separate **production platform** code from **research/paper** artifacts, and provide
a stable place (`modules/`) for reusable, modular capabilities.

```
SkyNetUamPlatform/
├── backend/                         # NestJS backend (production)
├── components/                      # React UI components (production)
├── pages/                           # React pages (production)
├── services/                        # Frontend services/adapters (production)
├── nexus_core/                      # Python core libraries (production + simulation core)
│
├── modules/                         # Reusable capability modules (pluggable)
│   ├── SkyKg/                       # Knowledge-engine paper module bundle
│   │   ├── voxel_airspace_core/     # Sparse octree + builder + A* + FastAPI router
│   │   └── SkyNet_Knowledge_Engine/ # Ontology + neuro-symbolic reasoning
│   ├── SkyRwa/                      # Real-World Assetization & pricing
│   └── SkyFlow/                     # Multi-UAV conflict detection (separate paper)
│
├── research/                        # Research workspace (paper-facing)
│   ├── experiments/
│   │   └── maddpg/                  # IJCAI experiments (MAPPO baseline + metrics + viz)
│   └── papers/                      # Local papers/assets (gitignored by default)
│       └── MyPapers/                # Your existing paper workspaces
│
├── docs/                            # Developer docs (tracked)
│   └── REPO_STRUCTURE.md
│
└── tools/                           # Small maintenance/refactor tools (tracked)
    └── refactor_paths.py
```

### Rules of Thumb
- **backend/** must not import from **research/**.
- **research/** may import from **nexus_core/** and **modules/**.
- **modules/** should stay small, well-documented, and testable.
- Large artifacts (e.g., `trajectories.npy`, videos, PDFs) remain **local** via `.gitignore`.


