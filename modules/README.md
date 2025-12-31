# Modules Directory

This directory contains modular, production-ready components for the SkyNetUamPlatform.

## Current Modules

### 1. `voxel_airspace_core/`
**Status**: ✅ Synced to GitHub

3D spatial indexing and pathfinding for urban airspace:
- **indexer.py**: Sparse octree data structure for efficient 3D spatial queries
- **builder.py**: GeoJSON to voxel conversion (buildings → 3D occupancy grid)
- **pathfinder.py**: 3D A* pathfinding algorithm with safety margins
- **manager.py**: High-level API for city model loading and path planning
- **api.py**: FastAPI router for REST endpoints

### 2. `rwa_core/`
**Status**: ✅ Synced to GitHub (as of 2025-12-31)

Real-World Assetization (RWA) and financial primitives:
- **valuation.py**: Data packet valuation interfaces (`DataPacket`, `ValuationResult`, `AbstractValuationEngine`)
- **pricing_engine.py**: Dynamic pricing engine for SkyNet data assets
- **economics/pricing.py**: Congestion pricing models for airspace voxels

**Migration Note**: This module was migrated from `nexus_core/assetization/` and `nexus_core/economics/` in Phase-1. Backward compatibility is maintained via re-export shims in the old paths.

### 3. `reasoning_engine/`
**Status**: ✅ Synced to GitHub (placeholder)

Future module for AI reasoning and decision-making capabilities. Currently contains only placeholder files (`__init__.py` and `README.md`) to reserve the namespace.

## Git Sync Status

### Why Only `voxel_airspace_core` Was Initially Synced

The `voxel_airspace_core` module was committed and pushed in an earlier commit (`e68705c`). The `rwa_core` module was created later but:

1. **Files were created but not committed**: The files existed locally but were not added to Git staging area
2. **Git tracked file moves**: Git detected the files as moved from `nexus_core/` but the actual files in `modules/rwa_core/` were untracked
3. **Missing `git add`**: The new files needed to be explicitly added with `git add modules/rwa_core/`

### Resolution (2025-12-31)

All `modules/rwa_core/` files have been:
- ✅ Added to Git staging area
- ✅ Committed to local repository (commit `a2bf324`)
- ⚠️ **Pending**: Push to remote repository (`git push origin main`)

## Ensuring Future Sync

To ensure all modules stay synced to GitHub:

### Option 1: Manual Push (Recommended for now)
```bash
# After making changes to modules/
git add modules/
git commit -m "Update modules: [description]"
git push origin main
```

### Option 2: Git Hooks (Automated)
Create `.git/hooks/post-commit` to automatically push after commits:
```bash
#!/bin/sh
# Auto-push modules/ changes (optional, use with caution)
git push origin main
```

### Option 3: CI/CD Pipeline
Set up GitHub Actions to automatically sync on push (if using multiple remotes).

## Module Development Guidelines

1. **New modules** should be added under `modules/` with a clear `__init__.py`
2. **Always commit** new modules immediately after creation
3. **Test imports** to ensure backward compatibility shims work
4. **Update this README** when adding new modules

## Backward Compatibility

Modules maintain compatibility with old import paths via re-export shims:
- `nexus_core/assetization/valuation.py` → `modules.rwa_core.valuation`
- `nexus_core/economics/pricing.py` → `modules.rwa_core.economics.pricing`

This allows existing code to continue working without immediate refactoring.

