# Voxel Airspace Core

Sparse Octree implementation for efficient 3D spatial indexing of airspace occupancy.

## Overview

This module implements a sparse octree data structure for managing 3D airspace occupancy. The octree recursively subdivides 3D space into eight octants, enabling efficient insertion and query operations for bounding boxes and points.

## Key Features

- **Sparse Representation**: Only nodes that intersect with occupied regions are created
- **Efficient Queries**: O(log N) point-in-voxel queries
- **Memory Efficient**: Significantly reduces memory usage compared to dense voxel grids
- **Configurable Resolution**: Adjustable maximum depth for fine-grained control

## Usage

```python
from modules.voxel_airspace_core import SparseOctree

# Create an octree covering 1000m x 1000m x 1000m airspace
# with maximum depth of 8 (minimum voxel size = 1000 / 2^8 ≈ 3.9m)
octree = SparseOctree(root_size=1000.0, max_depth=8, origin=(0, 0, 0))

# Insert an obstacle bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
octree.insert((100, 200, 50, 150, 250, 100))

# Query if a point is occupied
is_occupied = octree.query((120, 220, 75))  # Returns True

# Get all occupied voxels
occupied_voxels = octree.get_occupied_voxels()
# Returns: [(x1, y1, z1, size1), (x2, y2, z2, size2), ...]

# Get statistics
stats = octree.get_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Occupied nodes: {stats['occupied_nodes']}")
```

## Mathematical Foundation

### Octree Structure

Each node represents a cubic voxel in 3D space:
- **Root node**: Largest voxel covering the entire airspace
- **Internal nodes**: Subdivided into 8 child octants
- **Leaf nodes**: Smallest voxels (at max_depth)

### Subdivision Logic

For a node at depth d with center (cx, cy, cz) and size s:
- Child node size: `s_child = s / 2`
- Child node centers: `(cx ± s/4, cy ± s/4, cz ± s/4)`
- Creates 8 octants: (+++, ++-, +-+, +--, -++, -+-, --+, ---)

### Query Complexity

- Best case: O(log(root_size / min_size))
- Space complexity: O(N) where N is the number of occupied voxels (sparse)

## API Reference

### `VoxelNode`

Represents a single voxel in the octree.

**Attributes:**
- `x, y, z`: Center coordinates (float)
- `size`: Edge length (float)
- `is_occupied`: Occupancy status (bool)
- `children`: List of 8 child nodes or None (List[VoxelNode] | None)

**Methods:**
- `subdivide()`: Split node into 8 child octants
- `get_bounds()`: Get AABB as (min_x, min_y, min_z, max_x, max_y, max_z)

### `SparseOctree`

Main octree class for spatial indexing.

**Methods:**
- `insert(bbox)`: Insert a bounding box and mark intersecting voxels
- `query(point)`: Check if a point is occupied
- `get_occupied_voxels()`: Get all occupied voxels
- `get_statistics()`: Get tree statistics

## Performance Considerations

- **Memory**: Sparse representation only creates nodes for occupied regions
- **Query Speed**: Logarithmic in tree depth
- **Insertion Speed**: O(depth × intersections) per bounding box
- **Recommended max_depth**: 8-12 for most airspace applications

