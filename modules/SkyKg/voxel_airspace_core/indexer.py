"""
Sparse Octree Indexer for Voxel-Based Airspace Management.

This module implements a sparse octree data structure for efficient 3D spatial indexing
of airspace occupancy. The octree recursively subdivides 3D space into eight octants,
enabling efficient insertion and query operations for bounding boxes and points.

Key Concepts:
-------------
1. **Octree Structure**: Each node represents a cubic voxel in 3D space.
   - Root node: Largest voxel covering the entire airspace
   - Internal nodes: Subdivided into 8 child octants
   - Leaf nodes: Smallest voxels (at max_depth)

2. **Sparse Representation**: Only nodes that intersect with occupied regions are created,
   significantly reducing memory usage compared to dense voxel grids.

3. **Recursive Subdivision**: When a bounding box partially overlaps a node, the node
   is subdivided into 8 children, and the insertion continues recursively.

Mathematical Foundation:
------------------------
For a node at depth d with center (cx, cy, cz) and size s:
- Child node size: s_child = s / 2
- Child node centers are at: (cx ± s/4, cy ± s/4, cz ± s/4)
- This creates 8 octants: (+++, ++-, +-+, +--, -++, -+-, --+, ---)

Reference:
----------
Samet, H. (1989). "The Design and Analysis of Spatial Data Structures".
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class VoxelNode:
    """
    A node in the sparse octree representing a cubic voxel in 3D space.
    
    Attributes:
    -----------
    x, y, z : float
        Center coordinates of the voxel in 3D space.
    size : float
        Edge length of the cubic voxel.
    is_occupied : bool
        Whether this voxel (or any of its descendants) is occupied.
        True if the voxel intersects with any inserted bounding box.
    children : Optional[List[VoxelNode]]
        List of 8 child nodes (octants). None if this is a leaf node.
        Order: [+++, ++-, +-+, +--, -++, -+-, --+, ---]
        where +/- indicates positive/negative offset in (x, y, z) respectively.
    """
    x: float
    y: float
    z: float
    size: float
    is_occupied: bool = False
    children: Optional[List['VoxelNode']] = None
    
    def subdivide(self) -> List['VoxelNode']:
        """
        Subdivide this node into 8 child octants.
        
        The subdivision follows the standard octree convention:
        - Each child has half the size of the parent
        - Child centers are offset by ±size/4 from parent center
        - Creates 8 octants covering the entire parent volume
        
        Mathematical Details:
        --------------------
        For a parent node with center (cx, cy, cz) and size s:
        - Child size: s_child = s / 2
        - Child centers: (cx ± s/4, cy ± s/4, cz ± s/4)
        
        The 8 children are ordered as:
        0: (cx + s/4, cy + s/4, cz + s/4)  # +x, +y, +z
        1: (cx + s/4, cy + s/4, cz - s/4)  # +x, +y, -z
        2: (cx + s/4, cy - s/4, cz + s/4)  # +x, -y, +z
        3: (cx + s/4, cy - s/4, cz - s/4)  # +x, -y, -z
        4: (cx - s/4, cy + s/4, cz + s/4)  # -x, +y, +z
        5: (cx - s/4, cy + s/4, cz - s/4)  # -x, +y, -z
        6: (cx - s/4, cy - s/4, cz + s/4)  # -x, -y, +z
        7: (cx - s/4, cy - s/4, cz - s/4)  # -x, -y, -z
        
        Returns:
        --------
        List[VoxelNode]
            List of 8 child nodes, each representing one octant.
        """
        child_size = self.size / 2.0
        offset = self.size / 4.0
        
        # Pre-compute all 8 child centers using NumPy for efficiency
        # This avoids repeated floating-point operations
        centers = np.array([
            [self.x + offset, self.y + offset, self.z + offset],  # 0: +++
            [self.x + offset, self.y + offset, self.z - offset],  # 1: ++-
            [self.x + offset, self.y - offset, self.z + offset],  # 2: +-+
            [self.x + offset, self.y - offset, self.z - offset],  # 3: +--
            [self.x - offset, self.y + offset, self.z + offset],  # 4: -++
            [self.x - offset, self.y + offset, self.z - offset],  # 5: -+-
            [self.x - offset, self.y - offset, self.z + offset],  # 6: --+
            [self.x - offset, self.y - offset, self.z - offset],  # 7: ---
        ])
        
        # Create 8 child nodes
        children = [
            VoxelNode(
                x=centers[i, 0],
                y=centers[i, 1],
                z=centers[i, 2],
                size=child_size,
                is_occupied=False
            )
            for i in range(8)
        ]
        
        return children
    
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the axis-aligned bounding box (AABB) of this voxel.
        
        Returns:
        --------
        Tuple[float, float, float, float, float, float]
            (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        half_size = self.size / 2.0
        return (
            self.x - half_size,  # min_x
            self.y - half_size,  # min_y
            self.z - half_size,  # min_z
            self.x + half_size,  # max_x
            self.y + half_size,  # max_y
            self.z + half_size   # max_z
        )


class SparseOctree:
    """
    Sparse Octree for efficient 3D spatial indexing of airspace occupancy.
    
    This implementation uses a sparse representation where only nodes that
    intersect with occupied regions are created, making it memory-efficient
    for large airspaces with sparse obstacles.
    
    Attributes:
    -----------
    root : VoxelNode
        Root node of the octree, covering the entire airspace.
    max_depth : int
        Maximum depth of the octree. Limits the minimum voxel size.
        At depth d, voxel size = root_size / (2^d)
    origin : Tuple[float, float, float]
        Origin point (x, y, z) of the airspace coordinate system.
        The root node is centered at this origin.
    
    Mathematical Properties:
    -----------------------
    - Root node size determines the spatial extent: covers [-size/2, +size/2] in each axis
    - Depth d has 8^d potential nodes (but only occupied ones are created)
    - Minimum voxel size = root_size / (2^max_depth)
    - Query complexity: O(log(root_size / min_size)) in best case
    """
    
    def __init__(self, 
                 root_size: float,
                 max_depth: int = 10,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize a sparse octree.
        
        Parameters:
        -----------
        root_size : float
            Edge length of the root voxel. The octree covers the cube
            [origin - root_size/2, origin + root_size/2] in each axis.
        max_depth : int
            Maximum depth of the octree. Higher depth allows finer resolution
            but increases memory usage. Default: 10 (allows 2^10 = 1024x subdivision).
        origin : Tuple[float, float, float]
            Origin point (x, y, z) of the coordinate system. Default: (0, 0, 0).
        """
        self.max_depth = max_depth
        self.origin = np.array(origin, dtype=np.float64)
        
        # Create root node centered at origin
        self.root = VoxelNode(
            x=float(self.origin[0]),
            y=float(self.origin[1]),
            z=float(self.origin[2]),
            size=root_size,
            is_occupied=False
        )
    
    def insert(self, bbox: Union[Tuple, np.ndarray], current_node: Optional[VoxelNode] = None, depth: int = 0) -> None:
        """
        Insert a bounding box into the octree, marking intersecting voxels as occupied.
        
        This method recursively subdivides nodes that partially intersect with the
        bounding box, creating a sparse representation of the occupied space.
        
        Algorithm:
        ----------
        1. Check if bounding box intersects with current node
        2. If fully contained: mark node as occupied (leaf)
        3. If partially intersecting and depth < max_depth:
           a. Subdivide node into 8 children
           b. Recursively insert into each child that intersects
        4. If partially intersecting and depth == max_depth:
           Mark node as occupied (cannot subdivide further)
        
        Parameters:
        -----------
        bbox : Union[Tuple, np.ndarray]
            Bounding box as (min_x, min_y, min_z, max_x, max_y, max_z)
            or numpy array of shape (6,).
        current_node : Optional[VoxelNode]
            Current node being processed. None for root node (internal use).
        depth : int
            Current depth in the tree. 0 for root (internal use).
        
        Mathematical Details:
        --------------------
        Intersection test uses axis-aligned bounding box (AABB) overlap:
        - Two AABBs overlap if: max(min_a, min_b) < min(max_a, max_b) for all axes
        - Full coverage (for early-stop): min_bbox <= min_node AND max_bbox >= max_node for all axes
        
        Example:
        --------
        >>> octree = SparseOctree(root_size=1000.0, max_depth=8)
        >>> # Insert an obstacle bounding box
        >>> octree.insert((100, 200, 50, 150, 250, 100))
        >>> # Check if a point is occupied
        >>> octree.query((120, 220, 75))  # Returns True
        """
        if current_node is None:
            current_node = self.root
        
        # Convert bbox to numpy array for efficient computation
        bbox = np.array(bbox, dtype=np.float64)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        
        # Get node bounds
        node_min_x, node_min_y, node_min_z, node_max_x, node_max_y, node_max_z = current_node.get_bounds()
        
        # Check if bounding box intersects with node
        # Two AABBs intersect if they overlap on all three axes
        intersects = (
            max(min_x, node_min_x) < min(max_x, node_max_x) and
            max(min_y, node_min_y) < min(max_y, node_max_y) and
            max(min_z, node_min_z) < min(max_z, node_max_z)
        )
        
        if not intersects:
            return  # No intersection, nothing to do
        
        # Check if the *node* is fully covered by the bounding box.
        #
        # IMPORTANT (common pitfall):
        # - "bbox fully contained in node" is almost always true at the root, which would
        #   incorrectly mark the entire airspace as occupied after inserting the first object.
        # - What we want here is the opposite: if the bounding box fully covers the node,
        #   then the whole voxel is occupied and we can stop subdividing.
        node_fully_inside_bbox = (
            node_min_x >= min_x and node_max_x <= max_x and
            node_min_y >= min_y and node_max_y <= max_y and
            node_min_z >= min_z and node_max_z <= max_z
        )
        
        # If the node is fully covered by the bbox, we can mark it occupied and prune.
        if node_fully_inside_bbox:
            current_node.is_occupied = True
            current_node.children = None
            return
        
        # If we are at max depth, we cannot subdivide further: conservatively mark occupied.
        if depth >= self.max_depth:
            current_node.is_occupied = True
            current_node.children = None
            return
        
        # Partially intersecting and can subdivide: create children and recurse
        if current_node.children is None:
            current_node.children = current_node.subdivide()
        
        # Recursively insert into each child that intersects
        for child in current_node.children:
            self.insert(bbox, child, depth + 1)
        
        # After recursion, mark current node as occupied if any child is occupied.
        current_node.is_occupied = any(child.is_occupied for child in current_node.children)

        # Optional compression: if all 8 children are occupied, collapse them into this node.
        # This keeps the tree sparse and avoids unnecessary depth in fully-filled regions.
        if current_node.is_occupied and all(child.is_occupied and child.children is None for child in current_node.children):
            current_node.children = None
    
    def query(self, point: Union[Tuple, np.ndarray], current_node: Optional[VoxelNode] = None) -> bool:
        """
        Query whether a point is occupied (intersects with any inserted bounding box).
        
        This method traverses the octree from root to leaf, following the path
        that contains the query point. The traversal is O(log N) where N is the
        number of subdivisions.
        
        Algorithm:
        ----------
        1. Check if point is within current node bounds
        2. If not: return False (point is outside airspace)
        3. If node is a leaf (no children):
           - Return node.is_occupied
        4. If node has children:
           - Determine which child octant contains the point
           - Recursively query that child
        
        Parameters:
        -----------
        point : Union[Tuple, np.ndarray]
            Query point as (x, y, z) or numpy array of shape (3,).
        current_node : Optional[VoxelNode]
            Current node being processed. None for root node (internal use).
        
        Returns:
        --------
        bool
            True if the point is occupied (intersects with any inserted bbox),
            False otherwise.
        
        Mathematical Details:
        --------------------
        Point-in-voxel test: |point - center| <= size/2 for all axes
        Child selection: Compare point coordinates with node center to determine octant:
        - x >= center_x ? right half (0-3) : left half (4-7)
        - y >= center_y ? top half (0,1,4,5) : bottom half (2,3,6,7)
        - z >= center_z ? front half (0,2,4,6) : back half (1,3,5,7)
        
        Example:
        --------
        >>> octree = SparseOctree(root_size=1000.0, max_depth=8)
        >>> octree.insert((100, 200, 50, 150, 250, 100))
        >>> octree.query((120, 220, 75))  # Returns True
        >>> octree.query((50, 50, 50))    # Returns False
        """
        if current_node is None:
            current_node = self.root
        
        # Convert point to numpy array
        point = np.array(point, dtype=np.float64)
        px, py, pz = point
        
        # Get node bounds
        node_min_x, node_min_y, node_min_z, node_max_x, node_max_y, node_max_z = current_node.get_bounds()
        
        # Check if point is within node bounds
        if not (node_min_x <= px <= node_max_x and
                node_min_y <= py <= node_max_y and
                node_min_z <= pz <= node_max_z):
            return False  # Point is outside this node (and thus outside airspace)
        
        # If leaf node, return occupancy status
        if current_node.children is None:
            return current_node.is_occupied
        
        # Determine which child octant contains the point
        # This uses the same logic as subdivision: compare with center
        child_index = 0
        if px < current_node.x:
            child_index += 4  # Left half (children 4-7)
        if py < current_node.y:
            child_index += 2  # Bottom half (children 2,3,6,7)
        if pz < current_node.z:
            child_index += 1  # Back half (children 1,3,5,7)
        
        # Recursively query the appropriate child
        return self.query(point, current_node.children[child_index])
    
    def get_occupied_voxels(self, 
                            current_node: Optional[VoxelNode] = None,
                            depth: int = 0,
                            result: Optional[List[Tuple[float, float, float, float]]] = None) -> List[Tuple[float, float, float, float]]:
        """
        Get all occupied voxels as a list of (x, y, z, size) tuples.
        
        This is useful for visualization or exporting the occupancy map.
        
        Parameters:
        -----------
        current_node : Optional[VoxelNode]
            Current node being processed. None for root node (internal use).
        depth : int
            Current depth in the tree. 0 for root (internal use).
        result : Optional[List]
            Accumulator list for results (internal use).
        
        Returns:
        --------
        List[Tuple[float, float, float, float]]
            List of occupied voxels as (x, y, z, size) tuples.
        """
        if current_node is None:
            current_node = self.root
            result = []
        
        if current_node.is_occupied:
            if current_node.children is None:
                # Leaf node: add to results
                result.append((current_node.x, current_node.y, current_node.z, current_node.size))
            else:
                # Internal node: recursively check children
                for child in current_node.children:
                    self.get_occupied_voxels(child, depth + 1, result)
        
        return result
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the octree structure.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'total_nodes': Total number of nodes created
            - 'leaf_nodes': Number of leaf nodes
            - 'internal_nodes': Number of internal nodes
            - 'occupied_nodes': Number of occupied nodes
            - 'max_depth_reached': Maximum depth actually used
        """
        stats = {
            'total_nodes': 0,
            'leaf_nodes': 0,
            'internal_nodes': 0,
            'occupied_nodes': 0,
            'max_depth_reached': 0
        }
        
        def traverse(node: VoxelNode, depth: int):
            stats['total_nodes'] += 1
            stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)
            
            if node.is_occupied:
                stats['occupied_nodes'] += 1
            
            if node.children is None:
                stats['leaf_nodes'] += 1
            else:
                stats['internal_nodes'] += 1
                for child in node.children:
                    traverse(child, depth + 1)
        
        traverse(self.root, 0)
        return stats

