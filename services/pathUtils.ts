import { Coordinate, NoFlyZone } from '../types';

// Simple Grid-based pathfinding (simulating complex auto-routing)
export const findOptimalPath = (start: Coordinate, end: Coordinate, noFlyZones: NoFlyZone[]): Coordinate[] => {
  const GRID_SIZE = 100; // 100x100 virtual grid
  
  // Check if point is in any no-fly zone
  const isInNoFlyZone = (x: number, y: number) => {
    return noFlyZones.some(zone => {
      const dx = x - zone.x;
      const dy = y - zone.y;
      return Math.sqrt(dx * dx + dy * dy) < zone.radius;
    });
  };

  // A* / BFS Simplified Implementation
  // For this demo, we will do a simplified direct path with waypoints if blocked
  // This simulates "Auto Planning" without heavy computation for the browser
  
  const path: Coordinate[] = [start];
  let current = { ...start };
  
  const steps = 20;
  const dx = (end.x - start.x) / steps;
  const dy = (end.y - start.y) / steps;
  
  // Naive straight line generation first
  for (let i = 1; i < steps; i++) {
    const nextX = start.x + dx * i;
    const nextY = start.y + dy * i;
    
    if (isInNoFlyZone(nextX, nextY)) {
      // If blocked, detour! 
      // Simple detour logic: try moving perpendicular to the blockage
      // In a real app, this would be A* or Dijkstra
      const detourX = nextX + (dy * 5); // Perpendicular shift
      const detourY = nextY - (dx * 5);
      
      if (!isInNoFlyZone(detourX, detourY)) {
         path.push({ x: detourX, y: detourY });
      } else {
         // Try other side
         path.push({ x: nextX - (dy * 5), y: nextY + (dx * 5) });
      }
    } else {
      path.push({ x: nextX, y: nextY });
    }
  }
  
  path.push(end);
  return path;
};
