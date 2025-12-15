import React from 'react';
import { Aircraft, Route, NoFlyZone, Coordinate } from '../types';

interface MapProps {
  routes: Route[];
  aircraft: Aircraft[];
  noFlyZones?: NoFlyZone[];
  plannedPath?: Coordinate[];
  className?: string;
  showLabels?: boolean;
  onMapClick?: (coord: Coordinate) => void;
  onAircraftClick?: (aircraft: Aircraft) => void;
  selectedAircraftId?: string;
}

const MapVisualization: React.FC<MapProps> = ({ 
  routes, 
  aircraft, 
  noFlyZones = [], 
  plannedPath,
  className, 
  showLabels = true,
  onMapClick,
  onAircraftClick,
  selectedAircraftId
}) => {
  
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onMapClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    onMapClick({ x, y });
  };

  return (
    <div className={`relative bg-slate-100 rounded-xl overflow-hidden border border-slate-300 ${className}`}>
      {/* Grid Background */}
      <div className="absolute inset-0 pointer-events-none" style={{ 
        backgroundImage: 'radial-gradient(#cbd5e1 1px, transparent 1px)', 
        backgroundSize: '20px 20px' 
      }}></div>
      
      <svg className="relative z-10 w-full h-full cursor-crosshair" viewBox="0 0 100 100" preserveAspectRatio="none" onClick={handleSvgClick}>
        
        {/* No Fly Zones */}
        {noFlyZones.map(zone => (
          <g key={zone.id}>
            <circle cx={zone.x} cy={zone.y} r={zone.radius} fill="rgba(239, 68, 68, 0.2)" stroke="#ef4444" strokeWidth="0.5" strokeDasharray="2 1" />
            <text x={zone.x} y={zone.y} fontSize="2" fill="#ef4444" textAnchor="middle" opacity="0.7">RESTRICTED</text>
          </g>
        ))}

        {/* Standard Routes */}
        {routes.map((route) => (
          <g key={route.id}>
            <polyline
              points={route.coordinates.map(p => `${p.x},${p.y}`).join(' ')}
              fill="none"
              stroke={route.isCustom ? "#8b5cf6" : "#94a3b8"}
              strokeWidth={route.isCustom ? "1" : "0.5"}
              strokeDasharray={route.isCustom ? "none" : "2 1"}
            />
            {/* Start Point */}
            <circle cx={route.coordinates[0].x} cy={route.coordinates[0].y} r="1" fill="#3b82f6" />
            {/* End Point */}
            <circle cx={route.coordinates[route.coordinates.length - 1].x} cy={route.coordinates[route.coordinates.length - 1].y} r="1" fill="#ef4444" />
          </g>
        ))}

        {/* Dynamic Planned Path (Preview) */}
        {plannedPath && (
          <polyline
            points={plannedPath.map(p => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="#10b981"
            strokeWidth="1"
            strokeDasharray="1 1"
            className="animate-pulse"
          />
        )}

        {/* Aircraft */}
        {aircraft.map((ac) => {
            // Simple color coding based on status
            let color = '#22c55e'; // Available - Green
            if (ac.status === 'BUSY') color = '#3b82f6'; // Busy - Blue
            if (ac.status === 'MAINTENANCE') color = '#eab308'; // Maint - Yellow
            if (ac.status === 'OFFLINE') color = '#64748b'; // Offline - Gray
            if (ac.status === 'EMERGENCY') color = '#ef4444'; // Emergency - Red

            return (
              <g
                key={ac.id}
                onClick={(e) => {
                  if (!onAircraftClick) return;
                  e.stopPropagation(); // avoid triggering onMapClick
                  onAircraftClick(ac);
                }}
                style={{ cursor: onAircraftClick ? 'pointer' : 'default' }}
              >
                {/* Click hit target (invisible but large) */}
                <circle
                  cx={ac.currentLocation.x}
                  cy={ac.currentLocation.y}
                  r="4.5"
                  fill="transparent"
                  pointerEvents="all"
                />

                {/* Pulse effect for active aircraft */}
                {(ac.status === 'AVAILABLE' || ac.status === 'BUSY' || ac.status === 'EMERGENCY') && (
                   <circle cx={ac.currentLocation.x} cy={ac.currentLocation.y} r={ac.status === 'EMERGENCY' ? "5" : "3"} fill={color} opacity="0.2">
                      <animate attributeName="r" from="1" to={ac.status === 'EMERGENCY' ? "5" : "3"} dur={ac.status === 'EMERGENCY' ? "0.5s" : "1.5s"} repeatCount="indefinite" />
                      <animate attributeName="opacity" from="0.6" to="0" dur={ac.status === 'EMERGENCY' ? "0.5s" : "1.5s"} repeatCount="indefinite" />
                   </circle>
                )}

                {/* Selected highlight ring */}
                {selectedAircraftId && ac.id === selectedAircraftId && (
                  <circle
                    cx={ac.currentLocation.x}
                    cy={ac.currentLocation.y}
                    r="2.8"
                    fill="none"
                    stroke="#0f172a"
                    strokeWidth="0.5"
                    opacity="0.95"
                  />
                )}
                
                {/* Drone Icon Representation */}
                <circle cx={ac.currentLocation.x} cy={ac.currentLocation.y} r="1.5" fill={color} stroke="white" strokeWidth="0.2" />
                
                {showLabels && (
                  <text 
                    x={ac.currentLocation.x} 
                    y={ac.currentLocation.y - 2.5} 
                    fontSize="3" 
                    textAnchor="middle" 
                    fill="#1e293b"
                    fontWeight="bold"
                  >
                    {ac.regNumber}
                  </text>
                )}
              </g>
            );
        })}
      </svg>

      {/* Legend Overlay */}
      <div className="absolute bottom-2 left-2 bg-white/90 backdrop-blur px-2 py-1 rounded text-xs shadow-sm border border-slate-200">
        <div className="flex items-center gap-2 mb-1"><span className="w-2 h-2 rounded-full bg-green-500"></span> Available</div>
        <div className="flex items-center gap-2 mb-1"><span className="w-2 h-2 rounded-full bg-blue-500"></span> In Flight</div>
        <div className="flex items-center gap-2 mb-1"><span className="w-2 h-2 rounded-full bg-red-500"></span> Emergency/Restricted</div>
      </div>
    </div>
  );
};

export default MapVisualization;
