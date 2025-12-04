import React, { useState, useEffect, useRef } from 'react';
import MapVisualization from '../components/MapVisualization';
import VideoFeed from '../components/VideoFeed';
import { ROUTES, AIRCRAFT, ALERTS, NO_FLY_ZONES } from '../services/mockData';
import { findOptimalPath } from '../services/pathUtils';
import { AlertTriangle, Shield, Database, Radio, Map as MapIcon, Video, Siren, Navigation, Zap, Activity, Home } from 'lucide-react';
import { playTextToSpeech } from '../services/geminiService';
import { Aircraft as AircraftType, Coordinate } from '../types';

interface RegulatorAppProps {
  onBackToHome: () => void;
}

const RegulatorApp: React.FC<RegulatorAppProps> = ({ onBackToHome }) => {
  // Simulate live data updates
  const [aircraftPos, setAircraftPos] = useState(AIRCRAFT);
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'monitor' | 'compliance' | 'planning'>('monitor');
  const [emergencyMode, setEmergencyMode] = useState(false);
  const [simulationEnabled, setSimulationEnabled] = useState(true); // New: Control random failure injection
  
  // Path Planning State
  const [planningStart, setPlanningStart] = useState<Coordinate | null>(null);
  const [planningEnd, setPlanningEnd] = useState<Coordinate | null>(null);
  const [generatedPath, setGeneratedPath] = useState<Coordinate[] | undefined>(undefined);

  // Enhanced simulation physics with random emergency trigger
  useEffect(() => {
    const interval = setInterval(() => {
        const now = Date.now();
        
        setAircraftPos(prev => {
            // Pre-calculate if we should trigger a new random emergency event
            // We only trigger one if:
            // 1. Simulation is enabled
            // 2. Global emergency is OFF
            // 3. No current active emergencies (keep it controlled)
            const activeEmergencies = prev.filter(a => a.status === 'EMERGENCY').length;
            let aircraftToTriggerId: string | null = null;

            if (simulationEnabled && !emergencyMode && activeEmergencies === 0) {
               // ~1.5% chance per tick (1Hz) to trigger an event
               if (Math.random() < 0.015) {
                  const healthyAssets = prev.filter(a => a.status === 'BUSY' || a.status === 'AVAILABLE');
                  if (healthyAssets.length > 0) {
                      const victim = healthyAssets[Math.floor(Math.random() * healthyAssets.length)];
                      aircraftToTriggerId = victim.id;
                  }
               }
            }

            return prev.map(ac => {
                // 1. Determine Status
                let newStatus = ac.status;

                if (emergencyMode) {
                     // Force ground/emergency state
                     newStatus = 'EMERGENCY';
                } else if (ac.id === aircraftToTriggerId) {
                     // Newly triggered random failure
                     newStatus = 'EMERGENCY';
                } else if (ac.status === 'EMERGENCY') {
                     // Check for resolution (Auto-recovery or Pilot fix)
                     // ~8% chance per tick -> avg duration ~12.5s (fits 10-15s requirement)
                     if (Math.random() < 0.08) {
                         newStatus = Math.random() > 0.5 ? 'BUSY' : 'AVAILABLE';
                     }
                }
                
                // Keep offline/maintenance unless forced by global emergency (which we handled above)
                if (!emergencyMode && (ac.status === 'OFFLINE' || ac.status === 'MAINTENANCE')) {
                    return ac;
                }

                // 2. Calculate Physics (Speed & Altitude with Enhanced Noise)
                
                // Generate unique time-based oscillation for each aircraft
                const timeOffset = parseInt(ac.id.replace(/\D/g, '')) * 1000;
                const time = now + timeOffset;
                
                // Dynamic Target Speed
                let targetSpeed = 0;
                let currentSpeed = (ac.speed || 0);

                if (newStatus === 'BUSY') {
                    // Highly dynamic flight profile
                    const baseSpeed = 150;
                    // Major surges (e.g. wind gusts or pilot throttling)
                    const surge = 35 * Math.sin(time / 3000); 
                    // Minor operational adjustments
                    const microSurge = 15 * Math.cos(time / 800); 
                    targetSpeed = baseSpeed + surge + microSurge;
                } else if (newStatus === 'AVAILABLE') {
                    targetSpeed = 60; // Loitering
                } else if (newStatus === 'EMERGENCY') {
                    targetSpeed = 0; // Emergency landing attempt
                }

                // Dynamic Target Altitude
                let targetAlt = 0;
                let currentAlt = (ac.altitude || 0);

                if (newStatus === 'BUSY') {
                    const baseAlt = 400;
                    // Large altitude corridors
                    const altVariation = 60 * Math.cos(time / 6000); 
                    // Sudden altitude drops/climbs
                    const altJitter = 20 * Math.sin(time / 1500);
                    targetAlt = baseAlt + altVariation + altJitter;
                } else if (newStatus === 'AVAILABLE') {
                    targetAlt = 150;
                } else if (newStatus === 'EMERGENCY') {
                    targetAlt = 0; // Going down
                }

                // 3. Apply Noise & Physics
                // Increased noise for "Busy" state to make it look more live/raw
                const noiseFactor = newStatus === 'BUSY' ? 1.0 : 0.3;
                const speedNoise = (Math.random() - 0.5) * 30 * noiseFactor; // +/- 15 km/h jitter
                const altNoise = (Math.random() - 0.5) * 15 * noiseFactor;   // +/- 7.5m jitter
                
                // Acceleration Physics
                const accelRate = newStatus === 'EMERGENCY' ? 25 : 8; // Brake/Drop fast in emergency
                
                if (currentSpeed < targetSpeed) currentSpeed += Math.min(accelRate, targetSpeed - currentSpeed);
                else if (currentSpeed > targetSpeed) currentSpeed -= Math.min(accelRate, currentSpeed - targetSpeed);
                
                // Apply jitter
                currentSpeed += speedNoise;
                currentSpeed = Math.max(0, currentSpeed);

                // Altitude Physics
                const climbRate = newStatus === 'EMERGENCY' ? 20 : 6; 
                
                if (currentAlt < targetAlt) currentAlt += Math.min(climbRate, targetAlt - currentAlt);
                else if (currentAlt > targetAlt) currentAlt -= Math.min(climbRate, currentAlt - targetAlt);
                
                // Apply jitter
                currentAlt += altNoise;
                currentAlt = Math.max(0, currentAlt);

                // 4. Update Position (Orbit simulation)
                let newX = ac.currentLocation.x;
                let newY = ac.currentLocation.y;

                if (newStatus === 'BUSY' && currentSpeed > 10) {
                     // Simulate flying in a route/orbit around a center point
                     const centerX = 50;
                     const centerY = 50;
                     const dx = newX - centerX;
                     const dy = newY - centerY;
                     const radius = Math.sqrt(dx*dx + dy*dy) || 20;
                     const currentAngle = Math.atan2(dy, dx);
                     
                     // Angular velocity based on speed
                     const angularSpeed = (currentSpeed / 1200) * (Math.random() * 0.1 + 0.95); 
                     const nextAngle = currentAngle + angularSpeed;

                     newX = centerX + Math.cos(nextAngle) * radius;
                     newY = centerY + Math.sin(nextAngle) * radius;
                } else {
                    // Hover drift
                    const drift = newStatus === 'EMERGENCY' ? 0.0 : 0.1;
                    newX += (Math.random() - 0.5) * drift;
                    newY += (Math.random() - 0.5) * drift;
                }

                // Boundary check
                newX = Math.max(5, Math.min(95, newX));
                newY = Math.max(5, Math.min(95, newY));

                return {
                    ...ac,
                    status: newStatus,
                    speed: currentSpeed,
                    altitude: currentAlt,
                    currentLocation: { x: newX, y: newY }
                };
            });
        });
    }, 1000); // 1Hz update
    return () => clearInterval(interval);
  }, [emergencyMode, simulationEnabled]);

  // Watch for new emergencies to announce
  const prevStatusRef = useRef<Record<string, string>>({});
  useEffect(() => {
    aircraftPos.forEach(ac => {
        const prev = prevStatusRef.current[ac.id];
        
        // New Emergency Alert
        if (ac.status === 'EMERGENCY' && prev !== 'EMERGENCY' && !emergencyMode) {
            playTextToSpeech(`Alert. Anomaly detected on ${ac.regNumber}. Emergency status active.`);
        }

        // Resolution Alert
        if (prev === 'EMERGENCY' && ac.status !== 'EMERGENCY' && !emergencyMode) {
            playTextToSpeech(`Update. ${ac.regNumber} has recovered. Resuming flight plan.`);
        }

        prevStatusRef.current[ac.id] = ac.status;
    });
  }, [aircraftPos, emergencyMode]);

  const toggleEmergency = () => {
      const newMode = !emergencyMode;
      setEmergencyMode(newMode);
      if (newMode) {
          playTextToSpeech("Emergency Protocol Initiated. All aircraft grounded. Airspace closed.");
      } else {
          playTextToSpeech("Emergency Protocol Lifted. Normal operations resuming.");
      }
  };

  const handleMapClick = (coord: Coordinate) => {
      if (activeTab !== 'planning') return;
      
      if (!planningStart) {
          setPlanningStart(coord);
      } else if (!planningEnd) {
          setPlanningEnd(coord);
          // Auto calculate
          const path = findOptimalPath(planningStart, coord, NO_FLY_ZONES);
          setGeneratedPath(path);
          playTextToSpeech("Optimal path generated avoiding restricted zones.");
      } else {
          // Reset
          setPlanningStart(coord);
          setPlanningEnd(null);
          setGeneratedPath(undefined);
      }
  };

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden font-sans">
      
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col shadow-2xl z-20">
        <div className="p-6 border-b border-slate-800 flex items-center gap-3">
            <Shield className="text-blue-500" size={28} />
            <div>
                <h1 className="font-bold text-white leading-none tracking-wide">SKYGUARD</h1>
                <p className="text-xs text-slate-500 uppercase tracking-widest mt-1">Regulatory Command</p>
            </div>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
            <button 
                onClick={onBackToHome}
                className="w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-colors hover:bg-slate-800 text-slate-400 hover:text-white"
            >
                <Home size={18} /> Return Home
            </button>
            
            <div className="my-2 border-b border-slate-800"></div>

            <button 
                onClick={() => setActiveTab('monitor')}
                className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-colors ${activeTab === 'monitor' ? 'bg-slate-800 text-white border-l-4 border-blue-500' : 'hover:bg-slate-800'}`}
            >
                <Radio size={18} /> Live Monitoring
            </button>
            <button 
                 onClick={() => setActiveTab('planning')}
                 className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-colors ${activeTab === 'planning' ? 'bg-slate-800 text-white border-l-4 border-blue-500' : 'hover:bg-slate-800'}`}
            >
                <Navigation size={18} /> Airspace Planning
            </button>
            <button 
                 onClick={() => setActiveTab('compliance')}
                 className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-colors ${activeTab === 'compliance' ? 'bg-slate-800 text-white border-l-4 border-blue-500' : 'hover:bg-slate-800'}`}
            >
                <AlertTriangle size={18} /> Compliance & Alerts
            </button>
        </nav>

        {/* Simulation Controls */}
        <div className="px-4 py-2">
            <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-400 flex items-center gap-1"><Zap size={12} /> STRESS TEST</span>
                    <div className={`w-2 h-2 rounded-full ${simulationEnabled ? 'bg-amber-500 animate-pulse' : 'bg-slate-600'}`}></div>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500">Random Failures</span>
                    <button 
                        onClick={() => setSimulationEnabled(!simulationEnabled)}
                        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${simulationEnabled ? 'bg-blue-600' : 'bg-slate-600'}`}
                    >
                        <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${simulationEnabled ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                </div>
            </div>
        </div>

        {/* Emergency Button */}
        <div className="p-4">
            <button 
                onClick={toggleEmergency}
                className={`w-full py-3 rounded-lg font-bold text-white shadow-lg flex items-center justify-center gap-2 transition-all ${emergencyMode ? 'bg-red-600 animate-pulse' : 'bg-slate-700 hover:bg-red-900'}`}
            >
                <Siren size={20} /> {emergencyMode ? 'CANCEL EMERGENCY' : 'EMERGENCY STOP'}
            </button>
        </div>

        <div className="p-4 border-t border-slate-800">
            <div className="flex items-center gap-2 text-xs text-slate-500">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Secure Connection Active
            </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative">
        
        {/* Top Bar */}
        <header className={`border-b h-16 px-6 flex justify-between items-center shadow-sm z-10 transition-colors ${emergencyMode ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}>
            <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                {emergencyMode && <AlertTriangle className="text-red-600 animate-pulse" />}
                {activeTab === 'monitor' && 'Real-time Airspace Monitor (City Zone A)'}
                {activeTab === 'planning' && 'Automated Path Planning Simulation'}
                {activeTab === 'compliance' && 'Compliance Auditing Log'}
            </h2>
            <div className="flex gap-4">
                {simulationEnabled && activeTab === 'monitor' && (
                    <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 text-amber-700 px-3 py-1 rounded-full text-xs font-bold animate-pulse">
                        <Activity size={14} /> SIMULATION ACTIVE
                    </div>
                )}
                <div className="text-right">
                    <p className="text-xs text-slate-500">Active Flights</p>
                    <p className="font-bold text-slate-800 text-lg leading-none">3</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-slate-500">Violations (24h)</p>
                    <p className="font-bold text-red-600 text-lg leading-none">1</p>
                </div>
            </div>
        </header>

        {/* Map & Panels Container */}
        <div className="flex-1 overflow-hidden flex">
            
            {/* Main Workspace */}
            <div className="flex-1 p-6 relative bg-slate-200 overflow-y-auto">
                
                {activeTab === 'monitor' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                        {/* Map takes up 2 columns */}
                        <div className={`lg:col-span-2 bg-white rounded-2xl shadow-xl overflow-hidden h-96 lg:h-full border relative ${emergencyMode ? 'border-red-400' : 'border-slate-300'}`}>
                            <MapVisualization 
                                routes={ROUTES} 
                                aircraft={aircraftPos} 
                                noFlyZones={NO_FLY_ZONES}
                                className="h-full w-full" 
                            />
                             {emergencyMode && (
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-red-500/10">
                                    <div className="bg-red-600 text-white px-6 py-3 rounded-xl font-bold text-2xl animate-bounce shadow-lg border-2 border-white">
                                        AIRSPACE CLOSED - GROUNDING ALL ASSETS
                                    </div>
                                </div>
                             )}
                        </div>
                        
                        {/* Video Feeds Column */}
                        <div className="space-y-4 flex flex-col h-full overflow-y-auto pr-2">
                            <h3 className="font-semibold text-slate-700 flex items-center gap-2 sticky top-0 bg-slate-200 py-2 z-10">
                                <Video size={18} /> AI Surveillance
                            </h3>
                            {aircraftPos.map(ac => (
                                <div key={ac.id} className="transform transition-all duration-300">
                                    <div className="flex justify-between items-end mb-1">
                                        <p className="text-xs font-semibold text-slate-500">{ac.regNumber} ({ac.model})</p>
                                        {ac.status === 'EMERGENCY' && <span className="text-[10px] bg-red-600 text-white px-2 rounded animate-pulse">CRITICAL FAULT</span>}
                                    </div>
                                    <VideoFeed aircraft={ac} />
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'planning' && (
                    <div className="h-full flex flex-col">
                         <div className="bg-white p-4 rounded-t-2xl border-b border-slate-200 flex justify-between items-center">
                             <div className="text-sm text-slate-600">
                                 <p>Click two points on the map to generate an optimal path avoiding <span className="text-red-500 font-bold">No-Fly Zones</span>.</p>
                             </div>
                             <div className="flex gap-4 text-xs font-mono">
                                 <div>START: {planningStart ? `${planningStart.x.toFixed(1)}, ${planningStart.y.toFixed(1)}` : '--'}</div>
                                 <div>END: {planningEnd ? `${planningEnd.x.toFixed(1)}, ${planningEnd.y.toFixed(1)}` : '--'}</div>
                             </div>
                         </div>
                         <div className="flex-1 bg-white rounded-b-2xl shadow-xl overflow-hidden border border-slate-300">
                            <MapVisualization 
                                routes={[]} 
                                aircraft={[]} 
                                noFlyZones={NO_FLY_ZONES} 
                                plannedPath={generatedPath}
                                className="h-full w-full" 
                                onMapClick={handleMapClick}
                            />
                         </div>
                    </div>
                )}

                {activeTab === 'compliance' && (
                    <div className="bg-white rounded-2xl shadow-xl p-6 h-full overflow-y-auto">
                        <h3 className="font-bold text-lg mb-4 text-slate-800">Compliance Audit Log</h3>
                        <table className="w-full text-sm">
                            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 text-left">
                                <tr>
                                    <th className="p-3">Timestamp</th>
                                    <th className="p-3">Severity</th>
                                    <th className="p-3">Message</th>
                                    <th className="p-3">Aircraft ID</th>
                                    <th className="p-3">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {ALERTS.map(alert => (
                                    <tr key={alert.id} className="border-b border-slate-100 hover:bg-slate-50">
                                        <td className="p-3 font-mono text-slate-600">{new Date(alert.timestamp).toLocaleTimeString()}</td>
                                        <td className="p-3">
                                            <span className={`px-2 py-1 rounded text-xs font-bold text-white ${
                                                alert.severity === 'CRITICAL' ? 'bg-red-600' :
                                                alert.severity === 'MEDIUM' ? 'bg-amber-500' : 'bg-blue-500'
                                            }`}>
                                                {alert.severity}
                                            </span>
                                        </td>
                                        <td className="p-3 font-medium text-slate-800">{alert.message}</td>
                                        <td className="p-3 font-mono text-slate-500">{alert.aircraftId}</td>
                                        <td className="p-3">{alert.status}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
      </main>
    </div>
  );
};

export default RegulatorApp;