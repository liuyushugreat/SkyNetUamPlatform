import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import MapVisualization from '../MapVisualization';
import ValueChainChart from './ValueChainChart';
import AssetLogStream from './AssetLogStream';
import AssetPopup from './AssetPopup';
import { Aircraft } from '../../types';
import { AIRCRAFT, NO_FLY_ZONES } from '../../services/mockData';
import { Home, ArrowLeft, BarChart3, Globe, Zap, Database } from 'lucide-react';
import FullScreenButton from '../FullScreenButton';

interface AssetInsightDashboardProps {
    onBackToHome?: () => void;
    filterOperator?: string; // Optional: Show only aircraft from this operator
    className?: string;
}

// --- Simulation Helpers (Ported from RegulatorApp) ---
function mulberry32(a: number) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
}

function randomBetween(rng: () => number, min: number, max: number) {
    return min + rng() * (max - min);
}

function generateSyntheticFleet(count: number, seed = 1337): Aircraft[] {
    const rng = mulberry32(seed);
    const models = ['DJI Matrice 30', 'Skydio X10', 'Autel EVO Max', 'Wingcopter 198', 'Zipline P2'];
    const operators = ['MetroOps', 'AeroGrid', 'CityLogistics', 'SkyCourier', 'UrbanMed'];
  
    const fleet: Aircraft[] = [];
    for (let i = 0; i < count; i++) {
      // Keep positions mostly inside the map bounds and outside restricted circles
      let x = 50;
      let y = 50;
      for (let tries = 0; tries < 25; tries++) {
        const candidate = { x: randomBetween(rng, 6, 94), y: randomBetween(rng, 6, 94) };
        const isInsideRestricted = NO_FLY_ZONES.some((z) => {
          const dx = candidate.x - z.x;
          const dy = candidate.y - z.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          return dist < z.radius + 2;
        });
        if (!isInsideRestricted) {
          x = candidate.x;
          y = candidate.y;
          break;
        }
      }
  
      // Bias towards BUSY to make the airspace look active
      const p = rng();
      const status: Aircraft['status'] = p < 0.78 ? 'BUSY' : p < 0.98 ? 'AVAILABLE' : 'MAINTENANCE';
      const regNumber = `UAM-ZA-${String(i + 1).padStart(3, '0')}`;
  
      fleet.push({
        id: `sim-${String(i + 1).padStart(3, '0')}`,
        model: models[Math.floor(rng() * models.length)],
        regNumber,
        status,
        operator: operators[Math.floor(rng() * operators.length)],
        batteryLevel: Math.floor(randomBetween(rng, 35, 100)),
        currentLocation: { x, y },
        speed: status === 'BUSY' ? randomBetween(rng, 110, 210) : status === 'AVAILABLE' ? randomBetween(rng, 30, 80) : 0,
        altitude: status === 'BUSY' ? randomBetween(rng, 280, 520) : status === 'AVAILABLE' ? randomBetween(rng, 80, 200) : 0
      });
    }
  
    return fleet;
}

// Generate some fake history data for the popup
const generateFakeHistory = () => Array.from({ length: 20 }, () => Math.random() * 10 + 5);

const AssetInsightDashboard: React.FC<AssetInsightDashboardProps> = ({ onBackToHome, filterOperator, className = "h-screen" }) => {
    const [selectedAircraft, setSelectedAircraft] = useState<Aircraft | null>(null);
    const [popupData, setPopupData] = useState<any>(null);
    
    // Initialize with mock data + 100 synthetic drones
    const [aircraftList, setAircraftList] = useState<Aircraft[]>(() => {
        const baseFleet = [
            ...AIRCRAFT,
            ...generateSyntheticFleet(100)
        ];
        return filterOperator 
            ? baseFleet.filter(ac => ac.operator === filterOperator)
            : baseFleet;
    });

    const [stats, setStats] = useState({
        totalValue: 125432.50,
        tps: 1245,
        nodes: aircraftList.length,
        minted: 892
    });

    // Re-filter if props change
    useEffect(() => {
        const baseFleet = [
            ...AIRCRAFT,
            ...generateSyntheticFleet(100)
        ];
        const filtered = filterOperator 
            ? baseFleet.filter(ac => ac.operator === filterOperator)
            : baseFleet;
        
        setAircraftList(filtered);
    }, [filterOperator]);

    // Mock real-time data updates for the popup (High Frequency for selected)
    useEffect(() => {
        if (selectedAircraft) {
            // Initial data
            setPopupData((prev: any) => {
                // Preserve previous data if switching back to same aircraft, otherwise reset
                // For simplicity, always reset here or use a cache
                return {
                    value: Math.random() * 5 + 10,
                    accumulated: Math.random() * 500 + 100,
                    integrity: 98.5 + (Math.random() - 0.5),
                    history: generateFakeHistory()
                };
            });

            // Simulate updates
            const interval = setInterval(() => {
                setPopupData((prev: any) => {
                   if (!prev) return null;
                   const newValue = Math.max(0, prev.value + (Math.random() - 0.5) * 2);
                   const newHistory = [...prev.history.slice(1), newValue];
                   return {
                       ...prev,
                       value: newValue,
                       accumulated: prev.accumulated + newValue * 0.1,
                       history: newHistory
                   };
                });
            }, 1000);

            return () => clearInterval(interval);
        }
    }, [selectedAircraft?.id]); // Depend on ID to reset when switching

    // Mock Fleet Movement Simulation (Low Frequency for all)
    useEffect(() => {
        const interval = setInterval(() => {
            setAircraftList(prevList => {
                return prevList.map(ac => {
                    // Only move BUSY or AVAILABLE aircraft
                    if (ac.status !== 'BUSY' && ac.status !== 'AVAILABLE') return ac;

                    // Simple random walk
                    let dx = (Math.random() - 0.5) * 0.4;
                    let dy = (Math.random() - 0.5) * 0.4;
                    
                    let newX = Math.max(0, Math.min(100, ac.currentLocation.x + dx));
                    let newY = Math.max(0, Math.min(100, ac.currentLocation.y + dy));

                    return {
                        ...ac,
                        currentLocation: { x: newX, y: newY }
                    };
                });
            });
            
            // Update selected aircraft reference if it moves
            if (selectedAircraft) {
                // We rely on the map finding the aircraft by ID, but for the popup positioning to update 
                // smoothly with the map, we might need to sync `selectedAircraft` state. 
                // However, MapVisualization usually takes the list and handles display.
                // The popup uses `selectedAircraft.currentLocation`, which is stale in `selectedAircraft` state
                // unless we update it.
                
                // Let's find the updated version of the selected aircraft
                // But doing this inside setState callback is tricky.
                // We'll leave `selectedAircraft` stale for position, and let the Map click handler update it?
                // No, the popup needs live position.
            }

        }, 500); // 2fps for background movement

        return () => clearInterval(interval);
    }, []);

    // Sync selectedAircraft position with the list to keep popup attached
    useEffect(() => {
        if (!selectedAircraft) return;
        const current = aircraftList.find(a => a.id === selectedAircraft.id);
        if (current && (current.currentLocation.x !== selectedAircraft.currentLocation.x || current.currentLocation.y !== selectedAircraft.currentLocation.y)) {
             setSelectedAircraft(current);
        }
    }, [aircraftList, selectedAircraft?.id]);


    // Mock Global Stats Ticker
    useEffect(() => {
        const interval = setInterval(() => {
            setStats(prev => ({
                totalValue: prev.totalValue + (Math.random() * 100),
                tps: Math.floor(1200 + Math.random() * 100),
                nodes: prev.nodes, // Keep constant for now
                minted: prev.minted + (Math.random() > 0.5 ? 1 : 0)
            }));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    const handleAircraftClick = (aircraft: Aircraft) => {
        // Toggle: if clicking same one, deselect
        if (selectedAircraft?.id === aircraft.id) {
            setSelectedAircraft(null);
        } else {
            setSelectedAircraft(aircraft);
        }
    };

    return (
        <div className={`flex flex-col bg-slate-900 text-white overflow-hidden ${className}`}>
            
            {/* 1. Global HUD Header (Command Center Style) */}
            <header className="h-16 bg-slate-950/80 backdrop-blur border-b border-slate-800 flex items-center justify-between px-6 shrink-0 z-50">
                <div className="flex items-center gap-6">
                    {onBackToHome && (
                        <button onClick={onBackToHome} className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors">
                            <ArrowLeft size={20} />
                        </button>
                    )}
                    <h1 className="text-xl font-bold flex items-center gap-3">
                        <Database className="text-blue-500" />
                        <span className="tracking-wider">SKYNET <span className="text-slate-500">ASSET INSIGHT</span></span>
                        {filterOperator && (
                            <span className="text-xs bg-indigo-900 text-indigo-300 px-2 py-0.5 rounded border border-indigo-700 font-mono tracking-normal ml-2">
                                FILTER: {filterOperator.toUpperCase()}
                            </span>
                        )}
                    </h1>
                </div>

                {/* Ticker Stats */}
                <div className="flex items-center gap-8 font-mono text-sm">
                    <div className="flex flex-col items-end">
                        <span className="text-xs text-slate-500 uppercase">Total Asset Value (TVL)</span>
                        <span className="text-green-400 font-bold text-lg">${stats.totalValue.toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                    </div>
                    <div className="hidden md:flex flex-col items-end">
                        <span className="text-xs text-slate-500 uppercase">Real-time TPS</span>
                        <span className="text-blue-400 font-bold">{stats.tps} ops/sec</span>
                    </div>
                    <div className="hidden md:flex flex-col items-end">
                        <span className="text-xs text-slate-500 uppercase">Minted Assets</span>
                        <span className="text-purple-400 font-bold">{stats.minted} NFTs</span>
                    </div>
                     <div className="hidden md:flex flex-col items-end">
                        <span className="text-xs text-slate-500 uppercase">Active Nodes</span>
                        <span className="text-orange-400 font-bold">{stats.nodes} / {aircraftList.length}</span>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <FullScreenButton className="text-slate-400 hover:text-white" />
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        <span className="text-xs text-slate-400 uppercase tracking-widest">System Online</span>
                    </div>
                </div>
            </header>

            {/* 2. Main Content Area */}
            <main className="flex-1 flex gap-4 p-4 overflow-hidden relative">
                
                {/* Left Panel: Value Chain Topology & Logs */}
                {/* CHANGED: Fixed width for better map visibility on large screens */}
                <div className="w-80 xl:w-96 flex flex-col gap-4 z-10 shrink-0">
                    {/* Top Left: Topology */}
                    <div className="bg-slate-800/80 backdrop-blur rounded-xl p-4 border border-slate-700 h-2/5 shadow-lg flex flex-col">
                        <h3 className="text-sm font-bold mb-2 flex items-center gap-2 shrink-0 text-slate-300 uppercase tracking-wide">
                            <Globe size={16} className="text-indigo-500" />
                            Data Value Chain
                        </h3>
                        <div className="flex-1 min-h-0 relative">
                             {/* Decorative Grid */}
                            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none"></div>
                            <ValueChainChart />
                        </div>
                    </div>
                    
                    {/* Bottom Left: Asset Logs */}
                    <div className="bg-slate-800/80 backdrop-blur rounded-xl p-4 border border-slate-700 h-3/5 shadow-lg overflow-hidden flex flex-col">
                         <h3 className="text-sm font-bold mb-2 flex items-center gap-2 shrink-0 text-slate-300 uppercase tracking-wide">
                            <Zap size={16} className="text-emerald-500" />
                            Asset Stream
                        </h3>
                        <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">
                            <AssetLogStream />
                        </div>
                    </div>
                </div>

                {/* Right Panel: Map & Real-time Assets */}
                {/* CHANGED: Takes remaining space (flex-1) */}
                <div className="flex-1 bg-slate-800/50 rounded-xl border border-slate-700 shadow-2xl relative overflow-hidden group">
                     {/* Map Header Overlay */}
                     <div className="absolute top-4 left-4 z-20 pointer-events-none">
                        <div className="bg-slate-900/90 backdrop-blur px-4 py-2 rounded-lg border border-slate-600 shadow-xl">
                            <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                <BarChart3 size={18} className="text-blue-400"/> Live Fleet Assets
                            </h3>
                            <p className="text-xs text-slate-400">Monitoring {aircraftList.length} RWA Nodes {filterOperator ? `(${filterOperator})` : ''}</p>
                        </div>
                     </div>

                     {/* Map Container */}
                     <div className="w-full h-full relative">
                         <MapVisualization 
                            routes={[]} 
                            aircraft={aircraftList}
                            className="w-full h-full bg-slate-900"
                            onAircraftClick={handleAircraftClick}
                            selectedAircraftId={selectedAircraft?.id}
                         />
                         
                         {/* Popup Overlay - Positioned using % relative to map container */}
                         {selectedAircraft && popupData && (
                             <div 
                                style={{ 
                                    position: 'absolute', 
                                    left: `${selectedAircraft.currentLocation.x}%`, 
                                    top: `${selectedAircraft.currentLocation.y}%`, 
                                    transform: 'translate(-50%, -115%)', 
                                    zIndex: 50,
                                    transition: 'left 0.5s linear, top 0.5s linear' // Smooth movement
                                }}
                             >
                                 <AssetPopup 
                                    uavId={selectedAircraft.id}
                                    position={{ x: 0, y: 0 }} // Relative to the wrapper div
                                    data={popupData}
                                    onClose={() => setSelectedAircraft(null)}
                                 />
                             </div>
                         )}
                     </div>
                </div>
            </main>
        </div>
    );
};

export default AssetInsightDashboard;
