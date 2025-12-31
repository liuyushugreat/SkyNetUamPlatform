import React, { useState, useEffect, useRef } from 'react';
import MapVisualization from '../../../components/MapVisualization';
import VideoFeed from '../../../components/VideoFeed';
import { ROUTES, AIRCRAFT, ALERTS, NO_FLY_ZONES } from '../../../services/mockData';
import { findOptimalPath } from '../../../services/pathUtils';
import { AlertTriangle, Shield, Database, Radio, Map as MapIcon, Video, Siren, Navigation, Zap, Activity, Home, LogOut } from 'lucide-react';
import { playTextToSpeech } from '../../../services/geminiService';
import { Aircraft as AircraftType, Coordinate } from '../../../types';
import FullScreenButton from '../../../components/FullScreenButton';

interface RegulatorAppProps {
  onBackToHome: () => void;
}

// Deterministic PRNG for reproducible "busy airspace" visuals
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomBetween(rng: () => number, min: number, max: number) {
  return min + (max - min) * rng();
}

function svgToDataUrl(svg: string) {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function makeSyntheticVideoFrame(regNumber: string, seed: number) {
  const rng = mulberry32(seed);

  // Daytime-only virtual city viewpoints (different "angles"/landmarks)
  const views = [
    { name: 'Downtown Highrise', skyA: '#7dd3fc', skyB: '#e0f2fe', accent: '#0ea5e9' },
    { name: 'Riverfront', skyA: '#93c5fd', skyB: '#eff6ff', accent: '#2563eb' },
    { name: 'Harbor District', skyA: '#67e8f9', skyB: '#ecfeff', accent: '#06b6d4' },
    { name: 'Business Avenue', skyA: '#a5b4fc', skyB: '#f5f3ff', accent: '#6366f1' },
    { name: 'Park & Suburb', skyA: '#86efac', skyB: '#f0fdf4', accent: '#22c55e' },
    { name: 'Industrial Yard', skyA: '#fde68a', skyB: '#fff7ed', accent: '#f59e0b' },
    { name: 'Bridge Approach', skyA: '#bae6fd', skyB: '#f8fafc', accent: '#0284c7' },
    { name: 'Stadium Quarter', skyA: '#fecaca', skyB: '#fff1f2', accent: '#ef4444' }
  ];

  const v = views[Math.floor(rng() * views.length)];
  const w = 800;
  const h = 450;

  // "Camera" parameters (angle + horizon + vanishing point) for different city angles
  const tiltDeg = randomBetween(rng, -3.5, 3.5);
  const horizonY = Math.floor(randomBetween(rng, h * 0.46, h * 0.68));
  const vanishingX = Math.floor(randomBetween(rng, w * 0.35, w * 0.65));

  const boxCount = 2 + Math.floor(rng() * 4);
  const boxes = Array.from({ length: boxCount }).map(() => {
    const x = Math.floor(randomBetween(rng, 60, w - 180));
    const y = Math.floor(randomBetween(rng, 70, h - 160));
    const bw = Math.floor(randomBetween(rng, 60, 160));
    const bh = Math.floor(randomBetween(rng, 50, 140));
    const label = rng() > 0.5 ? 'OBST' : 'VEH';
    return { x, y, bw, bh, label };
  });

  const skylineCount = 10 + Math.floor(rng() * 14);
  const skyline = Array.from({ length: skylineCount }).map((_, i) => {
    const x = Math.floor((i / skylineCount) * w);
    const bw = Math.floor(randomBetween(rng, 30, 120));
    const bh = Math.floor(randomBetween(rng, 40, 240));
    const y = Math.max(34, horizonY - bh);
    return { x, y, bw, bh };
  });

  // Sun + clouds
  const sunX = Math.floor(randomBetween(rng, 80, w - 120));
  const sunY = Math.floor(randomBetween(rng, 55, Math.max(70, horizonY - 90)));
  const sunR = Math.floor(randomBetween(rng, 18, 34));

  const cloudCount = 3 + Math.floor(rng() * 6);
  const clouds = Array.from({ length: cloudCount }).map(() => {
    const cx = Math.floor(randomBetween(rng, 40, w - 60));
    const cy = Math.floor(randomBetween(rng, 40, Math.max(70, horizonY - 40)));
    const r = Math.floor(randomBetween(rng, 14, 32));
    return { cx, cy, r };
  });

  // Ground features (road / river / park) for different scenes
  const groundVariant = Math.floor(rng() * 3);
  const roadWidth = Math.floor(randomBetween(rng, 40, 70));
  const roadX = Math.floor(randomBetween(rng, w * 0.15, w * 0.7));

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${v.skyA}"/>
      <stop offset="100%" stop-color="${v.skyB}"/>
    </linearGradient>
    <filter id="noise">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch" />
      <feColorMatrix type="matrix" values="
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 .05 0"/>
    </filter>
    <linearGradient id="haze" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <!-- Daytime sky -->
  <rect width="${w}" height="${h}" fill="url(#bg)"/>
  <circle cx="${sunX}" cy="${sunY}" r="${sunR}" fill="rgba(253,224,71,0.95)"/>
  <circle cx="${sunX}" cy="${sunY}" r="${sunR + 22}" fill="rgba(253,224,71,0.18)"/>
  ${clouds
    .map(
      (c) => `
  <g opacity="0.78">
    <circle cx="${c.cx}" cy="${c.cy}" r="${c.r}" fill="rgba(255,255,255,0.95)"/>
    <circle cx="${c.cx + Math.floor(c.r * 0.9)}" cy="${c.cy + 4}" r="${Math.floor(c.r * 0.85)}" fill="rgba(255,255,255,0.92)"/>
    <circle cx="${c.cx - Math.floor(c.r * 0.8)}" cy="${c.cy + 6}" r="${Math.floor(c.r * 0.75)}" fill="rgba(255,255,255,0.90)"/>
  </g>
`
    )
    .join('\n')}

  <rect width="${w}" height="${h}" filter="url(#noise)"/>
  <rect width="${w}" height="${h}" fill="url(#haze)"/>

  <!-- Scene group (tilt gives different city angles) -->
  <g transform="rotate(${tiltDeg.toFixed(2)} ${Math.floor(w / 2)} ${Math.floor(h / 2)})">
    <!-- Ground -->
    <rect y="${horizonY}" width="${w}" height="${h - horizonY}" fill="rgba(148,163,184,0.50)"/>
    <rect y="${horizonY}" width="${w}" height="${h - horizonY}" fill="rgba(15,23,42,0.10)"/>

    ${
      groundVariant === 0
        ? `
    <!-- Road -->
    <rect x="${roadX}" y="${horizonY}" width="${roadWidth}" height="${h - horizonY}" fill="rgba(51,65,85,0.30)"/>
    <line x1="${roadX + Math.floor(roadWidth / 2)}" y1="${horizonY}" x2="${vanishingX}" y2="${horizonY - 60}" stroke="rgba(255,255,255,0.28)" stroke-width="3"/>
    <line x1="${roadX + Math.floor(roadWidth / 2)}" y1="${horizonY + 40}" x2="${vanishingX}" y2="${horizonY - 60}" stroke="rgba(255,255,255,0.12)" stroke-width="2"/>
`
        : groundVariant === 1
          ? `
    <!-- River -->
    <path d="M ${Math.floor(w * 0.08)} ${h} C ${Math.floor(w * 0.25)} ${Math.floor(h * 0.78)}, ${Math.floor(w * 0.55)} ${Math.floor(h * 0.72)}, ${Math.floor(w * 0.82)} ${h}" fill="rgba(37,99,235,0.20)"/>
    <path d="M ${Math.floor(w * 0.10)} ${h} C ${Math.floor(w * 0.28)} ${Math.floor(h * 0.80)}, ${Math.floor(w * 0.56)} ${Math.floor(h * 0.74)}, ${Math.floor(w * 0.84)} ${h}" fill="rgba(59,130,246,0.12)"/>
`
          : `
    <!-- Park -->
    <rect x="${Math.floor(w * 0.08)}" y="${horizonY + 22}" width="${Math.floor(w * 0.30)}" height="${Math.floor(h * 0.22)}" rx="18" fill="rgba(34,197,94,0.20)"/>
    <circle cx="${Math.floor(w * 0.22)}" cy="${horizonY + 70}" r="18" fill="rgba(34,197,94,0.16)"/>
    <circle cx="${Math.floor(w * 0.16)}" cy="${horizonY + 110}" r="14" fill="rgba(34,197,94,0.14)"/>
`
    }

    <!-- Skyline -->
    ${skyline
      .map(
        (b) =>
          `<rect x="${b.x}" y="${b.y}" width="${b.bw}" height="${Math.floor(horizonY - b.y + randomBetween(rng, 80, 220))}" fill="rgba(15,23,42,0.26)"/>`
      )
      .join('\n')}
  </g>

  <!-- HUD (brighter for daytime) -->
  <rect x="14" y="14" width="320" height="66" rx="10" fill="rgba(255,255,255,0.60)" stroke="rgba(2,132,199,0.30)"/>
  <text x="30" y="40" fill="${v.accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="16">CAM • ${v.name}</text>
  <text x="30" y="62" fill="#0f172a" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">${regNumber} • 1080p</text>

  <!-- Center reticle -->
  <circle cx="${Math.floor(w / 2)}" cy="${Math.floor(h / 2)}" r="42" fill="none" stroke="rgba(15,23,42,0.24)" stroke-width="2"/>
  <line x1="${Math.floor(w / 2) - 60}" y1="${Math.floor(h / 2)}" x2="${Math.floor(w / 2) + 60}" y2="${Math.floor(h / 2)}" stroke="rgba(15,23,42,0.16)" stroke-width="2"/>
  <line x1="${Math.floor(w / 2)}" y1="${Math.floor(h / 2) - 60}" x2="${Math.floor(w / 2)}" y2="${Math.floor(h / 2) + 60}" stroke="rgba(15,23,42,0.16)" stroke-width="2"/>

  <!-- AI boxes -->
  ${boxes
    .map(
      (b) => `
    <rect x="${b.x}" y="${b.y}" width="${b.bw}" height="${b.bh}" fill="none" stroke="rgba(250,204,21,0.85)" stroke-width="3"/>
    <rect x="${b.x}" y="${b.y - 22}" width="70" height="18" fill="rgba(250,204,21,0.85)"/>
    <text x="${b.x + 6}" y="${b.y - 9}" fill="#111827" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="12" font-weight="700">${b.label}</text>
  `
    )
    .join('\n')}

  <!-- Bottom telemetry strip -->
  <rect x="14" y="${h - 68}" width="${w - 28}" height="54" rx="10" fill="rgba(255,255,255,0.60)" stroke="rgba(2,132,199,0.20)"/>
  <text x="30" y="${h - 40}" fill="#0f172a" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">CITY ZONE A • LIVE</text>
  <text x="30" y="${h - 22}" fill="#334155" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Uplink OK • AI tracking engaged</text>
</svg>
`.trim();

  return svgToDataUrl(svg);
}

function generateSyntheticFleet(count: number, seed = 1337): AircraftType[] {
  const rng = mulberry32(seed);
  const models = ['DJI Matrice 30', 'Skydio X10', 'Autel EVO Max', 'Wingcopter 198', 'Zipline P2'];
  const operators = ['MetroOps', 'AeroGrid', 'CityLogistics', 'SkyCourier', 'UrbanMed'];

  const fleet: AircraftType[] = [];
  for (let i = 0; i < count; i++) {
    // Keep positions mostly inside the map bounds and outside restricted circles (simple retry)
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
    const status: AircraftType['status'] = p < 0.78 ? 'BUSY' : p < 0.98 ? 'AVAILABLE' : 'MAINTENANCE';

    const regNumber = `UAM-ZA-${String(i + 1).padStart(3, '0')}`;

    fleet.push({
      id: `sim-${String(i + 1).padStart(3, '0')}`,
      model: models[Math.floor(rng() * models.length)],
      regNumber,
      status,
      operator: operators[Math.floor(rng() * operators.length)],
      batteryLevel: Math.floor(randomBetween(rng, 35, 100)),
      currentLocation: { x, y },
      // Unique per-aircraft "scene" (offline, deterministic)
      videoFeedUrl: makeSyntheticVideoFrame(regNumber, seed * 1000 + i + 1),
      speed: status === 'BUSY' ? randomBetween(rng, 110, 210) : status === 'AVAILABLE' ? randomBetween(rng, 30, 80) : 0,
      altitude: status === 'BUSY' ? randomBetween(rng, 280, 520) : status === 'AVAILABLE' ? randomBetween(rng, 80, 200) : 0
    });
  }

  return fleet;
}


const RegulatorApp: React.FC<RegulatorAppProps> = ({ onBackToHome }) => {
  // Simulate live data updates
  const [aircraftPos, setAircraftPos] = useState<AircraftType[]>(() => [
    // Apply daytime virtual-city scenes to the original aircraft as well (keep regNumber stable)
    ...AIRCRAFT.map((a, idx) => ({
      ...a,
      videoFeedUrl: makeSyntheticVideoFrame(a.regNumber, 910000 + idx)
    })),
    ...generateSyntheticFleet(100)
  ]);
  const [selectedAircraftId, setSelectedAircraftId] = useState<string>(() => AIRCRAFT[0]?.id || 'sim-001');
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'monitor' | 'compliance' | 'planning'>('monitor');
  const [emergencyMode, setEmergencyMode] = useState(false);
  const [simulationEnabled, setSimulationEnabled] = useState(true); // New: Control random failure injection
  const [isMonitorHoverPaused, setIsMonitorHoverPaused] = useState(false);
  const [selectionFlashNonce, setSelectionFlashNonce] = useState(0);
  const flashTimeoutRef = useRef<number | null>(null);

  const playSelectBeep = () => {
    // User-gesture initiated click -> safe to start/resume AudioContext
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx: AudioContext = new AudioCtx();
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }

      const now = ctx.currentTime;

      const beep = (start: number, duration: number, freq: number) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, start);

        // Quick envelope to avoid clicking artifacts
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(0.14, start + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(start);
        osc.stop(start + duration + 0.02);
      };

      // "Doot-doot" confirmation (2 short beeps)
      beep(now + 0.00, 0.07, 880);
      beep(now + 0.10, 0.07, 660);

      // Close context shortly after to free resources
      window.setTimeout(() => {
        ctx.close().catch(() => {});
      }, 400);
    } catch {
      // ignore audio errors
    }
  };

  const triggerSelectionFlash = () => {
    setSelectionFlashNonce((n) => n + 1);
    if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = window.setTimeout(() => {
      // no-op; key remount handles the one-shot animation
    }, 250);
  };

  useEffect(() => {
    return () => {
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    };
  }, []);
  
  // Path Planning State
  const [planningStart, setPlanningStart] = useState<Coordinate | null>(null);
  const [planningEnd, setPlanningEnd] = useState<Coordinate | null>(null);
  const [generatedPath, setGeneratedPath] = useState<Coordinate[] | undefined>(undefined);

  // Enhanced simulation physics with random emergency trigger
  useEffect(() => {
    const interval = setInterval(() => {
        const now = Date.now();
        
        setAircraftPos(prev => {
            // Pause motion when hovering over the monitor map (freeze the scene)
            if (isMonitorHoverPaused) return prev;

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
  }, [emergencyMode, simulationEnabled, isMonitorHoverPaused]);

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

  const activeFlightsCount = aircraftPos.filter(
    (a) => a.status === 'AVAILABLE' || a.status === 'BUSY' || a.status === 'EMERGENCY'
  ).length;

  const selectedAircraft =
    aircraftPos.find((a) => a.id === selectedAircraftId) || aircraftPos[0] || null;

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
           <div className="flex gap-2">
               <FullScreenButton className="flex-1 bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white" />
               <button 
                onClick={onBackToHome}
                className="flex-1 flex items-center justify-center gap-2 p-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all border border-slate-700 hover:border-slate-600 group"
               >
                <Home size={18} /> Home
               </button>
           </div>
            
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
                    <p className="font-bold text-slate-800 text-lg leading-none">{activeFlightsCount}</p>
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
                        <div
                          className={`lg:col-span-2 bg-white rounded-2xl shadow-xl overflow-hidden h-96 lg:h-full border relative ${emergencyMode ? 'border-red-400' : 'border-slate-300'}`}
                          onMouseEnter={() => setIsMonitorHoverPaused(true)}
                          onMouseLeave={() => setIsMonitorHoverPaused(false)}
                        >
                            <style>{`
                              @keyframes skynetFlashOnce {
                                0% { opacity: 0; }
                                15% { opacity: 0.85; }
                                100% { opacity: 0; }
                              }
                            `}</style>
                            <MapVisualization 
                                routes={ROUTES} 
                                aircraft={aircraftPos} 
                                noFlyZones={NO_FLY_ZONES}
                                showLabels={false}
                                selectedAircraftId={selectedAircraftId}
                                onAircraftClick={(ac) => {
                                  setSelectedAircraftId(ac.id);
                                  playSelectBeep();
                                  triggerSelectionFlash();
                                }}
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
                            {selectedAircraft ? (
                              <div className="transform transition-all duration-300">
                                <div className="flex justify-between items-end mb-1">
                                  <p className="text-xs font-semibold text-slate-500">
                                    {selectedAircraft.regNumber} ({selectedAircraft.model})
                                  </p>
                                  {selectedAircraft.status === 'EMERGENCY' && (
                                    <span className="text-[10px] bg-red-600 text-white px-2 rounded animate-pulse">
                                      CRITICAL FAULT
                                    </span>
                                  )}
                                </div>
                                <div className="mb-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-slate-600 bg-white/60 border border-slate-200 rounded-lg p-2">
                                  <div><span className="font-semibold text-slate-700">ID:</span> {selectedAircraft.id}</div>
                                  <div><span className="font-semibold text-slate-700">Status:</span> {selectedAircraft.status}</div>
                                  <div><span className="font-semibold text-slate-700">Operator:</span> {selectedAircraft.operator}</div>
                                  <div><span className="font-semibold text-slate-700">Battery:</span> {selectedAircraft.batteryLevel.toFixed(0)}%</div>
                                  <div><span className="font-semibold text-slate-700">Alt:</span> {(selectedAircraft.altitude ?? 0).toFixed(0)}m</div>
                                  <div><span className="font-semibold text-slate-700">Spd:</span> {(selectedAircraft.speed ?? 0).toFixed(1)}km/h</div>
                                </div>
                                <div key={selectionFlashNonce} className="relative rounded-lg overflow-hidden">
                                  <VideoFeed aircraft={selectedAircraft} />
                                  <div
                                    className="absolute inset-0 pointer-events-none bg-white"
                                    style={{ opacity: 0, animation: 'skynetFlashOnce 220ms ease-out 1' }}
                                  />
                                </div>
                                <div className="mt-2 text-[11px] text-slate-500">
                                  Tip: click any aircraft dot on the map to inspect it here.
                                </div>
                              </div>
                            ) : (
                              <div className="text-sm text-slate-500">
                                Click an aircraft dot on the map to view its surveillance feed.
                              </div>
                            )}
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