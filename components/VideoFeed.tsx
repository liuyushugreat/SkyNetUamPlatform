import React from 'react';
import { Aircraft } from '../types';
import { Scan, Target, AlertOctagon } from 'lucide-react';

interface VideoFeedProps {
  aircraft: Aircraft;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ aircraft }) => {
  return (
    <div className={`relative rounded-lg overflow-hidden bg-black aspect-video group shadow-lg border transition-colors ${aircraft.status === 'EMERGENCY' ? 'border-red-600 ring-2 ring-red-500/50' : 'border-slate-700'}`}>
      {/* Simulated Video Background */}
      <img 
        src={aircraft.videoFeedUrl} 
        alt="Drone Feed" 
        className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
      />
      
      {/* HUD Overlay */}
      <div className="absolute inset-0 p-4 flex flex-col justify-between pointer-events-none">
        {/* Top HUD */}
        <div className="flex justify-between items-start text-xs font-mono text-green-400">
            <div className="bg-black/50 p-2 rounded backdrop-blur-sm border border-green-900/50">
                <p>CAM-01 • 1080p • 60FPS</p>
                <p>LAT: {aircraft.currentLocation.x.toFixed(4)} LON: {aircraft.currentLocation.y.toFixed(4)}</p>
            </div>
            <div className="bg-black/50 p-2 rounded backdrop-blur-sm border border-green-900/50 flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full animate-pulse ${aircraft.status === 'EMERGENCY' ? 'bg-red-500' : 'bg-green-500'}`}></div> 
                {aircraft.status === 'EMERGENCY' ? 'ALERT' : 'LIVE'}
            </div>
        </div>

        {/* Center Reticle */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-50">
            {aircraft.status === 'EMERGENCY' ? (
                <AlertOctagon size={64} className="text-red-500 animate-ping" strokeWidth={1} />
            ) : (
                <Target size={48} className="text-white/80" strokeWidth={1} />
            )}
        </div>

        {/* Simulated AI Bounding Boxes */}
        {aircraft.status !== 'OFFLINE' && (
            <div className="absolute top-1/3 left-1/4 w-16 h-16 border border-yellow-400 rounded-sm opacity-70">
                 <span className="absolute -top-4 left-0 bg-yellow-400 text-black text-[8px] px-1 font-bold">OBSTACLE</span>
            </div>
        )}
        
        {/* Bottom HUD */}
        <div className="flex justify-between items-end text-green-400 text-xs font-mono">
             <div className="bg-black/50 p-2 rounded backdrop-blur-sm border border-green-900/50">
                <p>BAT: {aircraft.batteryLevel}%</p>
                <p>ALT: {aircraft.altitude?.toFixed(0) || 0}m</p>
                <p>SPD: {aircraft.speed?.toFixed(1) || 0}km/h</p>
             </div>
             <div className="flex items-center gap-2">
                {aircraft.status === 'EMERGENCY' ? (
                    <span className="text-red-500 font-bold animate-pulse">ERR: MALFUNCTION</span>
                ) : (
                    <>
                        <Scan className="animate-spin-slow" size={16} />
                        <span>AI SCANNING...</span>
                    </>
                )}
             </div>
        </div>
      </div>

      {/* Scanlines effect */}
      <div className="absolute inset-0 pointer-events-none" style={{
        background: 'linear-gradient(transparent 50%, rgba(0, 0, 0, 0.25) 50%)',
        backgroundSize: '100% 4px'
      }}></div>
    </div>
  );
};

export default VideoFeed;