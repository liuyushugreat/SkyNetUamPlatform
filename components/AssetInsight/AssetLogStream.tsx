import React, { useEffect, useState, useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Shield, Database, Coins, FileCheck, Cpu } from 'lucide-react';

// --- Types ---
interface TelemetryPacket {
  id: string;
  uavId: string;
  timestamp: number;
  value: number;
  status: 'COLLECTED' | 'SIGNED' | 'VALUED' | 'MINTED';
  metadata?: {
      quality: number;
      scarcity: string;
  };
}

// --- Component ---
const AssetLogStream: React.FC = () => {
  const [packets, setPackets] = useState<TelemetryPacket[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [priceHistory, setPriceHistory] = useState<{time: string, value: number}[]>([]);
  const [mintedAssets, setMintedAssets] = useState<TelemetryPacket[]>([]);
  
  // Mock Data Stream
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString();
      
      // 1. Create new packet
      const newPacket: TelemetryPacket = {
        id: Math.random().toString(36).substr(2, 9),
        // Randomly pick a UAV ID from 1 to 100
        uavId: `UAV-${Math.floor(Math.random() * 100) + 1}`,
        timestamp: Date.now(),
        value: 0, // Init value
        status: 'COLLECTED',
        metadata: {
            quality: 0.8 + Math.random() * 0.2, // 0.8 - 1.0
            scarcity: Math.random() > 0.7 ? 'HIGH' : 'NORMAL'
        }
      };

      // Update Stream
      setPackets(prev => [newPacket, ...prev].slice(0, 6)); // Keep last 6

      // 2. Simulate Processing Pipeline (Async)
      processPacket(newPacket);

      // Update Chart Data (Simulating Global Asset Value Index)
      setPriceHistory(prev => {
          const newVal = (prev.length > 0 ? prev[prev.length-1].value : 10) + (Math.random() - 0.4);
          return [...prev, { time: timeStr, value: Math.max(0, newVal) }].slice(-20);
      });

    }, 2000); // Every 2 seconds

    return () => clearInterval(interval);
  }, []);

  const processPacket = async (packet: TelemetryPacket) => {
      // Step 1: Sign (Identity)
      await new Promise(r => setTimeout(r, 500));
      updatePacketStatus(packet.id, 'SIGNED');

      // Step 2: Value (Valuation)
      await new Promise(r => setTimeout(r, 800));
      // Calculate Value based on mock Pricing Engine logic
      const basePrice = 1.0;
      const scarcityMult = packet.metadata?.scarcity === 'HIGH' ? 2.0 : 1.0;
      const val = basePrice * (packet.metadata?.quality || 1) * scarcityMult;
      
      packet.value = val;
      setTotalValue(prev => prev + val);
      updatePacketStatus(packet.id, 'VALUED');

      // Step 3: Mint (Tokenization)
      if (Math.random() > 0.3) { // Not all become NFTs immediately
        await new Promise(r => setTimeout(r, 1000));
        updatePacketStatus(packet.id, 'MINTED');
        setMintedAssets(prev => [packet, ...prev].slice(0, 3)); // Show last 3 minted
      }
  };

  const updatePacketStatus = (id: string, status: TelemetryPacket['status']) => {
      setPackets(prev => prev.map(p => p.id === id ? { ...p, status } : p));
  };

  // --- Chart Config ---
  const chartOption = {
    grid: { top: 30, right: 20, bottom: 20, left: 40, containLabel: true },
    tooltip: { trigger: 'axis' },
    xAxis: { 
        type: 'category', 
        data: priceHistory.map(p => p.time),
        axisLine: { lineStyle: { color: '#94a3b8' } }
    },
    yAxis: { 
        type: 'value',
        splitLine: { lineStyle: { color: '#334155' } },
        axisLine: { lineStyle: { color: '#94a3b8' } }
    },
    series: [{
      data: priceHistory.map(p => p.value),
      type: 'line',
      smooth: true,
      areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(59, 130, 246, 0.5)' }, { offset: 1, color: 'rgba(59, 130, 246, 0.0)' }]
          }
      },
      lineStyle: { color: '#3b82f6', width: 3 },
      showSymbol: false,
    }]
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full text-slate-200">
        
        {/* Left: Data Pipeline Visualization */}
        <div className="lg:col-span-2 bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-xl">
            <h3 className="flex items-center gap-2 text-lg font-bold mb-6 text-white">
                <Activity className="text-blue-400" /> Data Assetization Pipeline
            </h3>
            
            <div className="space-y-4">
                <div className="flex justify-between text-xs text-slate-500 uppercase tracking-wider px-4">
                    <span>Telemetry In</span>
                    <span>DID Signing (TEE)</span>
                    <span>Valuation Engine</span>
                    <span>RWA Minting</span>
                </div>
                
                {/* Visual Pipeline Tracks */}
                <div className="relative h-64 border-l border-slate-700 ml-4 space-y-2 overflow-hidden">
                    <AnimatePresence>
                        {packets.map((packet) => (
                            <motion.div 
                                key={packet.id}
                                initial={{ x: -50, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                exit={{ x: 50, opacity: 0 }}
                                transition={{ type: 'spring' }}
                                className="flex items-center gap-4 bg-slate-900/80 p-3 rounded-lg border border-slate-700 hover:border-blue-500/50 transition-colors"
                            >
                                {/* Stage 1: Collection */}
                                <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-slate-400">
                                    <Cpu size={16} />
                                </div>
                                <div className="w-12 h-0.5 bg-slate-700"></div>

                                {/* Stage 2: Signing */}
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${['SIGNED', 'VALUED', 'MINTED'].includes(packet.status) ? 'bg-purple-900 text-purple-400' : 'bg-slate-800 text-slate-600'}`}>
                                    <Shield size={16} />
                                </div>
                                <div className={`w-12 h-0.5 ${['SIGNED', 'VALUED', 'MINTED'].includes(packet.status) ? 'bg-purple-900' : 'bg-slate-700'}`}></div>

                                {/* Stage 3: Valuation */}
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${['VALUED', 'MINTED'].includes(packet.status) ? 'bg-green-900 text-green-400' : 'bg-slate-800 text-slate-600'}`}>
                                    <Coins size={16} />
                                </div>
                                <div className={`w-12 h-0.5 ${['VALUED', 'MINTED'].includes(packet.status) ? 'bg-green-900' : 'bg-slate-700'}`}></div>

                                {/* Stage 4: Minting */}
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${packet.status === 'MINTED' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50' : 'bg-slate-800 text-slate-600'}`}>
                                    <Database size={16} />
                                </div>

                                {/* Info Card */}
                                <div className="ml-auto flex items-center gap-4 text-xs font-mono">
                                    <span className="text-slate-400">{packet.uavId}</span>
                                    {packet.value > 0 && <span className="text-green-400 font-bold">+${packet.value.toFixed(2)}</span>}
                                    <span className={`px-2 py-0.5 rounded ${packet.status === 'MINTED' ? 'bg-blue-500/20 text-blue-300' : 'bg-slate-800'}`}>
                                        {packet.status}
                                    </span>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            </div>
        </div>

        {/* Right: Metrics & Certificates */}
        <div className="space-y-6">
            
            {/* Real-time Value Chart */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-xl h-64">
                 <h3 className="text-sm font-bold text-slate-400 mb-2 uppercase">Global Data Asset Index</h3>
                 <div className="h-full -ml-4">
                    <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
                 </div>
            </div>

            {/* Latest Minted Assets */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-xl flex-1">
                <h3 className="flex items-center gap-2 text-lg font-bold mb-4 text-white">
                    <FileCheck className="text-green-400" /> Latest RWA Certificates
                </h3>
                <div className="space-y-3">
                    <AnimatePresence>
                        {mintedAssets.map(asset => (
                            <motion.div 
                                key={asset.id}
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                className="bg-white/5 p-4 rounded-xl border border-white/10 hover:bg-white/10 transition-colors"
                            >
                                <div className="flex justify-between items-start mb-2">
                                    <span className="text-xs text-blue-300 font-mono">ID: {asset.id}</span>
                                    <span className="text-xs bg-blue-600 text-white px-2 rounded-full">NFT</span>
                                </div>
                                <div className="flex justify-between items-end">
                                    <div>
                                        <div className="text-sm font-bold text-white">Flight Data Package</div>
                                        <div className="text-xs text-slate-400">{asset.uavId} • {asset.metadata?.scarcity} SCARCITY</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-lg font-bold text-green-400">${asset.value.toFixed(4)}</div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {mintedAssets.length === 0 && (
                        <div className="text-center text-slate-500 py-8 text-sm">Waiting for blockchain confirmation...</div>
                    )}
                </div>
            </div>

        </div>
    </div>
  );
};

export default AssetLogStream;

