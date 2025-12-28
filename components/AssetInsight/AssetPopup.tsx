import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as echarts from 'echarts';
import { Shield, Coins, Activity, TrendingUp } from 'lucide-react';

interface AssetPopupProps {
  uavId: string;
  position: { x: number; y: number }; 
  data: {
    value: number;
    accumulated: number;
    integrity: number;
    history: number[]; // Last 5 mins data points
  };
  onClose?: () => void;
}

const AssetPopup: React.FC<AssetPopupProps> = ({ uavId, position, data, onClose }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current || !data.history.length) return;
    
    const chart = echarts.init(chartRef.current);
    const option = {
      grid: { top: 5, right: 5, bottom: 5, left: 5 },
      xAxis: { type: 'category', show: false },
      yAxis: { type: 'value', show: false, min: 'dataMin' },
      series: [{
        data: data.history,
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: '#4ade80' },
        areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(74, 222, 128, 0.5)' },
              { offset: 1, color: 'rgba(74, 222, 128, 0)' }
            ])
        }
      }]
    };
    chart.setOption(option);
    return () => chart.dispose();
  }, [data.history]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="absolute z-50 bg-slate-900/95 backdrop-blur-md border border-slate-600 rounded-xl p-4 shadow-2xl w-72 pointer-events-auto"
        style={{ left: position.x, top: position.y }}
      >
        {/* Header */}
        <div className="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
          <div className="flex items-center gap-2">
             <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
             <span className="font-bold text-white text-sm">{uavId}</span>
          </div>
          <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/30 font-mono">
             RWA ACTIVE
          </span>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="bg-slate-800/50 p-2 rounded border border-slate-700">
                <p className="text-[10px] text-slate-400 uppercase">Current Value</p>
                <p className="text-lg font-bold text-green-400 flex items-center gap-1">
                    <Coins size={14} /> ${data.value.toFixed(2)}
                </p>
            </div>
            <div className="bg-slate-800/50 p-2 rounded border border-slate-700">
                <p className="text-[10px] text-slate-400 uppercase">Accumulated</p>
                <p className="text-lg font-bold text-blue-400">
                    ${data.accumulated.toFixed(2)}
                </p>
            </div>
        </div>

        {/* Integrity Score */}
        <div className="mb-3">
             <div className="flex justify-between text-xs mb-1">
                 <span className="text-slate-400 flex items-center gap-1"><Shield size={12}/> Data Integrity</span>
                 <span className="text-white font-mono">{data.integrity}%</span>
             </div>
             <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                 <div className="bg-purple-500 h-full rounded-full" style={{ width: `${data.integrity}%` }}></div>
             </div>
        </div>

        {/* Trend Chart */}
        <div className="h-16 w-full bg-slate-800/30 rounded border border-slate-700/50 relative overflow-hidden">
             <div className="absolute top-1 left-2 text-[10px] text-slate-500 flex items-center gap-1">
                 <TrendingUp size={10} /> 5m Trend
             </div>
             <div ref={chartRef} className="w-full h-full"></div>
        </div>

      </motion.div>
    </AnimatePresence>
  );
};

export default AssetPopup;

