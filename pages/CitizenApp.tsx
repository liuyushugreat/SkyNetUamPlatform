import React, { useState } from 'react';
import { Route, Order, OrderStatus } from '../types';
import { ROUTES, INITIAL_ORDERS } from '../services/mockData';
import MapVisualization from '../components/MapVisualization';
import RWAMarket from '../components/RWAMarket';
import { playTextToSpeech } from '../services/geminiService';
import { smartContractService } from '../services/smartContractService';
import { opsApi } from '../services/opsApi';
import { Loader2, Volume2, MapPin, Clock, CheckCircle, DollarSign, Plane, LayoutDashboard, Home } from 'lucide-react';

interface CitizenAppProps {
  onBackToHome: () => void;
}

const CitizenApp: React.FC<CitizenAppProps> = ({ onBackToHome }) => {
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [orders, setOrders] = useState<Order[]>(INITIAL_ORDERS);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState<'book' | 'trips' | 'market'>('book');

  const handleBooking = async () => {
    if (!selectedRoute) return;
    setIsProcessing(true);
    
    // Simulate booking process
    setTimeout(async () => {
      const newOrder: Order = {
        id: `ord-${Date.now()}`,
        userId: 'u-1',
        routeId: selectedRoute.id,
        status: OrderStatus.PAID, // Auto pay for MVP
        timestamp: Date.now(),
        amount: selectedRoute.price,
        txHash: `0x${Math.random().toString(16).slice(2)}` // Simulated hash
      };
      
      setOrders([newOrder, ...orders]);
      setIsProcessing(false);
      setActiveTab('trips');
      setSelectedRoute(null);

      // Mission lifecycle (paper-aligned): create -> schedule -> start -> complete
      try {
        const operatorId = 'SkyHigh Ops';
        const aircraftId = 'ac-001';
        const created = await opsApi.createMission({ userId: 'u-1', routeId: newOrder.routeId, operatorId });
        const scheduled = await opsApi.sendMissionEvent(created.id, 'MISSION_SCHEDULED', { operatorId, aircraftId });
        setOrders((prev) =>
          prev.map((o) => (o.id === newOrder.id ? { ...o, missionId: scheduled.id, missionState: scheduled.state, aircraftId } : o))
        );

        // Simulate execution timeline
        setTimeout(async () => {
          try {
            const active = await opsApi.sendMissionEvent(created.id, 'MISSION_STARTED', { operatorId, aircraftId });
            setOrders((prev) =>
              prev.map((o) => (o.id === newOrder.id ? { ...o, missionState: active.state, status: OrderStatus.IN_PROGRESS } : o))
            );
          } catch {}
        }, 800);

        setTimeout(async () => {
          try {
            const done = await opsApi.sendMissionEvent(created.id, 'MISSION_COMPLETED', { operatorId, aircraftId });
            setOrders((prev) =>
              prev.map((o) => (o.id === newOrder.id ? { ...o, missionState: done.state, status: OrderStatus.COMPLETED } : o))
            );
          } catch {}
        }, Math.max(1200, selectedRoute.durationMinutes * 120)); // scale down for demo
      } catch (e) {
        console.warn('Ops service unavailable, running in standalone demo mode.', e);
      }

      // Trigger RWA Logic: Record revenue on-chain
      // This links the flight data (booking) to the token value
      try {
        await smartContractService.recordAssetRevenue(selectedRoute.id, selectedRoute.price);
        console.log(`Revenue recorded on-chain for asset ${selectedRoute.id}`);
      } catch (e) {
        console.error("Failed to record revenue on smart contract", e);
      }

      // TTS Announcement
      try {
        await playTextToSpeech(`Booking confirmed. Your flight to ${selectedRoute.endPoint} is scheduled. Revenue has been recorded on the blockchain.`);
      } catch (e) {
        console.error("TTS failed", e);
      }

    }, 2000);
  };

  return (
    <div className="max-w-md mx-auto bg-white min-h-screen shadow-xl overflow-hidden flex flex-col">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md flex justify-between items-center">
        <div className="flex items-center gap-3">
          <button onClick={onBackToHome} className="p-1.5 hover:bg-blue-500 rounded-full transition-colors" title="Return Home">
            <Home size={20} />
          </button>
          <div>
            <h1 className="text-lg font-bold">SkyNet Mobility</h1>
            <p className="text-xs opacity-80">Citizen Access</p>
          </div>
        </div>
        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-xs font-bold border border-blue-400">
          ME
        </div>
      </header>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 bg-white">
        <button onClick={() => setActiveTab('book')} className={`flex-1 py-3 text-xs flex flex-col items-center gap-1 ${activeTab === 'book' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-slate-500'}`}>
           <Plane size={18} /> Book Flight
        </button>
        <button onClick={() => setActiveTab('trips')} className={`flex-1 py-3 text-xs flex flex-col items-center gap-1 ${activeTab === 'trips' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-slate-500'}`}>
           <LayoutDashboard size={18} /> My Trips
        </button>
        <button onClick={() => setActiveTab('market')} className={`flex-1 py-3 text-xs flex flex-col items-center gap-1 ${activeTab === 'market' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-slate-500'}`}>
           <DollarSign size={18} /> Data Market
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto bg-slate-50 p-4 custom-scrollbar">
        
        {activeTab === 'book' && (
          <div className="space-y-4">
            <div className="bg-white p-2 rounded-xl shadow-sm border border-slate-200">
               <h2 className="text-sm font-semibold text-slate-700 mb-2 px-2">Live Airspace</h2>
               <MapVisualization routes={ROUTES} aircraft={[]} className="h-48 w-full" showLabels={false} />
            </div>

            <h2 className="text-sm font-semibold text-slate-700 mt-4">Available Routes</h2>
            <div className="space-y-3">
              {ROUTES.map(route => (
                <div 
                  key={route.id}
                  onClick={() => setSelectedRoute(route)}
                  className={`p-4 rounded-xl border transition-all cursor-pointer ${selectedRoute?.id === route.id ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-100' : 'border-slate-200 bg-white hover:border-blue-300'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold text-slate-800">{route.name}</h3>
                    <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded-full font-medium">${route.price}</span>
                  </div>
                  <div className="flex items-center text-xs text-slate-500 gap-3">
                    <span className="flex items-center gap-1"><Clock size={14} /> {route.durationMinutes} min</span>
                    <span className="flex items-center gap-1"><MapPin size={14} /> {route.distanceKm} km</span>
                  </div>
                </div>
              ))}
            </div>

            {selectedRoute && (
              <div className="fixed bottom-0 left-0 right-0 p-4 bg-white border-t border-slate-200 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] max-w-md mx-auto z-10">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <p className="text-sm text-slate-500">Total Fare</p>
                    <p className="text-2xl font-bold text-slate-800">${selectedRoute.price}</p>
                  </div>
                  <button 
                    onClick={handleBooking}
                    disabled={isProcessing}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-semibold shadow-lg shadow-blue-200 flex items-center gap-2 disabled:opacity-70"
                  >
                    {isProcessing ? <Loader2 className="animate-spin" /> : 'Pay & Book'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trips' && (
          <div className="space-y-4">
            {orders.map(order => {
              const route = ROUTES.find(r => r.id === order.routeId);
              return (
                <div key={order.id} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h3 className="font-semibold text-slate-800">{route?.name || 'Unknown Route'}</h3>
                      <p className="text-xs text-slate-500">{new Date(order.timestamp).toLocaleDateString()} • {new Date(order.timestamp).toLocaleTimeString()}</p>
                      {order.missionId && (
                        <p className="text-[10px] text-slate-400 font-mono mt-1">
                          Mission: {order.missionId} • State: {order.missionState || '—'}
                        </p>
                      )}
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${order.status === 'COMPLETED' ? 'bg-slate-100 text-slate-600' : 'bg-blue-100 text-blue-600'}`}>
                      {order.status}
                    </span>
                  </div>
                  
                  {order.status === 'PENDING' && (
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-100 flex items-center gap-3">
                        <Loader2 className="text-blue-600 animate-spin" size={20} />
                        <p className="text-xs text-blue-700">Waiting for operator assignment...</p>
                    </div>
                  )}

                  {order.txHash && (
                     <div className="mt-3 pt-3 border-t border-slate-100 flex justify-between items-center">
                        <span className="text-[10px] text-slate-400 font-mono truncate w-32">Hash: {order.txHash}</span>
                        <div className="flex gap-2">
                            <button onClick={() => playTextToSpeech(`Your flight on ${route?.name} is confirmed. Data recorded on chain.`)} className="text-slate-400 hover:text-blue-600">
                                <Volume2 size={16} />
                            </button>
                            <CheckCircle size={16} className="text-green-500" />
                        </div>
                     </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {activeTab === 'market' && (
          <RWAMarket />
        )}

      </div>
    </div>
  );
};

export default CitizenApp;