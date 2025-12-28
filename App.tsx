import React, { useState } from 'react';
import CitizenApp from './pages/CitizenApp';
import OperatorApp from './pages/OperatorApp';
import RegulatorApp from './pages/RegulatorApp';
import AssetInsightDashboard from './components/AssetInsight/index';
import FullScreenButton from './components/FullScreenButton';
import { UserRole } from './types';
import { ArrowRight, Smartphone, Briefcase, ShieldCheck, Database } from 'lucide-react';

const App: React.FC = () => {
  const [currentRole, setCurrentRole] = useState<UserRole | null>(null);

  const handleBackToHome = () => setCurrentRole(null);

  if (currentRole === UserRole.CITIZEN) return <CitizenApp onBackToHome={handleBackToHome} />;
  if (currentRole === UserRole.OPERATOR) return <OperatorApp onBackToHome={handleBackToHome} />;
  if (currentRole === UserRole.REGULATOR) return <RegulatorApp onBackToHome={handleBackToHome} />;
  if (currentRole === UserRole.ASSET_INSIGHT) return <AssetInsightDashboard onBackToHome={handleBackToHome} />;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center p-6">
      <div className="max-w-6xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight mb-4">
            SkyNet <span className="text-blue-500">UAM</span> Platform
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto">
            The future of urban air mobility. An integrated platform for low-altitude economy monitoring, booking, and operations.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          {/* Citizen Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.CITIZEN)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-blue-500/20"
          >
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-blue-900/50 group-hover:scale-110 transition-transform">
              <Smartphone className="text-white" size={24} />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">Citizen App</h2>
            <p className="text-slate-400 text-sm mb-4 leading-relaxed">
              Book flights, track trips, and manage your personal mobility data assets.
            </p>
            <div className="flex items-center text-blue-400 text-xs font-bold group-hover:translate-x-2 transition-transform">
              Launch App <ArrowRight size={14} className="ml-2" />
            </div>
          </div>

          {/* Operator Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.OPERATOR)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-indigo-500/20"
          >
            <div className="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-indigo-900/50 group-hover:scale-110 transition-transform">
              <Briefcase className="text-white" size={24} />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">Operator Ops</h2>
            <p className="text-slate-400 text-sm mb-4 leading-relaxed">
              Manage fleets, accept missions, and view real-time revenue analytics.
            </p>
            <div className="flex items-center text-indigo-400 text-xs font-bold group-hover:translate-x-2 transition-transform">
              Open Dashboard <ArrowRight size={14} className="ml-2" />
            </div>
          </div>

          {/* Regulator Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.REGULATOR)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-green-500/20"
          >
            <div className="w-12 h-12 bg-green-600 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-green-900/50 group-hover:scale-110 transition-transform">
              <ShieldCheck className="text-white" size={24} />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">Regulator</h2>
            <p className="text-slate-400 text-sm mb-4 leading-relaxed">
              Monitor airspace compliance, handle alerts, and audit data logs.
            </p>
            <div className="flex items-center text-green-400 text-xs font-bold group-hover:translate-x-2 transition-transform">
              Access Command <ArrowRight size={14} className="ml-2" />
            </div>
          </div>

          {/* Data Economy Card (NEW) */}
          <div 
            onClick={() => setCurrentRole(UserRole.ASSET_INSIGHT)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-purple-500/20"
          >
            <div className="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-purple-900/50 group-hover:scale-110 transition-transform">
              <Database className="text-white" size={24} />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">Data RWA</h2>
            <p className="text-slate-400 text-sm mb-4 leading-relaxed">
              Visualize RWA data assets, valuation flows, and blockchain settlements.
            </p>
            <div className="flex items-center text-purple-400 text-xs font-bold group-hover:translate-x-2 transition-transform">
              Enter Insight <ArrowRight size={14} className="ml-2" />
            </div>
          </div>

        </div>

        <div className="mt-16 text-center border-t border-white/5 pt-8 relative">
            <p className="text-slate-500 text-xs">
                Powered by Google Gemini 2.5 Flash TTS • React • Tailwind CSS
            </p>
            <div className="absolute right-0 top-8">
               <FullScreenButton className="bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white" />
            </div>
        </div>
      </div>
    </div>
  );
};

export default App;