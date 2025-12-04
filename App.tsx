import React, { useState } from 'react';
import CitizenApp from './pages/CitizenApp';
import OperatorApp from './pages/OperatorApp';
import RegulatorApp from './pages/RegulatorApp';
import { UserRole } from './types';
import { ArrowRight, Smartphone, Briefcase, ShieldCheck } from 'lucide-react';

const App: React.FC = () => {
  const [currentRole, setCurrentRole] = useState<UserRole | null>(null);

  const handleBackToHome = () => setCurrentRole(null);

  if (currentRole === UserRole.CITIZEN) return <CitizenApp onBackToHome={handleBackToHome} />;
  if (currentRole === UserRole.OPERATOR) return <OperatorApp onBackToHome={handleBackToHome} />;
  if (currentRole === UserRole.REGULATOR) return <RegulatorApp onBackToHome={handleBackToHome} />;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center p-6">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight mb-4">
            SkyNet <span className="text-blue-500">UAM</span> Platform
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto">
            The future of urban air mobility. An integrated platform for low-altitude economy monitoring, booking, and operations.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          
          {/* Citizen Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.CITIZEN)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-blue-500/20"
          >
            <div className="w-14 h-14 bg-blue-600 rounded-xl flex items-center justify-center mb-6 shadow-lg shadow-blue-900/50 group-hover:scale-110 transition-transform">
              <Smartphone className="text-white" size={32} />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3">Citizen App</h2>
            <p className="text-slate-400 text-sm mb-6 leading-relaxed">
              Book flights, track trips, and manage your personal mobility data assets.
            </p>
            <div className="flex items-center text-blue-400 text-sm font-bold group-hover:translate-x-2 transition-transform">
              Launch App <ArrowRight size={16} className="ml-2" />
            </div>
          </div>

          {/* Operator Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.OPERATOR)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-indigo-500/20"
          >
            <div className="w-14 h-14 bg-indigo-600 rounded-xl flex items-center justify-center mb-6 shadow-lg shadow-indigo-900/50 group-hover:scale-110 transition-transform">
              <Briefcase className="text-white" size={32} />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3">Operator Ops</h2>
            <p className="text-slate-400 text-sm mb-6 leading-relaxed">
              Manage fleets, accept missions, and view real-time revenue analytics.
            </p>
            <div className="flex items-center text-indigo-400 text-sm font-bold group-hover:translate-x-2 transition-transform">
              Open Dashboard <ArrowRight size={16} className="ml-2" />
            </div>
          </div>

          {/* Regulator Card */}
          <div 
            onClick={() => setCurrentRole(UserRole.REGULATOR)}
            className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all cursor-pointer hover:-translate-y-1 hover:shadow-2xl hover:shadow-green-500/20"
          >
            <div className="w-14 h-14 bg-green-600 rounded-xl flex items-center justify-center mb-6 shadow-lg shadow-green-900/50 group-hover:scale-110 transition-transform">
              <ShieldCheck className="text-white" size={32} />
            </div>
            <h2 className="text-2xl font-bold text-white mb-3">Regulator</h2>
            <p className="text-slate-400 text-sm mb-6 leading-relaxed">
              Monitor airspace compliance, handle alerts, and audit data logs.
            </p>
            <div className="flex items-center text-green-400 text-sm font-bold group-hover:translate-x-2 transition-transform">
              Access Command <ArrowRight size={16} className="ml-2" />
            </div>
          </div>

        </div>

        <div className="mt-16 text-center border-t border-white/5 pt-8">
            <p className="text-slate-500 text-xs">
                Powered by Google Gemini 2.5 Flash TTS • React • Tailwind CSS
            </p>
        </div>
      </div>
    </div>
  );
};

export default App;