
import React, { useEffect, useState } from 'react';
import { RWAToken } from '../types';
import { smartContractService } from '../services/smartContractService';
import { TrendingUp, TrendingDown, DollarSign, PieChart, Loader2, RefreshCw } from 'lucide-react';
import { playTextToSpeech } from '../services/geminiService';

const RWAMarket: React.FC = () => {
  const [tokens, setTokens] = useState<RWAToken[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [buyingId, setBuyingId] = useState<string | null>(null);
  const [userBalance, setUserBalance] = useState<Record<string, number>>({});
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  const fetchMarketData = async () => {
    setIsLoading(true);
    try {
      // Simulate live market movement on refresh
      await smartContractService.simulateMarketActivity();
      
      const data = await smartContractService.getAllTokens();
      setTokens(data);
      
      // Fetch balances for current mock user
      const balances: Record<string, number> = {};
      for(const t of data) {
          balances[t.id] = await smartContractService.getUserBalance('user-citizen', t.id);
      }
      setUserBalance(balances);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Failed to fetch market data", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(fetchMarketData, 10000); // Poll for updates
    return () => clearInterval(interval);
  }, []);

  const handleBuy = async (tokenId: string) => {
    setBuyingId(tokenId);
    try {
        await smartContractService.purchaseToken('user-citizen', tokenId, 100); // Buy 100 tokens fixed for demo
        await playTextToSpeech("Transaction confirmed. Assets added to your wallet.");
        await fetchMarketData();
    } catch (e) {
        console.error(e);
        await playTextToSpeech("Transaction failed. Insufficient supply.");
    } finally {
        setBuyingId(null);
    }
  };

  const totalPortfolioValue = Object.entries(userBalance).reduce((acc: number, [tokenId, qty]) => {
      const token = tokens.find(t => t.id === tokenId);
      return acc + (token ? token.price * (qty as number) : 0);
  }, 0);

  return (
    <div className="space-y-6">
      {/* Portfolio Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 p-6 rounded-2xl text-white shadow-lg relative overflow-hidden">
        <div className="relative z-10">
            <h2 className="text-sm text-slate-400 uppercase tracking-wider mb-1">Total Asset Value</h2>
            <div className="flex items-baseline gap-2">
                <h1 className="text-3xl font-bold">${totalPortfolioValue.toFixed(2)}</h1>
                <span className="text-green-400 text-sm flex items-center font-medium"><TrendingUp size={14} className="mr-1"/> +4.5%</span>
            </div>
            <div className="mt-4 flex gap-3">
                <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">Deposit</button>
                <button className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">Withdraw</button>
            </div>
        </div>
        <PieChart className="absolute right-4 bottom-4 text-slate-700 opacity-20" size={100} />
      </div>

      {/* Market List */}
      <div>
        <div className="flex justify-between items-center mb-3">
            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <DollarSign className="text-blue-600" size={20} /> RWA Token Exchange
            </h3>
            <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400 hidden sm:block">
                    Updated {lastUpdated.toLocaleTimeString()}
                </span>
                <button 
                onClick={fetchMarketData} 
                disabled={isLoading}
                className="flex items-center gap-2 text-xs font-medium bg-blue-100 text-blue-700 px-3 py-1.5 rounded-lg hover:bg-blue-200 transition-all shadow-sm disabled:opacity-70 active:scale-95"
                title="Force refresh market data"
                >
                    <RefreshCw size={14} className={isLoading ? "animate-spin" : ""} />
                    {isLoading ? 'Syncing...' : 'Refresh Data'}
                </button>
            </div>
        </div>
        
        <div className="space-y-3">
            {tokens.map(token => {
                const soldPercentage = ((token.totalSupply - token.availableSupply) / token.totalSupply) * 100;
                const myQty = userBalance[token.id] || 0;

                return (
                <div key={token.id} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
                    <div className="flex justify-between items-start mb-2 relative z-10">
                        <div>
                            <div className="flex items-center gap-2">
                                <h4 className="font-bold text-slate-800">{token.name}</h4>
                                {myQty > 0 && <span className="text-[10px] bg-green-100 text-green-700 px-2 rounded-full">Owned: {myQty}</span>}
                            </div>
                            <p className="text-xs font-mono text-slate-500">{token.symbol}</p>
                        </div>
                        <div className="text-right">
                            <p className="font-bold text-slate-800">${token.price.toFixed(2)}</p>
                            <p className={`text-xs font-medium flex items-center justify-end ${token.change24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {token.change24h >= 0 ? <TrendingUp size={12} className="mr-1"/> : <TrendingDown size={12} className="mr-1"/>}
                                {token.change24h}%
                            </p>
                        </div>
                    </div>
                    
                    <p className="text-xs text-slate-600 mb-3 relative z-10">{token.description}</p>
                    
                    {/* Progress Bar for ICO/Funding */}
                    <div className="mb-3 relative z-10">
                        <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                            <span>Sold: {soldPercentage.toFixed(1)}%</span>
                            <span>Total Supply: {token.totalSupply.toLocaleString()}</span>
                        </div>
                        <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                            <div className="bg-blue-500 h-full rounded-full" style={{ width: `${soldPercentage}%` }}></div>
                        </div>
                    </div>

                    <div className="flex justify-between items-center pt-3 border-t border-slate-100 relative z-10">
                        <div className="text-xs text-slate-500">
                            APY: <span className="font-bold text-green-600">{token.yieldApy.toFixed(2)}%</span>
                        </div>
                        <button 
                            onClick={() => handleBuy(token.id)}
                            disabled={buyingId === token.id || token.availableSupply <= 0}
                            className="bg-blue-50 hover:bg-blue-100 text-blue-600 text-sm font-semibold px-4 py-1.5 rounded transition-colors flex items-center gap-2 disabled:opacity-50"
                        >
                            {buyingId === token.id ? <Loader2 size={14} className="animate-spin" /> : 'Buy 100'}
                        </button>
                    </div>
                </div>
            )})}
        </div>
      </div>
    </div>
  );
};

export default RWAMarket;
