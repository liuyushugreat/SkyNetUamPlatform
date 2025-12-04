
import { RWAToken, Aircraft, Route } from '../types';
import { RWA_TOKENS } from './mockData';

/**
 * SmartContractService
 * 
 * Simulates interaction with an Ethereum/Polygon smart contract for RWA.
 * In a production app, this would use ethers.js or web3.js to talk to a real node.
 */

// In-memory ledger simulation
let ledgerTokens: RWAToken[] = [...RWA_TOKENS];
let userBalances: Record<string, Record<string, number>> = {
  'user-citizen': { 't-1': 10 } // Initial balance for demo
};

// Helpers
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const generateAddress = () => '0x' + Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('');

export const smartContractService = {
  
  // 1. Fetch all available RWA tokens from the registry contract
  getAllTokens: async (): Promise<RWAToken[]> => {
    await delay(500); // Network latency
    return [...ledgerTokens];
  },

  // 2. Operator: "Mint" new tokens backed by an asset (Route/Aircraft)
  deployAssetToken: async (
    asset: Aircraft | Route, 
    valuation: number, 
    symbol: string,
    issuerId: string
  ): Promise<RWAToken> => {
    console.log(`[SmartContract] Deploying RWA contract for ${symbol}...`);
    await delay(1500); // Deployment taking place

    // Validate if asset is already tokenized
    const existing = ledgerTokens.find(t => t.assetId === asset.id);
    if (existing) {
      throw new Error(`Asset ${asset.id} is already tokenized as ${existing.symbol}`);
    }
    
    // Validate if symbol is taken
    if (ledgerTokens.some(t => t.symbol === symbol)) {
        throw new Error(`Symbol ${symbol} is already in use.`);
    }

    const totalSupply = Math.floor(valuation); // 1 token = $1 initial peg
    const assetName = 'name' in asset ? asset.name : asset.model;
    const contractAddr = generateAddress();
    const now = Date.now();

    const newToken: RWAToken = {
      id: `t-${now}`,
      assetId: asset.id,
      issuerId: issuerId,
      symbol: symbol.toUpperCase(),
      name: `${assetName} Revenue Share`,
      price: 1.00, // Initial Issue Price
      change24h: 0,
      marketCap: new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(valuation),
      yieldApy: 5 + Math.random() * 5, // Initial estimated APY
      description: `Official revenue sharing token for ${assetName}. Holders receive dividends from flight operations linked to Smart Contract ${contractAddr.substring(0, 8)}...`,
      totalSupply: totalSupply,
      availableSupply: totalSupply, // All available at mint
      contractAddress: contractAddr,
      ownerCount: 1, // Issuer is first owner
      deploymentDate: now
    };

    ledgerTokens.unshift(newToken); // Add to top of ledger
    
    // Initialize issuer balance
    if (!userBalances[issuerId]) userBalances[issuerId] = {};
    userBalances[issuerId][newToken.id] = totalSupply;

    console.log(`[SmartContract] Contract deployed at ${contractAddr}`);
    return newToken;
  },

  // 3. Citizen: Buy tokens (Calls 'transfer' or 'mint' on contract)
  purchaseToken: async (userId: string, tokenId: string, amount: number): Promise<boolean> => {
    console.log(`[SmartContract] Processing purchase of ${amount} tokens for ${tokenId}`);
    await delay(1500); // Transaction confirmation time

    const tokenIndex = ledgerTokens.findIndex(t => t.id === tokenId);
    if (tokenIndex === -1) throw new Error("Token not found");

    const token = ledgerTokens[tokenIndex];

    if (token.availableSupply < amount) {
      throw new Error("Insufficient supply");
    }

    // Update Ledger State
    ledgerTokens[tokenIndex] = {
      ...token,
      availableSupply: token.availableSupply - amount,
      price: token.price * (1 + (0.0005 * amount)), // Dynamic Bonding curve
      ownerCount: (token.ownerCount || 0) + 1
    };

    // Update User Balance
    if (!userBalances[userId]) userBalances[userId] = {};
    userBalances[userId][tokenId] = (userBalances[userId][tokenId] || 0) + amount;

    return true;
  },

  // 4. Get User Balance
  getUserBalance: async (userId: string, tokenId: string): Promise<number> => {
    return userBalances[userId]?.[tokenId] || 0;
  },

  // 5. Simulate Revenue Distribution (Smart Contract Logic)
  distributeRevenue: async (tokenId: string, revenue: number) => {
    const token = ledgerTokens.find(t => t.id === tokenId);
    if (!token) return;

    console.log(`[SmartContract] Distributing $${revenue} revenue to holders of ${token.symbol}`);

    const yieldBoost = (revenue / token.totalSupply) * 200; 
    token.yieldApy = Math.min(120, (token.yieldApy || 5) + yieldBoost);
    
    const priceImpact = (revenue / token.totalSupply) * 2.5; 
    token.price = token.price + priceImpact;
    
    token.change24h = parseFloat((token.change24h + (yieldBoost / 2)).toFixed(2));
  },

  // 6. Process Asset Revenue (Linker)
  recordAssetRevenue: async (assetId: string, revenue: number) => {
    const token = ledgerTokens.find(t => t.assetId === assetId);
    if (token) {
        await smartContractService.distributeRevenue(token.id, revenue);
    }
  },

  // 7. Simulate Market Activity (Random Fluctuation)
  simulateMarketActivity: async () => {
    // Introduces small micro-fluctuations to simulate an active order book
    ledgerTokens.forEach(token => {
      const volatility = 0.005; // 0.5% max swing
      const change = 1 + ((Math.random() * volatility * 2) - volatility);
      
      // Update price
      token.price = Math.max(0.01, token.price * change);
      
      // Update market cap display
      token.marketCap = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
        .format(token.price * token.totalSupply);

      // Update 24h change slightly to match trend
      const changeDelta = (change - 1) * 100;
      token.change24h = parseFloat((token.change24h + changeDelta).toFixed(2));
    });
  }
};
