import { Route, Aircraft, Order, OrderStatus, Alert, RWAToken, NoFlyZone, InsuranceClaim } from '../types';

export const ROUTES: Route[] = [
  {
    id: 'r-1',
    name: 'Skyline Scenic Tour',
    startPoint: 'Downtown Heliport',
    endPoint: 'Bay View Park',
    durationMinutes: 15,
    price: 150,
    distanceKm: 12,
    coordinates: [{x: 20, y: 80}, {x: 40, y: 60}, {x: 60, y: 30}]
  },
  {
    id: 'r-2',
    name: 'Airport Express Shuttle',
    startPoint: 'City Center',
    endPoint: 'Intl Airport',
    durationMinutes: 8,
    price: 95,
    distanceKm: 18,
    coordinates: [{x: 50, y: 50}, {x: 80, y: 80}]
  },
  {
    id: 'r-3',
    name: 'Medical Emergency Corridor',
    startPoint: 'General Hospital',
    endPoint: 'Northside Clinic',
    durationMinutes: 5,
    price: 200,
    distanceKm: 8,
    coordinates: [{x: 10, y: 10}, {x: 30, y: 20}]
  }
];

export const AIRCRAFT: Aircraft[] = [
  {
    id: 'ac-001',
    model: 'EHang 216',
    regNumber: 'UAM-NY-01',
    status: 'AVAILABLE',
    operator: 'SkyHigh Ops',
    batteryLevel: 85,
    currentLocation: { x: 20, y: 80 },
    videoFeedUrl: 'https://images.unsplash.com/photo-1473968512647-3e447244af8f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    speed: 0,
    altitude: 0
  },
  {
    id: 'ac-002',
    model: 'Joby S4',
    regNumber: 'UAM-NY-02',
    status: 'BUSY',
    operator: 'Urban Wings',
    batteryLevel: 62,
    currentLocation: { x: 45, y: 55 },
    videoFeedUrl: 'https://images.unsplash.com/photo-1551516594-56cb7830558c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    speed: 145,
    altitude: 450
  },
  {
    id: 'ac-003',
    model: 'Volocopter 2X',
    regNumber: 'UAM-NY-03',
    status: 'MAINTENANCE',
    operator: 'SkyHigh Ops',
    batteryLevel: 10,
    currentLocation: { x: 90, y: 90 },
    videoFeedUrl: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    speed: 0,
    altitude: 0
  }
];

export const INITIAL_ORDERS: Order[] = [
  {
    id: 'ord-1001',
    userId: 'u-1',
    routeId: 'r-1',
    aircraftId: 'ac-001',
    status: OrderStatus.COMPLETED,
    timestamp: Date.now() - 86400000,
    amount: 150,
    txHash: '0x71c...9a2'
  },
  {
    id: 'ord-1002',
    userId: 'u-1',
    routeId: 'r-2',
    status: OrderStatus.PENDING,
    timestamp: Date.now() - 3600000,
    amount: 95,
  }
];

export const ALERTS: Alert[] = [
  {
    id: 'al-1',
    severity: 'MEDIUM',
    message: 'UAM-NY-02 deviated slightly from corridor',
    timestamp: Date.now() - 120000,
    status: 'OPEN',
    aircraftId: 'ac-002'
  },
  {
    id: 'al-2',
    severity: 'LOW',
    message: 'UAM-NY-03 maintenance schedule overdue',
    timestamp: Date.now() - 5000000,
    status: 'OPEN',
    aircraftId: 'ac-003'
  }
];

export const RWA_TOKENS: RWAToken[] = [
  {
    id: 't-1',
    assetId: 'r-1',
    issuerId: 'SkyHigh Ops',
    symbol: 'SKY-R1',
    name: 'Scenic Route Data Fund',
    price: 12.45,
    change24h: 2.3,
    marketCap: '$1.2M',
    yieldApy: 8.5,
    description: 'Fractional ownership of flight data revenue from the Skyline Scenic Tour route.',
    totalSupply: 100000,
    availableSupply: 45000
  },
  {
    id: 't-2',
    assetId: 'infra-1',
    issuerId: 'City Govt',
    symbol: 'UAM-INFRA',
    name: 'City Vertiport Bond',
    price: 98.20,
    change24h: 0.1,
    marketCap: '$45M',
    yieldApy: 4.2,
    description: 'Tokenized debt instrument funding new vertiport infrastructure in District 9.',
    totalSupply: 500000,
    availableSupply: 120000
  },
  {
    id: 't-3',
    assetId: 'fleet-a',
    issuerId: 'Urban Wings',
    symbol: 'DRONE-IDX',
    name: 'Autonomous Fleet Index',
    price: 156.00,
    change24h: -1.2,
    marketCap: '$8.5M',
    yieldApy: 6.8,
    description: 'Basket token representing the aggregate performance of 50+ commercial drones.',
    totalSupply: 50000,
    availableSupply: 2000
  }
];

export const NO_FLY_ZONES: NoFlyZone[] = [
  { id: 'nfz-1', x: 50, y: 50, radius: 10, reason: 'Military Base' },
  { id: 'nfz-2', x: 80, y: 20, radius: 8, reason: 'High Rise Construction' }
];

export const INSURANCE_CLAIMS: InsuranceClaim[] = [
  {
    id: 'clm-001',
    aircraftId: 'ac-003',
    incidentDate: Date.now() - 100000000,
    description: 'Rotor damage during high wind landing.',
    status: 'APPROVED',
    amount: 2500
  }
];