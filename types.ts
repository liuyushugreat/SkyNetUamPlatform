
export enum UserRole {
  CITIZEN = 'CITIZEN',
  OPERATOR = 'OPERATOR',
  REGULATOR = 'REGULATOR',
  ASSET_INSIGHT = 'ASSET_INSIGHT'
}

export enum OrderStatus {
  PENDING = 'PENDING',
  PAID = 'PAID',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED'
}

// Mission lifecycle (paper-aligned)
export type MissionState =
  | 'Created'
  | 'Scheduled'
  | 'Active'
  | 'Completed'
  | 'Delayed'
  | 'Failed';

export type MissionEventType =
  | 'MISSION_CREATED'
  | 'MISSION_SCHEDULED'
  | 'MISSION_STARTED'
  | 'MISSION_DELAYED'
  | 'MISSION_FAILED'
  | 'MISSION_COMPLETED';

export interface Coordinate {
  x: number;
  y: number;
}

export interface Route {
  id: string;
  name: string;
  startPoint: string;
  endPoint: string;
  durationMinutes: number;
  price: number;
  distanceKm: number;
  coordinates: Coordinate[];
  isCustom?: boolean;
}

export interface Aircraft {
  id: string;
  model: string;
  regNumber: string;
  status: 'AVAILABLE' | 'BUSY' | 'MAINTENANCE' | 'OFFLINE' | 'EMERGENCY';
  operator: string;
  batteryLevel: number;
  currentLocation: Coordinate;
  videoFeedUrl?: string;
  speed?: number; // km/h
  altitude?: number; // meters
}

export interface Order {
  id: string;
  userId: string;
  routeId: string;
  aircraftId?: string;
  status: OrderStatus;
  timestamp: number;
  amount: number;
  txHash?: string;
  missionId?: string;
  missionState?: MissionState;
}

export interface Mission {
  id: string;
  userId: string;
  routeId: string;
  operatorId?: string;
  aircraftId?: string;
  state: MissionState;
  timestamps: {
    createdAt: number;
    scheduledAt?: number;
    startedAt?: number;
    endedAt?: number;
    lastUpdatedAt: number;
  };
  riskScore: number;
  valueScore: number;
  persistence?: {
    mode: 'off' | 'devnet' | 'l2' | 'l1';
    lastTxHash?: string;
    lastFinalitySeconds?: number;
  };
}

export interface Alert {
  id: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  timestamp: number;
  status: 'OPEN' | 'RESOLVED';
  aircraftId?: string;
}

// New Types for RWA & Advanced Features
export interface RWAToken {
  id: string;
  assetId: string; // Link to Route ID or Aircraft ID
  issuerId: string;
  symbol: string;
  name: string;
  price: number;
  change24h: number; // Percentage
  marketCap: string;
  yieldApy: number;
  description: string;
  totalSupply: number;
  availableSupply: number;
  contractAddress?: string; // Blockchain address
  ownerCount?: number;
  deploymentDate?: number;
}

export interface NoFlyZone {
  id: string;
  x: number;
  y: number;
  radius: number;
  reason: string;
}

export interface InsuranceClaim {
  id: string;
  aircraftId: string;
  incidentDate: number;
  description: string;
  status: 'SUBMITTED' | 'REVIEWING' | 'APPROVED' | 'REJECTED';
  amount: number;
}
