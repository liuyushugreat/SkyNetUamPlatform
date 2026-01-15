export type MissionState =
  | 'Created'
  | 'Scheduled'
  | 'Active'
  | 'Completed'
  | 'Delayed'
  | 'Failed';

export type PersistenceMode = 'off' | 'devnet' | 'l2' | 'l1';

export type MissionEventType =
  | 'MISSION_CREATED'
  | 'MISSION_SCHEDULED'
  | 'MISSION_STARTED'
  | 'MISSION_DELAYED'
  | 'MISSION_FAILED'
  | 'MISSION_COMPLETED';

export interface MissionTimestamps {
  createdAt: number;
  scheduledAt?: number;
  startedAt?: number;
  endedAt?: number;
  lastUpdatedAt: number;
}

export interface Mission {
  id: string;
  userId: string;
  routeId: string;
  operatorId?: string;
  aircraftId?: string;

  state: MissionState;
  timestamps: MissionTimestamps;

  // Operational indicators (for experiments/analytics)
  riskScore: number; // 0..1
  valueScore: number; // abstract operational value

  // Optional persistence metadata
  persistence?: {
    mode: PersistenceMode;
    lastTxHash?: string;
    lastFinalitySeconds?: number;
  };
}

export interface MissionEvent {
  id: string;
  missionId: string;
  type: MissionEventType;
  occurredAt: number;
  payload?: Record<string, unknown>;
}
