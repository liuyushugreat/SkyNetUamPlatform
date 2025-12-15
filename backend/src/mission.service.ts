import { Injectable } from '@nestjs/common';
import type { Mission, MissionEvent, MissionEventType, MissionState } from './domain.js';
import { transitionState } from './mission-state-machine.js';
import { PersistenceAdapter } from './persistence-adapter.js';

function now() {
  return Date.now();
}

function randomId(prefix: string) {
  return `${prefix}-${now()}-${Math.floor(Math.random() * 1e6)}`;
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

@Injectable()
export class MissionService {
  private readonly missions = new Map<string, Mission>();
  private readonly events: MissionEvent[] = [];
  private readonly persistence = PersistenceAdapter.fromEnv();

  listMissions(state?: MissionState): Mission[] {
    const all = [...this.missions.values()];
    return state ? all.filter((m) => m.state === state) : all;
  }

  getMission(id: string): Mission | undefined {
    return this.missions.get(id);
  }

  createMission(input: { userId: string; routeId: string; operatorId?: string }): Mission {
    const createdAt = now();
    const riskScore = clamp01(0.15 + Math.random() * 0.55);
    const valueScore = Math.max(0, 50 + (Math.random() * 200));

    const mission: Mission = {
      id: randomId('m'),
      userId: input.userId,
      routeId: input.routeId,
      operatorId: input.operatorId,
      state: 'Created',
      timestamps: {
        createdAt,
        lastUpdatedAt: createdAt
      },
      riskScore,
      valueScore,
      persistence: { mode: this.persistence.getMode() }
    };

    this.missions.set(mission.id, mission);
    // Emit initial event
    void this.applyEvent(mission.id, 'MISSION_CREATED', { routeId: mission.routeId, userId: mission.userId });

    return mission;
  }

  async applyEvent(missionId: string, type: MissionEventType, payload?: Record<string, unknown>): Promise<Mission> {
    const m = this.missions.get(missionId);
    if (!m) throw new Error(`Mission not found: ${missionId}`);

    const occurredAt = now();

    // Allow event payload to enrich operational fields (kept off-chain by default)
    if (payload) {
      const operatorId = payload['operatorId'];
      const aircraftId = payload['aircraftId'];
      if (typeof operatorId === 'string') m.operatorId = operatorId;
      if (typeof aircraftId === 'string') m.aircraftId = aircraftId;
    }

    // Map event to state transitions (Created is initial state)
    if (type !== 'MISSION_CREATED') {
      const next = transitionState(m.state, type);
      m.state = next;
    }

    // Timestamp bookkeeping
    if (type === 'MISSION_SCHEDULED') m.timestamps.scheduledAt = occurredAt;
    if (type === 'MISSION_STARTED') m.timestamps.startedAt = occurredAt;
    if (type === 'MISSION_COMPLETED' || type === 'MISSION_FAILED') m.timestamps.endedAt = occurredAt;
    m.timestamps.lastUpdatedAt = occurredAt;

    // Persist (optional / async)
    const persistenceResult = await this.persistence.recordMissionState(missionId, m.state);
    if (persistenceResult && m.persistence) {
      m.persistence.lastTxHash = persistenceResult.txHash;
      m.persistence.lastFinalitySeconds = persistenceResult.finalitySeconds;
    }

    const evt: MissionEvent = {
      id: randomId('e'),
      missionId,
      type,
      occurredAt,
      payload
    };

    this.events.push(evt);

    return m;
  }

  listEvents(missionId?: string): MissionEvent[] {
    return missionId ? this.events.filter((e) => e.missionId === missionId) : [...this.events];
  }
}
