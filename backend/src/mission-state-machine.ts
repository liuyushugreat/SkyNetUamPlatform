import type { MissionEventType, MissionState } from './domain.js';

const TRANSITIONS: Record<MissionState, Partial<Record<MissionEventType, MissionState>>> = {
  Created: {
    MISSION_SCHEDULED: 'Scheduled',
    MISSION_FAILED: 'Failed'
  },
  Scheduled: {
    MISSION_STARTED: 'Active',
    MISSION_DELAYED: 'Delayed',
    MISSION_FAILED: 'Failed'
  },
  Active: {
    MISSION_COMPLETED: 'Completed',
    MISSION_DELAYED: 'Delayed',
    MISSION_FAILED: 'Failed'
  },
  Delayed: {
    MISSION_STARTED: 'Active',
    MISSION_COMPLETED: 'Completed',
    MISSION_FAILED: 'Failed'
  },
  Completed: {},
  Failed: {}
};

export function transitionState(current: MissionState, eventType: MissionEventType): MissionState {
  const next = TRANSITIONS[current]?.[eventType];
  if (!next) {
    throw new Error(`Invalid transition: ${current} + ${eventType}`);
  }
  return next;
}

export function isTerminal(state: MissionState): boolean {
  return state === 'Completed' || state === 'Failed';
}
