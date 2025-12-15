import type { Mission, MissionEventType } from '../types';

const DEFAULT_BASE = 'http://localhost:3001';

function baseUrl() {
  // Vite env var
  const v = (import.meta as any)?.env?.VITE_OPS_API_BASE_URL;
  return (typeof v === 'string' && v.length > 0) ? v : DEFAULT_BASE;
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${baseUrl()}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {})
    }
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Ops API error ${res.status}: ${text || res.statusText}`);
  }

  return (await res.json()) as T;
}

export const opsApi = {
  createMission: async (input: { userId: string; routeId: string; operatorId?: string }): Promise<Mission> => {
    return await http<Mission>('/api/events/mission-created', {
      method: 'POST',
      body: JSON.stringify(input)
    });
  },

  sendMissionEvent: async (
    missionId: string,
    event: MissionEventType,
    payload?: Record<string, unknown>
  ): Promise<Mission> => {
    return await http<Mission>(`/api/events/${encodeURIComponent(missionId)}/${encodeURIComponent(event)}`, {
      method: 'POST',
      body: JSON.stringify(payload || {})
    });
  },

  listMissions: async (): Promise<Mission[]> => {
    return await http<Mission[]>('/api/missions');
  }
};
