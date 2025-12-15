import type { MissionState, PersistenceMode } from './domain.js';

export interface PersistenceResult {
  txHash: string;
  finalitySeconds: number;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function randomHex(n: number) {
  const chars = '0123456789abcdef';
  let s = '';
  for (let i = 0; i < n; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

export class PersistenceAdapter {
  constructor(private readonly mode: PersistenceMode) {}

  getMode() {
    return this.mode;
  }

  async recordMissionState(missionId: string, state: MissionState): Promise<PersistenceResult | null> {
    if (this.mode === 'off') return null;

    // Simulated finality to match the paper narrative (ms-level ops, sec-level chain finality)
    const finalitySeconds =
      this.mode === 'devnet' ? 0.08 : this.mode === 'l2' ? 2.3 : 13.5;

    // Simulate async submission + confirmation
    await delay(Math.max(20, finalitySeconds * 100));

    return {
      txHash: `0x${randomHex(64)}`,
      finalitySeconds
    };
  }

  static fromEnv(): PersistenceAdapter {
    const mode = (process.env.PERSISTENCE_MODE || 'off') as PersistenceMode;
    if (!['off', 'devnet', 'l2', 'l1'].includes(mode)) {
      throw new Error(`Invalid PERSISTENCE_MODE: ${process.env.PERSISTENCE_MODE}`);
    }
    return new PersistenceAdapter(mode);
  }
}
