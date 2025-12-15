import { Body, Controller, Get, Param, Post, Query } from '@nestjs/common';
import type { Mission, MissionEvent, MissionEventType, MissionState } from './domain.js';
import { MissionService } from './mission.service.js';

@Controller()
export class MissionController {
  constructor(private readonly missions: MissionService) {}

  // Paper-style event ingestion endpoints
  @Post('api/events/mission-created')
  createEventCreated(@Body() body: { userId: string; routeId: string; operatorId?: string }): Mission {
    return this.missions.createMission(body);
  }

  @Post('api/events/:missionId/:event')
  async ingestEvent(
    @Param('missionId') missionId: string,
    @Param('event') event: string,
    @Body() payload?: Record<string, unknown>
  ): Promise<Mission> {
    const type = normalizeEvent(event);
    return await this.missions.applyEvent(missionId, type, payload);
  }

  // Convenience REST endpoints
  @Get('api/missions')
  list(@Query('state') state?: MissionState): Mission[] {
    return this.missions.listMissions(state);
  }

  @Get('api/missions/:missionId')
  get(@Param('missionId') missionId: string): Mission {
    const m = this.missions.getMission(missionId);
    if (!m) throw new Error('NotFound');
    return m;
  }

  @Get('api/events')
  listEvents(@Query('missionId') missionId?: string): MissionEvent[] {
    return this.missions.listEvents(missionId);
  }
}

function normalizeEvent(event: string): MissionEventType {
  const key = event.toUpperCase();
  const allowed: Record<string, MissionEventType> = {
    'MISSION_SCHEDULED': 'MISSION_SCHEDULED',
    'MISSION_STARTED': 'MISSION_STARTED',
    'MISSION_DELAYED': 'MISSION_DELAYED',
    'MISSION_FAILED': 'MISSION_FAILED',
    'MISSION_COMPLETED': 'MISSION_COMPLETED'
  };
  const mapped = allowed[key];
  if (!mapped) throw new Error(`Unsupported event: ${event}`);
  return mapped;
}
