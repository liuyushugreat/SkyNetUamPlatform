import { Module } from '@nestjs/common';
import { MissionController } from './mission.controller.js';
import { MissionService } from './mission.service.js';

@Module({
  controllers: [MissionController],
  providers: [MissionService]
})
export class AppModule {}
