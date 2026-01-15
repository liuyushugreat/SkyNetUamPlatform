import { Module } from '@nestjs/common';
import { MissionController } from './mission.controller.js';
import { MissionService } from './mission.service.js';
import { Neo4jModule } from './integrations/neo4j/neo4j.module.js';

@Module({
  imports: [Neo4jModule],
  controllers: [MissionController],
  providers: [MissionService]
})
export class AppModule {}
