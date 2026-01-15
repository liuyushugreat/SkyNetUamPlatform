import { Module, Global } from '@nestjs/common';
import { Neo4jService } from './neo4j.service.js';

@Global()
@Module({
  providers: [Neo4jService],
  exports: [Neo4jService],
})
export class Neo4jModule {}

