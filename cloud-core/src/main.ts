import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module.js';

const PORT = Number(process.env.PORT || 3001);

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { cors: true });

  await app.listen(PORT);
  // eslint-disable-next-line no-console
  console.log(`[OpsService] listening on http://localhost:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`[OpsService] PERSISTENCE_MODE=${process.env.PERSISTENCE_MODE || 'off'}`);
}

bootstrap();
