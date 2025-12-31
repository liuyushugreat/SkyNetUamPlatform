import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import neo4j, { Driver, Session } from 'neo4j-driver';

@Injectable()
export class Neo4jService implements OnModuleInit, OnModuleDestroy {
  private driver: Driver;

  constructor() {
    const uri = process.env.NEO4J_URI || 'bolt://localhost:7687';
    const user = process.env.NEO4J_USER || 'neo4j';
    const password = process.env.NEO4J_PASSWORD || 'skynet123';

    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, password));
  }

  async onModuleInit() {
    // 验证连接
    try {
      await this.driver.verifyConnectivity();
      console.log('[Neo4j] Connected successfully');
    } catch (error) {
      console.error('[Neo4j] Connection failed:', error);
      throw error;
    }
  }

  async onModuleDestroy() {
    await this.driver.close();
    console.log('[Neo4j] Connection closed');
  }

  getDriver(): Driver {
    return this.driver;
  }

  getSession(): Session {
    return this.driver.session();
  }

  async executeQuery<T = any>(
    query: string,
    parameters?: Record<string, any>
  ): Promise<T[]> {
    const session = this.getSession();
    try {
      const result = await session.run(query, parameters);
      return result.records.map((record) => record.toObject());
    } finally {
      await session.close();
    }
  }

  async executeWrite<T = any>(
    query: string,
    parameters?: Record<string, any>
  ): Promise<T[]> {
    const session = this.getSession();
    try {
      const result = await session.writeTransaction((tx) =>
        tx.run(query, parameters)
      );
      return result.records.map((record) => record.toObject());
    } finally {
      await session.close();
    }
  }
}

