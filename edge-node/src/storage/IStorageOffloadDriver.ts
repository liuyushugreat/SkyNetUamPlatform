/**
 * Interface for the Storage Offload Driver.
 * 
 * This interface defines the contract for high-performance storage operations
 * at the edge. In a production deployment, this would connect to a high-speed
 * NVMe-oF C++ driver via Foreign Function Interface (FFI) to bypass the OS kernel 
 * for zero-copy data transfer, ensuring low latency for heavy IO workloads.
 */
export interface IStorageOffloadDriver {
  /**
   * Writes a buffer to storage using zero-copy mechanism.
   * @param buffer The data buffer to write.
   */
  writeZeroCopy(buffer: Buffer): Promise<void>;

  /**
   * Reads a Key-Value chunk from storage.
   * @param key The key to look up.
   * @returns The data chunk associated with the key.
   */
  readKVChunk(key: string): Promise<Buffer>;
}
