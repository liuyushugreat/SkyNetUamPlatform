# GP Spark Simulation Environment
This module contains the discrete-event simulation scripts used to generate the experimental results for the paper 'Cloud-Edge-End Collaborative Data Storage Architecture'.

## Experiments
1. **Data Loading**: Simulates 1TB dataset loading over SATA, USB 3.0, and NVMe-oF (GP Spark).
2. **KV Cache Offloading**: Simulates inference latency when VRAM is exhausted (OOM vs. Swap).
3. **RWA Privacy**: Simulates CPU usage and throughput for software vs. hardware SHA-256 hashing.
