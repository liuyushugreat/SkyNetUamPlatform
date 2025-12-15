# SkyNetUAM: Lifecycle-Aware Low-Altitude UAM Operations Platform

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FXXX.2025.XXXXXXX-blue)](https://doi.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

> **Official implementation** of our Drones submission (2025): a mission-lifecycle-aware operational platform for scalable low-altitude UAM/drone operations.  
> Note: on-chain components are treated as an **optional audit/settlement extension** and do not change the core operational logic.

## 📖 Overview

**SkyNetUAM** is a lifecycle-aware operations platform for low-altitude UAM/drone missions. It models missions as first-class operational entities (Created → Scheduled → Active → Completed/Failed/Delayed), enabling consistent state propagation across scheduling, monitoring, reporting, and (optionally) durable state persistence for audit/settlement.

### Key Features
*   **Mission lifecycle management**: deterministic state machine with event-driven transitions and timestamped records.
*   **Operational dashboards (frontend demo)**: citizen booking, operator monitoring, and regulator oversight views.
*   **Operational State Service (backend)**: NestJS service that ingests mission events and maintains consistent lifecycle state.
*   **Optional persistence adapter**: can be enabled as an asynchronous extension for auditability/settlement-style workflows (kept out of the critical operational path).

## 🏗️ System Architecture

The system is designed around a lifecycle-aware operational core, with optional persistence as a non-blocking extension.

```mermaid
graph TD
    A[UAM Operator] -->|Booking Request| B(SkyNet Platform Frontend)
    B -->|Mission Events| C[Operational State Service (NestJS)]
    C -->|State Updates| B
    C -->|Optional Async Persistence| D[(Persistence Adapter)]
```

## 🚀 Getting Started

### Prerequisites
*   Node.js v18+
*   (Optional) Python 3.10+ for reproducible experiments

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/liuyushugreat/SkyNetUamPlatform.git
    cd SkyNetUamPlatform
    ```

2.  **Install dependencies**
    ```bash
    npm install
    cd backend && npm install
    ```

3.  **Run the backend Operational State Service (optional but recommended)**
    ```bash
    cd backend
    npm run dev
    ```

4.  **Run the frontend**
    ```bash
    cd ..
    npm run dev
    ```

## 🧪 Experiments & Reproduction

To reproduce the daily 100k-mission workload (used to stress lifecycle management under congestion and permission constraints):

```bash
python experiments/simulate_100k_day.py
```

Outputs are written to `experiments/outputs/` (CSV + publication-ready plots).

## 📚 Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{Liu2025SkyNetUAM,
  title={SkyNetUAM: A Lifecycle-Aware Low-Altitude UAM Operations Platform},
  author={Liu, Yushu and Wang, Longbiao and Du, Chenglin and Zhai, Haixiao},
  journal={arXiv preprint arXiv:25XX.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
*Developed by the SkyNet Research Team.*
