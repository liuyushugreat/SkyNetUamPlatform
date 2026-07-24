# SkyRwa → Autonomous-Vehicle (AV) Port

This folder supports the paper claim (Sect.~8, "Generalizability and Semantic
Web Agents") that the SkyRwa lifecycle is **domain-parametric**: the contribution
is the *pattern*, not the UAM vocabulary.

## What changes vs. what is inherited

**Changed (evidence layer only, ~40 lines of Turtle):**

| UAM concept              | AV concept                   |
| ------------------------ | ---------------------------- |
| `skyrwa:FlightEvidence`  | `avport:DriveEvidence`       |
| `skyrwa:UAV`             | `avport:Vehicle`             |
| `skyrwa:FlightMission`   | `avport:Trip`                |
| `skyrwa:performedByUAV`  | `avport:performedByVehicle`  |
| `skyrwa:hasMission`      | `avport:hasTrip`             |
| `FlightEvidenceShape`    | `DriveEvidenceShape`         |

**Inherited unchanged:**

- `AssetCandidate`, `GovernanceDecision`, `GovernedDataProduct`,
  `RevenueRight`, `SettlementRecord`
- All governance rules `GOV-001` … `GOV-004`
- The typed-governance-transition definition and Prop. 1 (necessity of
  SHACL-SPARQL for conditional gates)

## Reproduce

```bash
python run_av_port.py
```

Expected output (written to `av_port_result.json`):

```json
{
  "ontology_triples": 302,
  "data_triples": 163,
  "self_inverse_violations": 0,
  "shacl_conforms": true
}
```
