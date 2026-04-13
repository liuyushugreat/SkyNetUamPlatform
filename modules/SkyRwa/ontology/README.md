# SkyRwa Ontology

## Design Principles

1. **Minimal but complete** — every class and property needed to drive the KG and SPARQL queries is defined; nothing redundant is added.
2. **Standards-aligned** — classes and properties map explicitly to PROV-O, DCAT, ODRL, and Dublin Core / Schema.org.
3. **Layered lifecycle** — the ontology enforces a strict `FlightEvidence → AssetCandidate → GovernedDataProduct → RevenueRight` progression.

## Vocabulary Alignments

| SkyRwa Concept | Standard | Aligned Class / Property |
|---|---|---|
| FlightEvidence | PROV-O | `prov:Entity` |
| GovernanceDecision | PROV-O | `prov:Activity` |
| GovernedDataProduct | DCAT | `dcat:Dataset` |
| RightsProfile | ODRL 2.2 | `odrl:Policy` (semantic mapping) |
| flightId | Dublin Core | `dcterms:identifier` |
| startTime / endTime | Schema.org | `schema:startDate / endDate` |

## Core Classes

- `skyrwa:FlightEvidence` — raw attestation (not tradable)
- `skyrwa:AssetCandidate` — governed, valued candidate
- `skyrwa:GovernedDataProduct` — aggregated, tradable data product
- `skyrwa:RevenueRight` — revenue entitlement (on-chain mapping target)
- `skyrwa:UsageEvent`, `skyrwa:SettlementRecord` — revenue lifecycle
- `skyrwa:GovernanceDecision`, `skyrwa:ValuationExplanation` — audit trail
- `skyrwa:FlightMission`, `skyrwa:UAV`, `skyrwa:Operator`, `skyrwa:DataConsumer` — agents

## Core Properties

Object properties: `derivedFromEvidence`, `governedBy`, `promotedToProduct`, `hasUsageEvent`, `hasSettlementRecord`, `hasValuation`, `hasRightsProfile`, `hasLineage`, `aggregatesCandidate`, `operatedBy`, `performedByUAV`, `consumedBy`, `settledVia`.

Datatype properties: `flightId`, `uavId`, `startTime`, `endTime`, `hasDigest`, `hasSignature`, `hasAssetClass`, `hasStatus`, `complianceScore`, `riskScore`, `dataQualityScore`, `estimatedValue`, `isTradable`, `requiresDesensitization`, `grossAmount`, `sharePct`, `usageType`, `consumer`.

## Example Triples

```turtle
@prefix skyrwa: <urn:skyrwa:ontology#> .
@prefix inst:   <urn:skyrwa:> .

inst:evidence:EV001 a skyrwa:FlightEvidence ;
    skyrwa:flightId "FLT-001" ;
    skyrwa:hasDigest "sha256:abc123..." ;
    skyrwa:startTime "2026-01-15T08:00:00Z"^^xsd:dateTime .

inst:asset:AU001 a skyrwa:AssetCandidate ;
    skyrwa:derivedFromEvidence inst:evidence:EV001 ;
    skyrwa:hasAssetClass "route_optimization_sample" ;
    skyrwa:complianceScore "0.92"^^xsd:float .
```

## Files

| File | Purpose |
|---|---|
| `skyrwa.ttl` | Main ontology (classes, properties, annotations) |
| `prefixes.ttl` | Shared prefix declarations |
| `alignments.ttl` | Mappings to PROV-O, DCAT, ODRL, Schema.org |
