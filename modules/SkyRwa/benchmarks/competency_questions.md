# Competency Questions

These questions define the minimum query capabilities the SkyRwa knowledge
graph must support.  Each question has a corresponding `.rq` SPARQL file.

## CQ1 — Tradable Assets

**Question:** Which flight evidence records can be promoted to tradable asset candidates?

**File:** `queries/competency/cq_01_tradable_assets.rq`

**Expected:** Assets with `isTradable = true` in their rights profile.

---

## CQ2 — Desensitization Required

**Question:** Which asset candidates must be desensitized before external licensing?

**File:** `queries/competency/cq_02_assets_requiring_desensitization.rq`

**Expected:** Assets where `requiresDesensitization = true` AND `isTradable = true`.

---

## CQ3 — Assets by Scenario / Product Lineage

**Question:** Which route-optimization data products derive from which flight evidence?

**File:** `queries/competency/cq_03_assets_by_scenario.rq`

**Expected:** Products of class `route_optimization_sample` with their source candidates and evidence.

---

## CQ4 — Revenue by Participant

**Question:** How much settled revenue has each operator earned across all assets?

**File:** `queries/competency/cq_04_revenue_by_participant.rq`

**Expected:** Aggregated `amount` per `partyId` and `role` from settlement records.

---

## CQ5 — Invalid Assets (Governance Violations)

**Question:** Which objects are marked as settlement-ready but lack a rights profile?

**File:** `queries/competency/cq_05_invalid_assets.rq`

**Expected:** Assets in `settlement_ready` status without a `hasRightsProfile` link.

---

## CQ6 — Multi-Source Product Lineage

**Question:** Which data products are aggregated from multiple flight candidates with complete lineage?

**File:** `queries/competency/cq_06_product_lineage.rq`

**Expected:** Products where `aggregatesCandidate` count > 1, each candidate having `derivedFromEvidence`.
