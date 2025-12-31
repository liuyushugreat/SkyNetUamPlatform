# RegulatorAPP (Not used)

This folder is **not** the location for the Regulator UI code.

## Where the Regulator frontend lives

Regulator (UI) is located at:
- `apps/regulator-app/`

## Industrial deployment note

The regulator app is planned to be **independently deployable** (network isolation)
and will call a dedicated **Regulator Gateway** rather than the internal `backend`
directly. See:
- `apps/regulator-app/adapters/README.md`


