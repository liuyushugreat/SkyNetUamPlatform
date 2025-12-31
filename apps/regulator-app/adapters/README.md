# Regulator Adapters (Future)

This directory is reserved for the **Regulator** app's deployment boundary.

When `regulator-app` is deployed separately (inner/outer network isolation),
it should **not** call the internal `backend` directly. Instead it will call a
**Regulator Gateway** (API boundary), which then fans out to internal services.

Planned contents:
- API client creation for regulator gateway baseURL
- audit headers / trace correlation
- request timeout / retry policy hardened for gov network constraints


