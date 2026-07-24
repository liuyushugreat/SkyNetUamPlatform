# Archiving the Submission Snapshot on Zenodo

The paper's Data Availability statement promises an archived, DOI-citable
snapshot of the exact artifact version used in the article. The snapshot is
the git tag **`v1.0-jws-submission`**. Follow these steps once to mint the
DOI, then paste it into the manuscript.

## 1. Enable the GitHub–Zenodo integration

1. Log in at <https://zenodo.org> (use GitHub sign-in so the account link is automatic).
2. Open <https://zenodo.org/account/settings/github/>.
3. Find `liuyushugreat/SkyNetUamPlatform` in the repository list and flip the toggle to **On**.
   - The toggle must be enabled *before* the release is created; Zenodo only archives releases made after enabling.

## 2. Create a GitHub release from the tag

The tag already exists. Create a release from it either in the web UI
(*Releases → Draft a new release → choose tag `v1.0-jws-submission`*) or with
the GitHub CLI:

```bash
gh release create v1.0-jws-submission \
  --title "JWS submission snapshot (SkyRwa artifact v1.0)" \
  --notes "Exact artifact version evaluated in the JWS submission: ontology 1.0.0, 12 CQs, dual-engine SHACL validation, 105-flight benchmark (seed 42). See modules/SkyRwa/README.md for the paper-to-code mapping."
```

Zenodo detects the release within a few minutes and archives a snapshot
automatically.

## 3. Retrieve the DOI

1. Go to <https://zenodo.org/account/settings/github/>, click the repository name; the new upload appears with a DOI badge.
2. Zenodo mints **two** DOIs: a *version DOI* (this exact release) and a *concept DOI* (always resolves to the latest version). **Use the version DOI in the paper** — the Data Availability statement promises the exact version.

## 4. Finish the Zenodo record

On the Zenodo upload page, edit the metadata before publishing:

- **Title:** `SkyRwa: Modeling Governable Flight-to-Asset Lifecycles with Knowledge Graphs, SHACL, and Provenance (artifact)`
- **Authors:** fill in the paper's author list (TODO: authors to complete)
- **License:** Apache License 2.0
- **Related identifiers:** add the GitHub repo URL as `isSupplementTo`, and the paper DOI once assigned.
- **Keywords:** Knowledge Graph, Ontology, SHACL, Provenance, Data Governance, Ontology Design Pattern

## 5. Update the manuscript

Replace the placeholder in the paper's Data Availability statement
(`DOI: TODO_ZENODO_DOI` in `main.tex`) with the version DOI, e.g.
`10.5281/zenodo.XXXXXXX`, and recompile.

## Note on repository scope

Zenodo archives the whole `SkyNetUamPlatform` repository (the module is not a
standalone repo). This is acceptable — the paper points readers to
`modules/SkyRwa/` — but if a leaner archive is preferred, upload a zip of
`modules/SkyRwa/` manually via *New upload* instead of the GitHub integration
(the tag still documents the exact commit).
