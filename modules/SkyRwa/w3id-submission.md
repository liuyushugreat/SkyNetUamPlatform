# Registering `https://w3id.org/skyrwa#` — step-by-step

> **Status: COMPLETED.** The registration PR
> ([perma-id/w3id.org#6412](https://github.com/perma-id/w3id.org/pull/6412))
> was merged in July 2026; `https://w3id.org/skyrwa` now resolves with
> content negotiation, and `migrate_namespace.py --apply` has been executed
> (37 files, 12,582 occurrences migrated). The steps below are kept for
> reference.

w3id.org permanent identifiers are managed through pull requests against the
public repository [`perma-id/w3id.org`](https://github.com/perma-id/w3id.org).
Follow these steps once, then run `migrate_namespace.py` (see step 8).

## 1. Fork and clone

```bash
# fork https://github.com/perma-id/w3id.org in the GitHub UI, then:
git clone https://github.com/<your-user>/w3id.org.git
cd w3id.org
git checkout -b add-skyrwa
```

## 2. Create the identifier directory

```bash
mkdir skyrwa
```

## 3. Add the redirect rules

Copy `modules/SkyRwa/ontology/w3id/.htaccess` from this repository into the new
directory as `skyrwa/.htaccess` (content negotiation: Turtle / RDF/XML /
JSON-LD → GitHub raw files; HTML → documentation page; `/1.0.0` → pinned
version snapshot).

## 4. Add the README with administrative contact

Create `skyrwa/README.md`:

```markdown
# SkyRwa Flight-to-Asset Ontology

Permanent identifier for the SkyRwa ontology namespace
(`https://w3id.org/skyrwa#`), an OWL ontology for governance transitions
that turn UAM flight evidence into governed data products.

- Source repository: https://github.com/liuyushugreat/SkyNetUamPlatform/tree/main/modules/SkyRwa
- Contact: Yushu Liu <liuyushu@tju.edu.cn> (GitHub: @liuyushugreat)
```

w3id requires at least one responsible person with a GitHub handle in the
README so maintainers can verify future change requests.

## 5. Test the rewrite rules locally (optional but recommended)

```bash
# from the w3id.org repo root; requires Docker
./run-local-server.sh
curl -sIL -H "Accept: text/turtle"          http://localhost:8080/skyrwa | grep -i location
curl -sIL -H "Accept: application/rdf+xml"  http://localhost:8080/skyrwa | grep -i location
curl -sIL -H "Accept: application/ld+json"  http://localhost:8080/skyrwa | grep -i location
curl -sIL -H "Accept: text/html"            http://localhost:8080/skyrwa | grep -i location
```

## 6. Commit and open the pull request

```bash
git add skyrwa/
git commit -m "Add w3id for SkyRwa Flight-to-Asset Ontology"
git push -u origin add-skyrwa
```

Open a PR against `perma-id/w3id.org:master`. In the PR description state the
purpose (persistent namespace for a published OWL ontology accompanying a
journal article) and confirm you are the maintainer. Reviews are usually
handled within a few days; respond to maintainer comments from the same
GitHub account listed in the README.

## 7. Prerequisites on our side (do these before or in parallel)

- [ ] Enable **GitHub Pages** for `liuyushugreat/SkyNetUamPlatform` (Settings →
      Pages → deploy from `main`), otherwise the HTML redirect target
      `https://liuyushugreat.github.io/SkyNetUamPlatform/modules/SkyRwa/docs/ontology/index.html`
      returns 404. Alternative: change the two HTML rules in `.htaccess` to any
      other stable documentation URL before submitting.
- [ ] Keep `ontology/skyrwa.ttl`, `ontology/skyrwa.owl`,
      `ontology/skyrwa.jsonld` at their current paths — the `.htaccess` targets
      point at them on the `main` branch.

## 8. After the PR is merged

Run the namespace migration exactly once from `modules/SkyRwa/`:

```bash
python migrate_namespace.py          # dry run, prints planned replacements
python migrate_namespace.py --apply  # performs the rewrite
```

This rewrites `urn:skyrwa:ontology#` → `https://w3id.org/skyrwa#` across all
`.ttl`, `.py`, and `.rq` files, then regenerate the serializations and docs:

```bash
python docs/generate_ontology_docs.py
```

Finally re-run the evaluation scripts (the graphs embed namespace IRIs) and
commit everything as a single "migrate to w3id namespace" commit.
