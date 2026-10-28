# Assayer

**A plain-language dataset requirement becomes reproducible synthetic data, proven by executed checks, delivered as a CC0 package with its evidence.**

An assay is a deterministic test that proves what something contains. That is the whole idea here: a model is never asked to vouch for data it produced. Code produces the data, checks prove it, and a human accepts it.

## The four rules

Every design decision in this repository traces to one of these.

1. **The model plans; it never writes data rows.** The plan is a JSON object constrained by a schema — row counts, distributions, constraints, checks. Not a single value in the delivered dataset comes out of a model.
2. **Seeded code generates.** Same plan + same seed ⇒ same bytes. The generator is ordinary Python with an explicit PRNG seed, so a run is replayable by anyone.
3. **Deterministic checks prove.** Every quality claim is a check result bound to the SHA-256 of the artifact it was computed over. A claim with no check behind it does not appear.
4. **A human accepts.** No model approves its own output. The critic at stage 7 is advisory and cannot change a pass/fail verdict.

Two consequences that are easy to violate by accident, so they are enforced rather than documented:

- **No raw source rows ever reach a model.** The profile stage passes summary statistics only. This is a leakage control and a licence-obligation control at once.
- **If repair attempts are exhausted, the run stops and reports the failing checks.** It never ships as passing.

## The lifecycle

```
request → source intake → profile → plan → generate → validate → critique → repair → review → release
            (hash,          (stats    (Meta   (seeded    (determ-   (open-     (bounded  (human   (CC0
             licence,         only)    model)  code)      inistic)   weight,    loop)     accept)  ZIP)
             attribution)                                            advisory)
```

Every stage appends one event to an append-only, hash-chained log, so a run can be replayed and a tampered record fails verification instead of passing quietly.

## Layout

| Path | What lives there |
|---|---|
| `engine/` | The lifecycle, the append-only log, content addressing |
| `profiles/hylogger/` | The first domain: reader, generator, checks for drill-hole mineralogy |
| `models/` | Model adapters — the only code that talks to a model |
| `review/` | The human accept / modify / reject page |
| `runs/` | Run artifacts, addressed by SHA-256 |

Adding a domain means adding a folder under `profiles/`. The engine does not change.

## Models

| Stage | Model | Why |
|---|---|---|
| Plan, Repair, Baseline | `muse-spark-1.3` | Schema-guaranteed structured output |
| Critique | Muse Glimmer 30B, open weights, run locally | An independent critic that does not share the planner's blind spots |

The planner is reached through the Muse Code CLI (`muse exec --output-schema`), which is schema-guaranteed on the meta provider and authenticates by account login. Token usage is recorded per call so a run can be costed.

## Domain: HyLogger drill-hole mineralogy

Source is the public NVCL WA catalogue (CC BY 4.0, attribution retained in every run manifest). Fields map from a TSG package: `depth_m` ← `Depth (m)`, `valid` ← `Final Mask == 1`, `swir_group` ← `Grp1 sTSAS`, `tir_group` ← `Grp1 sjCLST`.

These are **TSG interpretations, not fresh mineral identification**. Geological conclusions and resource estimation are out of scope.

## Status

Early. See `docs/` for the design and the running log of what is proven versus assumed.
