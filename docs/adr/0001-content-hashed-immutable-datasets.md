# Datasets are content-hashed, immutable directories with hardlinked history

A Dataset is a directory named by its content hash, partitioned by Instrument then year, with a manifest written last. The hash is a Merkle hash over the *logical* rows (canonical order and encoding) plus `schema_version` and bar size, never the Parquet bytes and never provenance. Appending or fixing data writes a new Dataset that records its `parent` and hardlinks unchanged closed partitions; nothing is edited in place. Ingest reads only from a write-once raw download cache, so re-ingesting the same raw inputs with the same rules reproduces the same hash.

## Considered Options

- **Hash Parquet file bytes**: rejected because writer versions and settings change the bytes without changing the data, breaking reproducibility checks.
- **Mutable Dataset with an append log**: rejected because a Candidate must name the exact Dataset it saw (see the Dataset and Candidate glossary entries).
- **Migrate old Datasets on schema change**: rejected; a schema change re-ingests from the raw cache into a new Dataset instead.

## Consequences

- Provenance (creation time, source fetch times, builder revision) lives in the manifest but is not hashed, so it cannot be used to tell two identical-content Datasets apart.
- The raw cache is never pruned automatically; it is the source of truth for reproducibility.
