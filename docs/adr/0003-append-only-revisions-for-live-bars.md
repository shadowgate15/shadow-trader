# Live Bars are append-only Revisions, sealed after a repair window

A live Bar closes by event time (a watermark from quote and heartbeat timestamps plus a 1 s grace). Within a repair window (default 1 h) a late or catch-up quote re-emits the slot as the next Revision; earlier Revisions are never overwritten or deleted. After the window the Bar is sealed and later quotes are dropped and counted. Every live quote is appended to a write-once raw log (source and arrival time), so both views can be rebuilt: as-seen (the Revision current at a moment, for the decision audit trail) and latest (highest Revision, for the running context window and for Datasets). Live Datasets are built explicitly from the raw log over sealed slots only.

## Considered Options

- **Strictly immutable emitted Bars**: rejected. A running agent reasons over recent history, so a Gap or wrong Bar left by a feed hiccup would corrupt later decisions with no way to repair it.
- **Mutable in place with background re-checking**: rejected. It destroys the record of what the agent actually saw, breaking the audit trail and Promotion path evidence.
- **Reconcile against a historical backfill**: not adopted as repair. Another source is a different feed and may legitimately differ, so it is an advisory diff report only; Datasets from the two sources are never merged.

## Consequences

- Repair only works where an adapter can supply the missed quotes (`fetch_quotes`); otherwise Gap Bars stay flagged `suspected_outage`.
- Consumers must choose a view; how the agent re-reads or distrusts its window belongs to the environment and live orchestration.
- Bars carry a `revision` in the live stream. Whether a Dataset row carries it is left to the interface-shape work, since Datasets hold only the latest.
