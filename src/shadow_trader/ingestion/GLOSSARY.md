
**Warm-up**:
The stretch at the start of a Session, or of a Dataset, during which a history-dependent feature has not yet seen a full lookback inside it. Features are null throughout; a window never crosses a Session boundary.
_Avoid_: Burn-in, priming

**Usable Bar**:
A prepared Bar that is priced, carries no distrusted quality flag, and has every history-dependent feature present. Everything else is unusable, and a Bar records why. Raw rows are never changed to express this.
_Avoid_: Valid Bar, clean Bar
