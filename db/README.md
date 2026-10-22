# Database

## Files

- `schema.sql`: the Module 4 base schema (CIS 576, June 2026), captured from the Supabase Schema Visualizer on 2026-10-18. Eight tables: `users`, `properties`, `review_batches`, `reviews`, `analyses`, `authenticity_flags`, `flag_reviews`, `audit_log`.
- `policies.sql`: row level security state at capture time, reconstructed as DDL.
- `migrations/`: v2 changes, numbered, applied on top of `schema.sql`. `0001_detector_signals.sql` adds the embedding and classifier signal types, a numeric `score` and a `details` JSON column on flags, and the foreign key indexes.

## What the capstone design got right

- Raw reviews, derived signals, and generated outputs are stored separately, so authenticity findings are auditable later instead of recomputed and lost.
- Every flag carries a `rationale`, a `confidence` band, and a review `status` (unconfirmed, disputed, reviewed, dismissed): a signal for human review, not a verdict.
- An `audit_log` tied to users, batches, and flags.
- Row level security enabled on every table.

## Known gaps, to be fixed in the v2 schema

1. RLS was enabled on all eight tables but policies were written only for `properties`. The other seven tables are inaccessible to API roles until policies exist. v2 scopes batches, reviews, analyses, and flags through property ownership, and adds reviewer and admin access.
2. `authenticity_flags.signal_type` allows only the three capstone signals (`duplicate_phrasing`, `timing_cluster`, `rating_burst`). Addressed by migration 0001: `embedding_similarity` and `deceptive_language` added.
3. `confidence` is a text band (low, medium, high). Addressed by migration 0001: the band stays, `score` (0 to 1) and `details` (the reasoning) are added.
4. `reviews` has no source platform, external id, or language. v2 adds them so CSV imports can be de-duplicated and non English reviews handled fairly (a bias named in the Module 6 audit).
5. No indexes beyond primary keys, and no vector column. Foreign key indexes added by migration 0001; the pgvector embedding column on `reviews` comes with the storage step.
6. `properties.property_type` includes `restaurant`; v2 keeps it, the detector is not lodging specific.

The base schema is kept unchanged as the starting point; v2 changes land as migrations.
