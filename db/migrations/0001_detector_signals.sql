-- Migration 0001: detector signals beyond the capstone rules.
-- Applies on top of schema.sql. Addresses gaps 2 and 3 in db/README.md.
--
-- * signal_type gains the two v2 signals: embedding_similarity (near duplicate
--   meaning) and deceptive_language (classifier). timing_cluster from the
--   capstone design is kept for compatibility; the rules emit rating_burst.
-- * score: the numeric value behind the confidence band (0 to 1). Jaccard or
--   cosine peak for the duplicate signals, probability for the classifier.
-- * details: the structured reasoning behind the flag (thresholds, model name,
--   cue terms), so a flag can be explained later without recomputing it.

begin;

alter table public.authenticity_flags
  drop constraint if exists authenticity_flags_signal_type_check;

alter table public.authenticity_flags
  add constraint authenticity_flags_signal_type_check
  check (signal_type = any (array[
    'duplicate_phrasing'::text,
    'timing_cluster'::text,
    'rating_burst'::text,
    'embedding_similarity'::text,
    'deceptive_language'::text
  ]));

alter table public.authenticity_flags
  add column if not exists score numeric(5, 4)
    check (score >= 0 and score <= 1);

alter table public.authenticity_flags
  add column if not exists details jsonb not null default '{}'::jsonb;

comment on column public.authenticity_flags.score is
  'Numeric value behind the confidence band, 0 to 1. Signal specific: similarity peak or classifier probability.';
comment on column public.authenticity_flags.details is
  'Structured reasoning for the flag: thresholds, model, cue terms. Mirrors Flag.details in the Python package.';

-- Foreign key lookups the application makes on every batch.
create index if not exists reviews_batch_id_idx on public.reviews (batch_id);
create index if not exists authenticity_flags_batch_id_idx on public.authenticity_flags (batch_id);
create index if not exists flag_reviews_review_id_idx on public.flag_reviews (review_id);
create index if not exists review_batches_property_id_idx on public.review_batches (property_id);
create index if not exists analyses_batch_id_idx on public.analyses (batch_id);

commit;
