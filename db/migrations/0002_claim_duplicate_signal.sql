-- Migration 0002: the claim duplicate signal from the LangGraph graph (step 5).
-- Applies on top of schema.sql and 0001.
--
-- * signal_type gains claim_duplicate: reviews that make the same specific claims in
--   different words, judged pair by pair by a language model. score is the share of
--   pairs in the group judged the same; details holds each pair judgment and its reason.

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
    'deceptive_language'::text,
    'claim_duplicate'::text
  ]));

commit;
