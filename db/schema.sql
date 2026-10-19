-- Review Synthesizer: Module 4 base schema (CIS 576, June 2026)
-- Source: Supabase Schema Visualizer "Copy as SQL", captured 2026-10-18.
-- Table DDL and constraints only. Row level security policies, indexes,
-- functions and triggers are captured separately in this folder.
-- Note from the export tool: table order and constraints may not be valid for execution as-is.

CREATE TABLE public.users (
  id uuid NOT NULL,
  email text,
  display_name text,
  role text NOT NULL DEFAULT 'operator'::text CHECK (role = ANY (ARRAY['operator'::text, 'reviewer'::text, 'admin'::text])),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT users_pkey PRIMARY KEY (id),
  CONSTRAINT users_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
CREATE TABLE public.properties (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  owner_id uuid NOT NULL,
  name text NOT NULL,
  property_type text CHECK (property_type = ANY (ARRAY['hotel'::text, 'inn'::text, 'restaurant'::text, 'other'::text])),
  region text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT properties_pkey PRIMARY KEY (id),
  CONSTRAINT properties_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.users(id)
);
CREATE TABLE public.review_batches (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL,
  submitted_by uuid NOT NULL,
  source_label text,
  review_count integer NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'analyzed'::text, 'failed'::text])),
  submitted_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT review_batches_pkey PRIMARY KEY (id),
  CONSTRAINT review_batches_property_id_fkey FOREIGN KEY (property_id) REFERENCES public.properties(id),
  CONSTRAINT review_batches_submitted_by_fkey FOREIGN KEY (submitted_by) REFERENCES public.users(id)
);
CREATE TABLE public.reviews (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  batch_id uuid NOT NULL,
  body text NOT NULL,
  star_rating integer CHECK (star_rating >= 1 AND star_rating <= 5),
  reviewer_handle text,
  posted_date date,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT reviews_pkey PRIMARY KEY (id),
  CONSTRAINT reviews_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES public.review_batches(id)
);
CREATE TABLE public.analyses (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  batch_id uuid NOT NULL,
  theme_map jsonb,
  aspect_scores jsonb,
  action_list jsonb,
  model_version text NOT NULL,
  generated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT analyses_pkey PRIMARY KEY (id),
  CONSTRAINT analyses_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES public.review_batches(id)
);
CREATE TABLE public.authenticity_flags (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  batch_id uuid NOT NULL,
  signal_type text CHECK (signal_type = ANY (ARRAY['duplicate_phrasing'::text, 'timing_cluster'::text, 'rating_burst'::text])),
  confidence text CHECK (confidence = ANY (ARRAY['low'::text, 'medium'::text, 'high'::text])),
  rationale text,
  status text NOT NULL DEFAULT 'unconfirmed'::text CHECK (status = ANY (ARRAY['unconfirmed'::text, 'disputed'::text, 'reviewed'::text, 'dismissed'::text])),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT authenticity_flags_pkey PRIMARY KEY (id),
  CONSTRAINT authenticity_flags_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES public.review_batches(id)
);
CREATE TABLE public.flag_reviews (
  flag_id uuid NOT NULL,
  review_id uuid NOT NULL,
  CONSTRAINT flag_reviews_pkey PRIMARY KEY (flag_id, review_id),
  CONSTRAINT flag_reviews_flag_id_fkey FOREIGN KEY (flag_id) REFERENCES public.authenticity_flags(id),
  CONSTRAINT flag_reviews_review_id_fkey FOREIGN KEY (review_id) REFERENCES public.reviews(id)
);
CREATE TABLE public.audit_log (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  event_type text NOT NULL,
  actor_id uuid,
  target_batch_id uuid,
  target_flag_id uuid,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT audit_log_pkey PRIMARY KEY (id),
  CONSTRAINT audit_log_actor_id_fkey FOREIGN KEY (actor_id) REFERENCES public.users(id),
  CONSTRAINT audit_log_target_batch_id_fkey FOREIGN KEY (target_batch_id) REFERENCES public.review_batches(id),
  CONSTRAINT audit_log_target_flag_id_fkey FOREIGN KEY (target_flag_id) REFERENCES public.authenticity_flags(id)
);
