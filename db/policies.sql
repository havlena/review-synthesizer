-- Review Synthesizer: row level security state of the Module 4 schema
-- Captured 2026-10-18 from pg_class / pg_policies on the Supabase project.
-- Reconstructed as DDL. All policies are permissive (the default).

alter table public.users              enable row level security;
alter table public.properties         enable row level security;
alter table public.review_batches     enable row level security;
alter table public.reviews            enable row level security;
alter table public.analyses           enable row level security;
alter table public.authenticity_flags enable row level security;
alter table public.flag_reviews       enable row level security;
alter table public.audit_log          enable row level security;

create policy "owner reads own properties"
  on public.properties for select to public
  using (owner_id = auth.uid());

create policy "owner writes own properties"
  on public.properties for insert to public
  with check (owner_id = auth.uid());

-- No other policies existed. With RLS enabled and no policy, the other seven
-- tables are unreadable and unwritable through the API roles (anon,
-- authenticated); only the service role bypasses RLS. See README.md.
