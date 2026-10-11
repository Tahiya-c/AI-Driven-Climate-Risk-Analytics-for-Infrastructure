-- =============================================================================
-- Port Pile Deterioration: Supabase schema, version 2 (no login, Port Kembla)
--
-- What it stores (shared: every upload is visible to everyone using the app):
--   * the saved models and how well they score
--   * every import: the file, its validation issues, its piles, inspections,
--     both models' forecasts, and per-pile results
--   * maintenance/replacement flags
--
-- Access: row-level security is ON with no policies, so the public API is
-- closed. Only the Streamlit server, using the secret (service_role) key, can
-- read or write. Keys never reach the browser. See docs/database_plan.md §7.
--
-- How to run: Supabase dashboard -> SQL Editor -> New query -> paste ALL of this
-- -> Run. Safe to run again: it only creates what doesn't exist yet.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Model registry: one row per saved model file, plus its scores
-- -----------------------------------------------------------------------------

create table if not exists public.model_versions (
  model_version    text primary key,              -- e.g. timber_transfer_mlp_20261004_143534
  material         text not null check (material in ('timber', 'concrete')),
  model_kind       text not null,                 -- gbm | mlp | rnn | wc_regress | ridge
  saved_at         timestamptz,
  trained_on       text,                          -- e.g. 'PileA' or '49 series, 7 piles'
  library_versions jsonb not null default '{}'::jsonb,
  params           jsonb not null default '{}'::jsonb,
  metadata         jsonb not null default '{}'::jsonb,  -- the artifact's full .json
  is_active        boolean not null default true,       -- the version the app uses now
  created_at       timestamptz not null default now()
);

-- Only one active version per material + model kind.
create unique index if not exists model_versions_one_active
  on public.model_versions (material, model_kind) where is_active;

create table if not exists public.model_evaluations (
  id             bigint generated always as identity primary key,
  model_version  text not null references public.model_versions (model_version) on delete cascade,
  evaluation     text not null,   -- e.g. pileB_all_years, pileB_forecast_a1fi, pileA_holdout,
                                  --      known_pile, unseen_pile
  asset_id       text,            -- the pile scored (e.g. 'PileB'); null = pooled over piles
  n_rows         integer,
  r2             double precision,
  rmse_mm        double precision,
  mae_mm         double precision,
  notes          text,
  created_at     timestamptz not null default now(),
  unique nulls not distinct (model_version, evaluation, asset_id)
);


-- -----------------------------------------------------------------------------
-- 2. Imports: one row per uploaded file
-- -----------------------------------------------------------------------------

create table if not exists public.imports (
  import_id      uuid primary key default gen_random_uuid(),
  material       text not null check (material in ('timber', 'concrete')),
  site           text not null default 'Port Kembla',
  file_name      text,
  storage_path   text,            -- <material>/<import_id>/<file_name> in the 'imports' bucket
  uploaded_by    text,            -- optional name typed on upload (there is no login)
  status         text not null default 'validated'
                 check (status in ('invalid', 'validated', 'forecast', 'failed')),
  n_errors       integer not null default 0,
  n_warnings     integer not null default 0,
  climate_input  jsonb,           -- the uploaded climate sheets, kept for reproducibility
  created_at     timestamptz not null default now()
);
create index if not exists imports_created on public.imports (material, created_at desc);

-- Every validation problem, exactly as the validator reports it.
create table if not exists public.import_issues (
  id            bigint generated always as identity primary key,
  import_id     uuid not null references public.imports (import_id) on delete cascade,
  level         text not null check (level in ('error', 'warning')),
  sheet         text,
  sheet_row     integer,          -- 1 = first row under the header
  sheet_column  text,
  message       text not null
);
create index if not exists import_issues_import on public.import_issues (import_id);


-- -----------------------------------------------------------------------------
-- 3. The piles in each import (one table per material: their columns differ)
-- -----------------------------------------------------------------------------

create table if not exists public.timber_assets (
  import_id                uuid not null references public.imports (import_id) on delete cascade,
  asset_id                 text not null,
  description              text,
  timber_type              text,
  year_installed           integer not null,
  zone                     text not null check (zone in ('A', 'B', 'C', 'D', 'E', 'F', 'G')),
  initial_diameter_mm      double precision not null check (initial_diameter_mm > 0),
  sapwood_layer_mm         double precision not null check (sapwood_layer_mm >= 0),
  replacement_diameter_mm  double precision not null check (replacement_diameter_mm >= 0),
  k_heartwood              double precision not null check (k_heartwood > 0),
  k_sapwood                double precision not null check (k_sapwood > 0),
  k_shelter                double precision not null check (k_shelter > 0),
  k_protect                double precision not null check (k_protect > 0),
  k_contact                double precision not null check (k_contact > 0),
  k_knot                   double precision not null check (k_knot > 0),
  primary key (import_id, asset_id)
);

create table if not exists public.concrete_assets (
  import_id  uuid not null references public.imports (import_id) on delete cascade,
  asset_id   text not null,
  zone       text not null check (zone in ('atmospheric', 'splash', 'tidal')),
  wc_ratio   double precision not null check (wc_ratio > 0),
  fc_mpa     double precision not null check (fc_mpa > 0),
  primary key (import_id, asset_id)
);


-- -----------------------------------------------------------------------------
-- 4. Inspections (the measured history of each pile)
-- -----------------------------------------------------------------------------

create table if not exists public.timber_observations (
  import_id              uuid not null,
  asset_id               text not null,
  inspection_year        integer not null,
  remaining_diameter_mm  double precision not null check (remaining_diameter_mm >= 0),
  notes                  text,
  primary key (import_id, asset_id, inspection_year),
  foreign key (import_id, asset_id)
    references public.timber_assets (import_id, asset_id) on delete cascade
);

create table if not exists public.concrete_observations (
  import_id      uuid not null,
  asset_id       text not null,
  year           integer not null,
  carb_depth_mm  double precision not null check (carb_depth_mm >= 0),
  primary key (import_id, asset_id, year),
  foreign key (import_id, asset_id)
    references public.concrete_assets (import_id, asset_id) on delete cascade
);


-- -----------------------------------------------------------------------------
-- 5. Forecasts: one run per model per import, and every forecast point
-- -----------------------------------------------------------------------------

create table if not exists public.forecast_runs (
  run_id         uuid primary key default gen_random_uuid(),
  import_id      uuid not null references public.imports (import_id) on delete cascade,
  model_version  text not null references public.model_versions (model_version),
  method         text not null,      -- gbm | mlp | rnn | wc_regress | ridge
  pathway        text,               -- timber: A1FI or A1B; concrete: null (all scenarios)
  anchored       boolean not null default false,
  anchor         jsonb,              -- offsets applied, window, any offsets over 15 mm
  created_at     timestamptz not null default now(),
  unique nulls not distinct (import_id, method, pathway, anchored)
);
create index if not exists forecast_runs_import on public.forecast_runs (import_id);

-- The same columns as the dashboard CSVs (material comes from the import).
create table if not exists public.forecast_points (
  run_id    uuid not null references public.forecast_runs (run_id) on delete cascade,
  asset_id  text not null,
  scenario  text not null,           -- e.g. 'A1FI|CSIRO Mk3.5'
  year      integer not null,
  value     double precision not null,
  unit      text not null default 'mm',
  metric    text not null,           -- deterioration_depth | carbonation_depth
  method    text not null,           -- observed | gbm | gbm_anchored | rnn | ...
  primary key (run_id, asset_id, scenario, method, year)
);


-- -----------------------------------------------------------------------------
-- 6. Per-pile performance and outcomes for each run
-- -----------------------------------------------------------------------------

-- How well the model matches this pile's own inspections.
create table if not exists public.pile_results (
  run_id            uuid not null references public.forecast_runs (run_id) on delete cascade,
  asset_id          text not null,
  n_inspections     integer not null default 0,
  first_inspection  integer,
  last_inspection   integer,
  fit_mae_mm        double precision,  -- model vs inspections, before anchoring
  fit_rmse_mm       double precision,
  anchor_offset_mm  double precision,  -- shift applied; null if not anchored
  large_offset      boolean not null default false,  -- shift over 15 mm: model fits this pile poorly
  primary key (run_id, asset_id)
);

-- The headline numbers per pile and climate scenario.
create table if not exists public.scenario_outcomes (
  run_id          uuid not null references public.forecast_runs (run_id) on delete cascade,
  asset_id        text not null,
  scenario        text not null,
  value_2050      double precision,
  final_year      integer,
  final_value     double precision,
  threshold_year  integer,   -- timber: year the diameter reaches replacement_diameter_mm
  primary key (run_id, asset_id, scenario)
);


-- -----------------------------------------------------------------------------
-- 7. Maintenance / replacement flags (today they only last for a session)
--    Shared like everything else: a flag on a pile is seen by everyone.
-- -----------------------------------------------------------------------------

create table if not exists public.pile_flags (
  id          bigint generated always as identity primary key,
  material    text not null check (material in ('timber', 'concrete')),
  asset_id    text not null,           -- a reference pile (e.g. 'PileB') or an imported one
  import_id   uuid references public.imports (import_id) on delete cascade,  -- null = reference pile
  purpose     text not null check (purpose in ('Maintenance', 'Replacement')),
  value_mm    double precision not null check (value_mm >= 0),
  created_at  timestamptz not null default now(),
  unique nulls not distinct (material, asset_id, import_id, purpose, value_mm)
);


-- -----------------------------------------------------------------------------
-- 8. A view in the dashboard CSV format, for charts and CSV downloads
-- -----------------------------------------------------------------------------

create or replace view public.forecast_points_dashboard
with (security_invoker = on) as
select
  i.material,
  p.asset_id,
  p.scenario,
  p.year,
  p.value,
  p.unit,
  p.metric,
  p.method,
  r.run_id,
  r.import_id,
  r.model_version,
  i.created_at as imported_at
from public.forecast_points p
join public.forecast_runs r on r.run_id = p.run_id
join public.imports i on i.import_id = r.import_id;


-- -----------------------------------------------------------------------------
-- 9. Access: row-level security ON, no policies
--    The public (anon/publishable) key gets nothing. The app's server uses the
--    secret key, which bypasses row security. Add policies later if a login
--    system is added.
-- -----------------------------------------------------------------------------

do $$
declare
  t text;
begin
  foreach t in array array[
    'model_versions', 'model_evaluations', 'imports', 'import_issues',
    'timber_assets', 'concrete_assets', 'timber_observations',
    'concrete_observations', 'forecast_runs', 'forecast_points',
    'pile_results', 'scenario_outcomes', 'pile_flags'
  ] loop
    execute format('alter table public.%I enable row level security', t);
  end loop;
end
$$;


-- -----------------------------------------------------------------------------
-- 10. File storage: a private bucket for the original uploads
--     (private + no policies = only the secret key can read or write it)
-- -----------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('imports', 'imports', false)
on conflict (id) do nothing;


-- -----------------------------------------------------------------------------
-- 11. Starting data: the models in use today and their published scores
--     Sources: docs/timber_model_updated.md (Summary, §5, §12.6) and
--     docs/concrete_model_comparison.md / docs/concrete_rnn_tuning.md.
--     The concrete version names are placeholders until the sync script
--     (Phase 4 of the plan) reads them from the artifacts' own metadata.
-- -----------------------------------------------------------------------------

insert into public.model_versions (model_version, material, model_kind, saved_at, trained_on, params)
values
  ('timber_transfer_gbm_20261004_143534', 'timber', 'gbm', '2026-10-04T14:35:34Z', 'PileA',
   '{"n_estimators": 25, "learning_rate": 0.05, "max_depth": 2, "num_leaves": 4, "linear_tree": true}'),
  ('timber_transfer_mlp_20261004_143534', 'timber', 'mlp', '2026-10-04T14:35:34Z', 'PileA',
   '{"hidden_layer_sizes": [136, 136, 136], "activation": "relu", "batch_size": 128}'),
  ('concrete_rnn_tuned_v1', 'concrete', 'rnn', null, '49 series, 7 piles',
   '{"window": 6, "hidden": 14, "lr": 0.0196, "weight_decay": 0.00757, "patience": 23}'),
  ('concrete_ridge_v1', 'concrete', 'ridge', null, '49 series, 7 piles', '{"alpha": 1.0}')
on conflict (model_version) do nothing;

insert into public.model_evaluations
  (model_version, evaluation, asset_id, n_rows, r2, rmse_mm, mae_mm, notes)
values
  ('timber_transfer_gbm_20261004_143534', 'pileB_all_years',    'PileB', 497,  0.926, 25.18, 17.52, 'trained on PileA only'),
  ('timber_transfer_gbm_20261004_143534', 'pileB_forecast_a1fi', 'PileB', 132,  0.969,  7.2,   4.3,  '2027-2070, 3 scenarios'),
  ('timber_transfer_gbm_20261004_143534', 'pileA_holdout',       'PileA', 98,  -1.102, 28.72, 28.18, 'last 20% of years'),
  ('timber_transfer_mlp_20261004_143534', 'pileB_all_years',    'PileB', 497,  1.000,  1.27,  0.74, 'trained on PileA only'),
  ('timber_transfer_mlp_20261004_143534', 'pileB_forecast_a1fi', 'PileB', 132,  1.000,  0.7,   0.4,  '2027-2070, 3 scenarios'),
  ('timber_transfer_mlp_20261004_143534', 'pileA_holdout',       'PileA', 98,   0.985,  2.39,  1.67, 'last 20% of years'),
  ('concrete_rnn_tuned_v1', 'known_pile',  null, null,  0.799, 0.134, 0.103, '2027-2070 rollout, 49 series'),
  ('concrete_rnn_tuned_v1', 'unseen_pile', null, null,  0.341, 0.313, 0.260, 'leave-one-pile-out, 7 folds'),
  ('concrete_ridge_v1',     'known_pile',  null, null, -5.77,  0.829, 0.639, 'deliberate worst-case comparison'),
  ('concrete_ridge_v1',     'unseen_pile', null, null, -7.89,  null,  null,  'leave-one-pile-out, 7 folds')
on conflict (model_version, evaluation, asset_id) do nothing;


-- -----------------------------------------------------------------------------
-- 12. Check it worked (run these on their own afterwards)
-- -----------------------------------------------------------------------------
-- select table_name from information_schema.tables where table_schema = 'public' order by 1;
-- select model_version, evaluation, r2, mae_mm from public.model_evaluations order by 1, 2;
-- select tablename, rowsecurity from pg_tables where schemaname = 'public' order by 1;
