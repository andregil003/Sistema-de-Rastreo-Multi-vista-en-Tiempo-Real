-- ============================================================
-- ESQUEMA SUPABASE - Sistema de Reconocimiento Facial Multi-cámara
-- Pegar completo en: supabase.com > SQL Editor > New query > Run
-- ============================================================

-- Personas registradas desde el navegador (celular o PC)
create table if not exists public.personas (
  id uuid primary key default gen_random_uuid(),
  nombre text not null,
  codigo_registro text not null default '',
  creado_en timestamptz not null default now()
);

-- Embeddings faciales (vector de 128 dimensiones SFace)
create table if not exists public.rostros_entrenados (
  id bigint generated always as identity primary key,
  persona_id uuid not null references public.personas(id) on delete cascade,
  embedding jsonb not null,
  origen text not null default 'celular',        -- 'celular' | 'camara_laptop'
  imagen_url text,                                -- URL en Supabase Storage
  creado_en timestamptz not null default now()
);
create index if not exists idx_rostros_persona on public.rostros_entrenados(persona_id);

-- Cada reconocimiento / detección de desconocido
create table if not exists public.eventos_deteccion (
  id bigint generated always as identity primary key,
  persona_id uuid references public.personas(id) on delete set null,
  nombre_detectado text,
  camara text not null,
  confianza numeric(5,4),
  distancia_aprox_m numeric(6,2),
  tipo text not null default 'reconocimiento',   -- 'reconocimiento' | 'desconocido'
  creado_en timestamptz not null default now()
);
create index if not exists idx_eventos_creado on public.eventos_deteccion(creado_en desc);

-- Alertas de posible transmisión por teléfono (replay attack)
create table if not exists public.alertas_replay (
  id bigint generated always as identity primary key,
  camara text not null,
  captura_url text,
  detalle text,
  atendida boolean not null default false,
  creado_en timestamptz not null default now()
);

-- Row Level Security: el backend usa service_key (bypassa RLS).
-- El público NO puede leer ni escribir directamente.
alter table public.personas enable row level security;
alter table public.rostros_entrenados enable row level security;
alter table public.eventos_deteccion enable row level security;
alter table public.alertas_replay enable row level security;

-- Bucket de Storage para fotos de rostros y capturas de alertas
insert into storage.buckets (id, name, public)
values ('rostros-registrados', 'rostros-registrados', true)
on conflict (id) do nothing;
