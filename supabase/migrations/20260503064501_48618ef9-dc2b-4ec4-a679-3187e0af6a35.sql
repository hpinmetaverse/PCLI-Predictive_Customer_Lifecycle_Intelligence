-- Roles enum & table
create type public.app_role as enum ('admin', 'user');

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  full_name text,
  created_at timestamptz not null default now()
);
alter table public.profiles enable row level security;

create table public.user_roles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role app_role not null,
  created_at timestamptz not null default now(),
  unique (user_id, role)
);
alter table public.user_roles enable row level security;

create or replace function public.has_role(_user_id uuid, _role app_role)
returns boolean
language sql stable security definer set search_path = public
as $$
  select exists (select 1 from public.user_roles where user_id = _user_id and role = _role)
$$;

-- Predictions
create table public.predictions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  inputs jsonb not null,
  churn_probability numeric not null,
  lifetime_value numeric not null,
  shap_image text,
  gauge_image text,
  hazard_image text,
  survival_image text,
  created_at timestamptz not null default now()
);
alter table public.predictions enable row level security;

-- Profiles policies
create policy "users view own profile" on public.profiles for select using (auth.uid() = id);
create policy "admins view all profiles" on public.profiles for select using (public.has_role(auth.uid(),'admin'));
create policy "users update own profile" on public.profiles for update using (auth.uid() = id);
create policy "users insert own profile" on public.profiles for insert with check (auth.uid() = id);

-- Roles policies
create policy "users view own roles" on public.user_roles for select using (auth.uid() = user_id);
create policy "admins view all roles" on public.user_roles for select using (public.has_role(auth.uid(),'admin'));

-- Predictions policies
create policy "users view own predictions" on public.predictions for select using (auth.uid() = user_id);
create policy "admins view all predictions" on public.predictions for select using (public.has_role(auth.uid(),'admin'));
create policy "users insert own predictions" on public.predictions for insert with check (auth.uid() = user_id);

-- Auto profile + role on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email, full_name)
  values (new.id, new.email, new.raw_user_meta_data->>'full_name');

  if new.email = 'harshvardhanchauhan776@gmail.com' then
    insert into public.user_roles (user_id, role) values (new.id, 'admin');
  else
    insert into public.user_roles (user_id, role) values (new.id, 'user');
  end if;
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();