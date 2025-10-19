# Canon data storage for locations and nations

This project stores structured canon for places and political entities in the
primary Postgres tables that back the lore systems. The columns below are the
sources that should be used when assembling canonical rules or summaries.

## Location canon

Canonical attributes for locations live on the `Locations` table. They are
persisted as JSONB arrays so downstream consumers can safely treat the values
as lists.

| Column | Shape | Canonical content |
| --- | --- | --- |
| `notable_features` | `JSONB` array | Distinct environmental or narrative features for the location. |
| `hidden_aspects` | `JSONB` array | Secrets or concealed traits that matter for canon enforcement. |
| `access_restrictions` | `JSONB` array | Rules that gate entry or behavior inside the location. |
| `local_customs` | `JSONB` array | Cultural expectations, rituals, or etiquette unique to the area. |

All of these columns are declared with a `DEFAULT '[]'::jsonb`, so callers can
expect a JSON list even when no values have been recorded yet.【F:db/schema_and_seed.py†L340-L373】

## Nation canon

Nation-level canon is recorded on the `Nations` table. The schema mixes JSONB
arrays for list-shaped data with a textual `notable_features` column that holds
a concise narrative description.

| Column | Shape | Canonical content |
| --- | --- | --- |
| `major_resources` | `JSONB` array | Key resources the nation controls. |
| `major_cities` | `JSONB` array | Important population centers and their descriptors. |
| `cultural_traits` | `JSONB` array | Cultural pillars, traditions, or societal norms. |
| `neighboring_nations` | `JSONB` array | Canonical adjacency information for diplomacy/conflicts. |
| `notable_features` | `TEXT` | Free-form summary of standout national characteristics. |

As with the location fields, the JSONB columns default to `[]`, making them
safe to treat as lists without null-guard boilerplate.【F:db/schema_and_seed.py†L1098-L1124】

When enriching lore contexts or generating canonical rules, prefer these columns
so that structured canon stays synchronized across systems.
