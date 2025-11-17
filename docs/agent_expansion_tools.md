# Nyx Agent Expansion Tools

Nyx agents can request additional context mid-generation via registered tools. The
following helpers are wired into `nyx.nyx_agent.assembly.register_expansion_tools`
and are available to all Response agents:

| Tool Name        | Purpose                                      | Key Parameters |
|------------------|----------------------------------------------|----------------|
| `expand_npc`     | Pull detailed NPC dossiers (backstory, goals)| `npc_id`, `fields` |
| `get_more_memories` | Retrieve additional episodic memories     | `entity_ids`, `k` |
| `expand_lore`    | Warm lore bundles for entities/locations     | `entities`, `depth` |
| `expand_world`   | **New.** Calls `ContextBroker.expand_world()` to fetch
world-state aspects (time, location, tension) for a scope. | `entities`, `aspects`, `depth` |
| `check_world_state` | Snapshot specific aspects without heavy expansion | `aspects` |
| `get_conflict_details` | Canon conflict stakes + involvement     | `conflict_ids` |

`expand_world` should be used when the agent needs structured data beyond the
lightweight `check_world_state` snapshot (e.g., validating emergent choices or
prepping a governance report). Governance logging automatically records the
request metadata for auditing.
