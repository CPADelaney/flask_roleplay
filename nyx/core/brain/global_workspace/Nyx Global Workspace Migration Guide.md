Nyx Global Workspace Migration Guide

PurposeThis guide shows how to move every existing Nyx module to the new Global Workspace Architecture (GWA) in one structured pass—no piecemeal slog.  It explains:

Core concepts & moving parts of GWA

How modules map to new roles (Contributor / Integrator / Monitor)

A categorical checklist for adapting each file under flask_roleplay/nyx/core

Concurrency, salience, and performance tips

Final wiring in NyxEngine

NotationWS = GlobalContextWorkspace    WM = WorkspaceModule base    AM = AttentionMechanism    CO = Coordinator

1  Global Workspace Recap

Component

Function

What modules do

WS

Async blackboard holding shared state & incoming Proposals

Call submit() to add data, observe() to read

AM

Picks highest‑salience proposals & sets WS.focus

No action; selection is automatic

CO

Combines focused data into final action/utterance

May invoke Integrator modules for merging

ReflectionLoop

Last‑pass coherence/safety check

Monitor modules can inject fixes

GoalManager

Top‑down bias / veto

Exposed via WS.state["goals"]

Proposal Anatomy

Proposal(source="emotion", content={"anger":0.8}, salience=0.7, context_tag="affect")

Higher salience → more chance to gain global focus that tick.

Module Roles in GWA

Role

Trait

Typical Nyx examples

Contributor

Produce domain data / suggestions

emotions, memory_core, prediction_engine, spatial

Integrator

Merge/compose multiple inputs

expression_system, reasoning_core, agentic_action_generator

Monitor

Post‑hoc validation / adjustment

harm_detection, conditioning_system, reflection_engine

A single module can implement multiple roles.

2  Adapting a Module – Template

Import the base

from nyx.core.global_workspace.global_workspace_architecture import WorkspaceModule

Subclass & store the workspace

class EmotionModule(WorkspaceModule):
    async def contribute(self):
        # read latest user input
        focus = self.observe("focus")
        if focus and focus.context_tag == "user_input":
            anger = self._infer_anger(focus.content)
            self.submit({"anger": anger}, salience=0.6, tag="affect")

Remove event‑bus calls (event_bus.publish / @subscribe).

Expose helper APIs with the module’s own async methods (optional).

Register the module when building NyxEngine.

3  Category‑by‑Category Checklist

A. Input / Attention layer

File

New Role

Notes

input_processor.py

Contributor & pre‑WS injector

Convert user text ⇒ Proposal(context_tag="user_input", salience=1.0)

attentional_controller.py

Delete (logic now inside AttentionMechanism)

context_awareness.py

Contributor

Post environment signals as proposals

B. Affect & Persona

Module file

Role

Migration hints

emotions/ emotional_core.py

Contributor

Compute valence/arousal ⇒ submit; cache recent result to avoid spam

mood_manager.py

Contributor (low rate)

Hourly mood drift → proposal with low salience

dominance.py, femdom_coordinator.py, interaction_mode_manager.py

Integrator & Monitor

Wrap text style rules inside contribute(); when WS.focus.context_tag=="reply_draft", rewrite style & boost/fix salience

C. Memory & Knowledge

File

Role

Integration

memory_core.py (now wrapped)

Contributor

Use provided memory_module.py wrapper – or implement similar pattern for knowledge_core.py / recognition_memory.py

experience_consolidation.py, reflection_engine.py

Monitor (background tasks)

Schedule with asyncio.create_task; periodically check WS.state & run maintenance

D. Cognition & Planning

File

Role

Integration

reasoning_core.py

Integrator

Wait until both user_input & relevant memory_recall visible; compose logical answer → submit reply_draft

goal_system.py

Contributor + writes to GoalManager

Update goals via ws.state["goals"]; post goal‑related proposals

internal_thoughts.py, theory_of_mind.py

Contributor

Surface chain‑of‑thought snippets with low salience, useful for reflection

prediction_engine.py

Contributor

Post predicted user next‑step (tag prediction)

E. Expression / Output

File

Role

Integration

expression_system.py

Integrator

Consume reply_draft, affect, persona proposals; assemble final reply → high salience outgoing_speech

body_image.py, spatial/navigator_agent.py

Contributor

Provide action proposals / location updates

F. Safety & Conditioning

File

Role

Integration

harm_detection.py

Monitor

Inspect reply_draft; if disallowed → submit veto proposal (salience=1.0, tag safety_veto)

conditioning_system.py, conditioning_maintenance.py

Monitor

Post‑process drafts to enforce style constraints

G. Orchestration Utilities

Remove or refactor:

integration/event_bus.py – delete after migration.

processing_manager.* – replace with NyxEngine.

Logging, tracing, distributed helpers remain but drop bus deps.

4  Concurrency & Performance Tips

Use async/await everywhere; never block the loop (no time.sleep).

Limit expensive calls (LLM, DB) with module‑local cool‑downs.

Cap WS.proposals list length; cleared each cycle by AttentionMechanism.

If two modules spam identical content, add deduplication inside their contribute().

5  Bootstrapping the Engine

from nyx.core.global_workspace.global_workspace_architecture import NyxEngine
from nyx.core.global_workspace.memory_module          import MemoryModule
from nyx.core.global_workspace.emotion_module         import EmotionModule
from nyx.core.global_workspace.reasoning_module       import ReasoningModule
from nyx.core.global_workspace.expression_module      import ExpressionModule
# … import the rest …

async def build_engine():
    modules = [
        MemoryModule(None),
        EmotionModule(None),
        ReasoningModule(None),
        ExpressionModule(None),
        # add others here …
    ]
    engine = NyxEngine(modules)
    await engine.start()
    return engine

NyxEngine injects its workspace into each module on start‑up; passing None for ws in constructors is fine if the module assigns later.

6  Migration Path Summary

Create wrapper files per category (see tables).

Delete event‑bus imports and convert logic to submit()/observe().

Register all modules in one place → NyxEngine.

Incrementally enhance salience & consensus scoring once functional.

Remove legacy integration/ folder after thorough tests.

Your entire codebase now shares a single cognitive stage—Nyx finally thinks with one mind.
