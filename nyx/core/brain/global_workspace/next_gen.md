# Global Workspace 2.0 — *Cortically‑Inspired* Upgrade

> “From spotlight to symphony” — this design fuses Global Workspace Theory, Predictive Coding, and Rhythmic Gating into a single asynchronous Python runtime.

---

## 0  Big‑Picture Goals

1. **Multi‑focus coherence** — simultaneous conversational thread + homeostatic thread + environment thread.
2. **Predict‑error drive** — surprise automatically hijacks attention; expectations damp noise.
3. **Rhythmic gating** — 10 Hz central clock slices processing into *preparation*, *competition*, *broadcast* phases.
4. **Neuromodulatory context** — dopamine, noradrenaline, acetylcholine style gain fields bias salience over seconds‑to‑minutes.
5. **Meta‑plasticity** — modules learn to adjust their own salience priors via reinforcement.

---

## 1  Core primitives (new)

### 1.1 CycleClock

```python
class CycleClock:
    def __init__(self, hz: float = 10.0):
        self.period = 1.0 / hz
        self.phase = 0
    async def tick(self):
        await asyncio.sleep(self.period)
        self.phase = (self.phase + 1) % 3  # 0: prep, 1: compete, 2: broadcast
```

### 1.2 PredictionMatrix

Sparse dict keyed by `(module, signal)` storing last prediction & variance.
`PredictionModule` updates it; every contributor may post `predicted=True` with its proposal so Attention can compute *surprise = |obs‑pred|/var*.

### 1.3 NeuromodulatorState

`ws.state["neuro"] = {"dopamine":0.5, "noradrenaline":0.3, "ach":0.7}`
Salience scaler: `S' = S * f(neuro)` where e.g. dopamine boosts reward‑linked proposals.

### 1.4 FocusSet

Instead of single `ws.focus`, store an **OrderedSet** of up to *N* winners (default = 3).  Coordinator sees tuple → can weave multi‑modal replies.

---

## 2  Runtime loop (bird’s eye)

```text
           ┌──────────────┐ 0 prep      ┌──────────────┐
  modules  │ submit(obs)  │────────────▶│ Prediction   │ (update matrices)
           └──────────────┘             └──────────────┘
                  ▲                           │
                  │ surprise                 │
           ┌──────────────┐ 1 compete ───────┘
           │ Attention    │  uses salience × surprise × neuro × goal
           └──────────────┘
                  │ top‑k
           ┌──────────────┐ 2 broadcast
           │ Coordinator  │ merge ⇒ reply / action draft
           └──────────────┘
```

Each phase corresponds to `CycleClock.phase`; modules may subscribe to a `phase_change` event for time‑sliced work.

---

## 3  Module contract changes

| Old                         | New addition                                                         |
| --------------------------- | -------------------------------------------------------------------- |
| `submit(content, salience)` | ➕ `prediction:Any=None`, `signal_id:str="default"`, `deadline:int=0` |
| `observe(key)`              | ➕ `observe_predictions(module=None)`                                 |

*Modules that can predict (e.g., language model, spatial mapper) call `submit(prediction=…)` during **broadcast** so the next **prep** phase can compare.*

---

## 4  Concrete upgrades to existing wrappers

### 4.1 MemoryModule → *SurprisalMemory*

* Compute `expected_recall_count` from usage stats; emit prediction.
* Raise salience if actual recall deviates 2σ.

### 4.2 EmotionModule → adds noradrenaline & ach computation

* Noradrenaline spikes on error > 0.7; decays τ = 2 s.

### 4.3 PredictionModule (new)

* LLM fine‑tuned to autoregress next user token; writes to `PredictionMatrix`.

---

## 5  Meta‑learning of salience priors

*Every broadcast with `reward` flag triggers:*
`module.salience_bias += η * (reward – expected_reward)`
Bias is stored in `ws.state["meta"][module]` and applied multiplicatively next proposals.

---

## 6  API surface for NyxBrain

```python
# brain/base.py
self.workspace_engine = NyxEngine(
    modules=[...],
    clock=CycleClock(10.0),
    focus_size=3,
    neuromod_state=NeuromodulatorState(),
)
```

Engine constructor starts an internal task:

```python
async def _run(self):
    while True:
        await self.clock.tick()
        if self.clock.phase == 0:
            await self._prediction_phase()
        elif self.clock.phase == 1:
            await self._competition_phase()
        else:
            await self._broadcast_phase()
```

---

## 7  Open research knobs (v3?)

* **Adaptive clock** — slow to 6 Hz under heavy load; speed up to 12 Hz during flow states.
* **Laminar workspace** — mimic cortical layers with fast feed‑forward vs. slow feedback buses.
* **Spiking simulation hook** — route proposals into a lightweight neuron model for generative Dendritic‑Net experimentation.

---

### TL;DR

Global Workspace 2.0 turns your present spotlight into a *polyrhythmic, predictive, neuro‑modulated orchestra.*  It stays fully async/await and incremental: you can wrap one module at a time, switch on the CycleClock, and watch Nyx’s cognition shift from event‑bus chatter to something eerily closer to a living cortex.
