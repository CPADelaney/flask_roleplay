Global Workspace Architecture for Nyx AI (Replacing Event Bus)
Overview and Objectives
The new unified architecture for Nyx AI is designed around a Global Workspace Theory (GWT) model, creating a central "blackboard" where all modules share information. This replaces the old event bus system
github.com
 with a more integrated, high-performance approach. Key goals of the new design include:
High Performance: Minimize inter-module overhead and latency by using direct async communication instead of broad event broadcasting.
Shared Context: Provide a single GlobalContextWorkspace where all modules can submit information and observe the overall state (replacing scattered events with a unified view).
Coordinated Decision-Making: Use an AttentionMechanism and Coordinator to select and merge module contributions, ensuring the AI’s response or action is coherent and optimized.
Meta-Cognition and Goal Alignment: Integrate a ReflectionLoop to catch incoherent outputs and a GoalManager for top-down guidance, improving consistency with goals and context.
All components will be implemented in a new subpackage (e.g. nyx/core/global_workspace/), allowing gradual migration of existing Nyx modules off the old integration event bus. The sections below describe each component and how they interoperate to form the new architecture.
GlobalContextWorkspace
The GlobalContextWorkspace is the centerpiece of the new architecture – a shared, thread-safe, async-compatible context store accessible by all modules. It functions as a global memory and communication medium, analogous to a “blackboard” where modules post their contributions. Key characteristics include:
Central State Store: It holds the global state that was previously scattered or passed via events. This includes persistent context (e.g. affective state, active goals, etc. similar to the old SystemContext
github.com
) and transient proposals from modules.
Thread-Safety & Async Support: Access to the workspace is protected by async locks to ensure consistency during concurrent reads/writes. Multiple modules can await workspace operations safely without race conditions. For example, adding a proposal or reading the current focus will acquire a lock only briefly to avoid contention.
Active State and Focus: The workspace tracks the current “focus of attention” – e.g., which piece of information is globally broadcast as most salient. It may maintain, for instance, a queue or list of recent proposals and a pointer to the currently selected item (as determined by the AttentionMechanism).
Implementation: The GlobalContextWorkspace can be a class with methods for modules and system components to interact with it. These include:
submit_proposal(proposal) – Add a new Proposal (module contribution) into the workspace.
get_proposals() – Retrieve current pending proposals (possibly filtered or sorted by salience).
set_focus(proposal) / get_focus() – Update or read the current focus of attention (the item chosen by the attention mechanism).
State accessors – Methods to update or retrieve global state (e.g., current goals, user context, affective state). These can wrap or incorporate the existing SystemContext state for continuity.
For clarity, a Proposal can be defined as a small data structure (e.g., a dataclass) containing at least: the content or suggestion, a salience score, a context tag (indicating relevant context or module), and possibly the source module’s name and a timestamp. Example:
python
Copy
Edit
@dataclass
class Proposal:
    source: str
    content: Any         # e.g., a suggested action, response text, etc.
    salience: float      # importance or relevance score
    context_tag: str     # category or context label for the proposal
    timestamp: float     # submission time (for recency)
And the core workspace might look like:
python
Copy
Edit
class GlobalContextWorkspace:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.proposals: List[Proposal] = []        # incoming contributions
        self.focus: Optional[Proposal] = None      # currently selected proposal (attention focus)
        # Integrate persistent state (e.g., SystemContext fields)
        self.active_goals: List[Dict[str, Any]] = []
        self.global_state = {}  # could include affective state, etc.
    
    async def submit_proposal(self, proposal: Proposal) -> None:
        """Add a module's proposal into the global workspace."""
        async with self._lock:
            self.proposals.append(proposal)
        # (Optionally signal attention mechanism that a new proposal is available)
    
    async def get_proposals(self) -> List[Proposal]:
        """Safely get a snapshot of current proposals."""
        async with self._lock:
            return list(self.proposals)
    
    def set_focus(self, proposal: Proposal) -> None:
        """Set the current attention focus to a specific proposal."""
        self.focus = proposal
    
    def get_focus(self) -> Optional[Proposal]:
        """Get the current focus of attention."""
        return self.focus
In the above snippet, we use an asyncio.Lock to protect modifications to the proposals list, ensuring thread-safe updates even in an async context. The design uses in-memory structures (lists, dicts) for speed, avoiding unnecessary I/O or heavy frameworks. The workspace could also include convenience methods to integrate with existing state (for example, if needed, it might wrap the old SystemContext or replicate its fields like active_goals, affective_state, etc., to provide a unified interface). Performance: By centralizing proposals in memory and using direct function calls/async awaits, we avoid the overhead of serializing events and invoking multiple subscriber callbacks as in the event bus. A single lock around critical sections (appending proposals, etc.) ensures minimal contention. Furthermore, we can use event-driven triggers – for instance, after a proposal is submitted, notify the AttentionMechanism (via an asyncio Event or direct call) so it can process immediately without polling.
AttentionMechanism
The AttentionMechanism is responsible for selecting high-salience information from the pool of module contributions in the GlobalContextWorkspace. This component implements the idea of focusing “global attention” on the most relevant pieces of data or suggestions, akin to making them conscious to the whole system. Responsibilities and Behavior:
Salience Evaluation: Continuously (or whenever triggered by new input) evaluate proposals in the workspace to determine which are most important. Each proposal comes with a salience score (possibly initially provided by the module or computed rudimentarily). The AttentionMechanism may adjust or normalize these scores based on factors like recency, module reliability, or relevance to current goals.
Selection of Focus: Pick the top proposal(s) to broadcast or focus on. In a simple implementation, this might select the single proposal with the highest salience. More advanced versions might select a small set of top-ranked items or maintain a global focus list.
Async Operation: Implemented as an asynchronous loop or callback, the attention mechanism can wake when new proposals arrive. For example, GlobalContextWorkspace.submit_proposal can signal an asyncio.Event that the AttentionMechanism awaits. This avoids busy-wait and ensures prompt attention shifts.
Implementation: This could be a class with an async method like process_new_proposals() or a long-running coroutine. A basic outline:
python
Copy
Edit
class AttentionMechanism:
    def __init__(self, workspace: GlobalContextWorkspace):
        self.workspace = workspace
    
    async def select_high_salience(self) -> Optional[Proposal]:
        """Evaluate proposals and select the most salient one."""
        proposals = await self.workspace.get_proposals()
        if not proposals:
            return None
        # Simple selection: max salience
        highest = max(proposals, key=lambda p: p.salience)
        # Optionally, could remove it from the list if we only handle once
        return highest
    
    async def run_attention_loop(self):
        """Continuously monitor and update the global focus."""
        while True:
            # Wait for a trigger or new proposals (could use an asyncio.Event or short sleep)
            # ... await some notification of new proposal ...
            top = await self.select_high_salience()
            if top:
                self.workspace.set_focus(top)
            # Yield control (e.g., short sleep or re-await event trigger)
            await asyncio.sleep(0.01)
In practice, we would integrate this loop with the system’s main event loop. The focus selected (stored via workspace.set_focus) will then be available for the Coordinator to integrate. Initially, the salience criteria can be trivial (e.g., trust the module’s provided score or use recency), but this structure allows later plugging in more complex attention scoring algorithms. Efficiency: Selecting the top proposal from a list is O(n) in the number of active proposals, which is typically small (and can be bounded or periodically cleared once handled). This is efficient and will not bottleneck the system. Because the mechanism runs asynchronously, it can yield to other tasks, maintaining responsiveness. By focusing only on high-salience data, we reduce unnecessary processing of low-relevance inputs, improving overall runtime performance.
Coordinator
The Coordinator is the decision integrator that merges suggestions and responses from all active modules to produce a unified outcome. Once the AttentionMechanism designates the important contributions (the “broadcast” information), the Coordinator combines them, resolving any conflicts, to determine the system’s next action or output. Key Functions:
Collect Focused Inputs: The Coordinator looks at the current focus of attention (and possibly a few secondary proposals) from the workspace. These are the candidates for inclusion in the final decision or response.
Weighted Consensus: Each proposal or module suggestion may carry a weight (e.g., salience or module priority). The Coordinator uses these weights to decide outcomes. For example, if multiple modules propose different actions, it might choose the one with highest weight, or if they propose compatible pieces (like parts of a sentence or plan), it could merge them.
Conflict Resolution: In cases of contradictory proposals (one module suggests doing X and another says do Y), the Coordinator applies rules or weights to resolve the conflict. This could involve deferring to a higher priority system (e.g., safety module vetoes unsafe action) or merging ideas if possible.
Implementation: The Coordinator can be an async component that is invoked whenever the focus is updated (meaning there's a new salient proposal to consider). It might expose a method like integrate_and_decide() which returns the final chosen action/response. For instance:
python
Copy
Edit
class Coordinator:
    def __init__(self, workspace: GlobalContextWorkspace, goal_manager: 'GoalManager'):
        self.workspace = workspace
        self.goal_manager = goal_manager
    
    async def integrate_and_decide(self) -> Any:
        """Merge high-priority proposals into a single decision/output."""
        focus = self.workspace.get_focus()
        if focus is None:
            return None  # nothing to do
        # In a basic form, just take the focused proposal as the decision:
        decision = focus.content
        # (Advanced: consider secondary proposals or goal alignment here)
        
        # Check alignment with top-level goals
        decision = await self.goal_manager.adjust_decision(focus, decision)
        return decision
In the simplistic placeholder above, the Coordinator simply uses the focus proposal’s content as the system’s decision. In a more complete implementation, it could look at other proposals that are still high salience and attempt to combine them. For example, if multiple modules provide complementary information (one suggests what to say, another suggests an emotion/tone to use), the Coordinator could merge them into a richer final output. The weighted consensus logic can also evolve. Initially, it might just pick the highest salience item. Later, it can incorporate module reliability weights or even do a voting system among proposals. The design should make it easy to plug in such logic (e.g., a method merge_proposals(list_of_proposals) that can be improved over time). Performance considerations: The Coordinator’s merging operation should be kept lightweight. By dealing only with the top few proposals (rather than hundreds of events), it reduces computation. Any heavy decision logic (e.g., natural language generation combining multiple inputs) can be optimized or offloaded if needed. The Coordinator primarily orchestrates calls to modules or uses simple rules, which is fast in Python. Also, since it's triggered only on relevant changes (not continuously looping without purpose), it avoids unnecessary CPU usage.
ReflectionLoop
The ReflectionLoop is a meta-cognitive layer that evaluates and possibly revises the system’s output before it is finalized. Its role is to catch incoherence, contradictions, or context-breaking elements in the Coordinator’s decision, functioning as a safety and consistency net. Functions and Flow:
Output Evaluation: When the Coordinator produces a decision or response (e.g., a sentence for the AI to speak, or an action to perform), the ReflectionLoop reviews it along with the current context. This could involve checking if the response contradicts known facts, violates character personality or goals, or breaks narrative consistency.
Feedback and Revision: If an issue is detected, the ReflectionLoop can flag it for revision. In a more advanced implementation, it might adjust the output (e.g., remove a problematic phrase) or request the Coordinator to consider an alternate proposal (perhaps the second-best option). It could also loop back and allow certain modules (like a reflection module or a consistency checker) to contribute a fix.
Iterative Refinement: The “loop” aspect implies it can iterate: for example, if the first output is flawed, apply a correction or choose another proposal, then validate again. To avoid infinite loops, there might be a limit (e.g., revise at most once or twice) or a confidence threshold under which it defaults to a safe fallback.
Implementation: The ReflectionLoop can be implemented as a function or coroutine that wraps around the final output production. For instance:
python
Copy
Edit
class ReflectionLoop:
    def __init__(self, workspace: GlobalContextWorkspace):
        self.workspace = workspace
        # possibly could use or incorporate a specialized module for reflection
    
    async def reflect_and_refine(self, decision: Any) -> Any:
        """Evaluate the decision and refine if needed."""
        # Simple placeholder: check for a known placeholder indicating an incoherence
        if isinstance(decision, str) and "<incoherent>" in decision:
            # In a real case, we'd detect actual incoherence patterns
            logging.debug("ReflectionLoop: Detected incoherence, requesting alternative.")
            # Remove or replace incoherent marker for now
            decision = decision.replace("<incoherent>", "")
            # (Could also choose to select another proposal or ask modules for clarification)
        # Additional checks (context-breaking references, contradictions) can go here
        return decision
In the initial iteration, this could be very basic – perhaps just a no-op that returns the decision unchanged, or a trivial check like the above. The key is that the structure is in place to incorporate more complex analyses later (such as using a knowledge base to ensure no contradiction with facts, or using a language model to self-critique the response before finalizing). Performance: Running a reflection step adds a slight overhead to each output cycle, but this can be minor if the checks are simple or only applied to final output text. Since it runs only once per cycle (not per module), it scales well. If more intensive checks are added (like an AI model to evaluate coherence), those can be optimized or made optional. The architecture allows this layer to be toggled or enriched as needed without affecting the lower-level modules.
GoalManager
The GoalManager is a top-down controller that maintains the AI’s current goals and influences the behavior of other components accordingly. This reflects the hierarchical control aspect of GWT, where high-level goals and intentions guide lower-level processes and attention. Role and Features:
Goal Tracking: The GoalManager holds a structured representation of active goals. For example, goals might include short-term objectives (answer the user’s question, maintain polite tone) and long-term ones (build trust with the user, follow the overarching narrative arc). This could be stored as a list or hierarchy of goal objects with attributes like id, description, priority, and status
github.com
.
Influencing Attention & Decisions: The GoalManager can affect how proposals are handled:
It might boost or lower salience of proposals based on how they align with current goals. (E.g., if the top goal is to ensure safety, and a module’s proposal might violate that, the GoalManager can reduce its salience or veto it.)
It can also inject top-down information into the workspace – for instance, if a new goal becomes active, it could post a high-salience proposal representing that goal (so that the attention mechanism will prioritize goal-related actions).
Hierarchical Control: In a hierarchical view, some modules could be working on sub-goals. The GoalManager ensures these sub-goals serve the active top-level goals. This might mean pausing or discarding proposals that are off-topic or creating new proposals to refocus modules when necessary.
Implementation: We can implement GoalManager as a singleton-like component accessible by others (especially the Coordinator and possibly AttentionMechanism). It may expose methods such as:
add_goal(goal) / remove_goal(goal_id)
get_active_goals()
evaluate_proposal(proposal) – returns an adjusted salience or an approval/denial based on goal alignment.
adjust_decision(proposal, decision) – modifies or comments on a final decision if it conflicts with goals.
For example:
python
Copy
Edit
class GoalManager:
    def __init__(self):
        self.active_goals: List[Dict[str, Any]] = []  # each goal could be a dict or a Goal object
    
    def add_goal(self, goal: Dict[str, Any]) -> None:
        self.active_goals.append(goal)
    
    def get_active_goals(self) -> List[Dict[str, Any]]:
        return list(self.active_goals)
    
    async def evaluate_proposal(self, proposal: Proposal) -> float:
        """Adjust a proposal's salience based on goal alignment."""
        # Placeholder: if proposal.context_tag matches a goal, boost its salience
        for goal in self.active_goals:
            if goal.get("id") in proposal.context_tag or goal.get("description") in proposal.content:
                return min(proposal.salience * 1.2, 1.0)  # boost by 20% up to max of 1.0
        return proposal.salience
    
    async def adjust_decision(self, proposal: Proposal, decision: Any) -> Any:
        """Optionally modify the final decision based on goals."""
        # Placeholder example: if top goal is "stay quiet" then cancel any speech
        if self.active_goals:
            top_goal = self.active_goals[0]
            if isinstance(decision, str) and "stay quiet" in top_goal.get("description", "").lower():
                return ""  # override decision to do nothing
        return decision
In this hypothetical snippet, evaluate_proposal might be called by the AttentionMechanism or Coordinator to tweak salience (ensuring goal-related proposals are favored). adjust_decision is used in the Coordinator example to final-check a decision against goals. Initially, these methods can be simplistic. The architecture’s aim is to have the hooks in place: as the system evolves, the GoalManager can become more sophisticated (e.g., multi-criteria goal matching, using a planner to break down goals, etc.). Integration: The GoalManager should be accessible to the Coordinator (to adjust decisions) and to the AttentionMechanism (to influence salience ranking). It might also be used by modules: e.g., a module could query the GoalManager for the current goals to contextually tailor its proposals. Being a singleton or a component passed to modules during initialization (like how we pass the workspace) can achieve this. Performance: Maintaining a list of goals and simple checks against them is very fast (few list/dict operations). Even as the logic grows, the number of active goals is usually limited, so this remains lightweight. Moreover, top-down filtering prevents wasted effort: modules won’t chase irrelevant directions for long if the GoalManager suppresses those, which in turn saves computation downstream.
Base Module Interface for Workspace Interaction
To facilitate all existing and new modules migrating to this architecture, we provide a base class or interface that standardizes how modules interact with the GlobalContextWorkspace. Instead of each module directly dealing with locks or events, the base class will offer clear methods (as hinted: submit, observe, sync, contribute) to encapsulate common patterns. Key methods to include:
submit_proposal(data, salience, context): A convenience method for modules to create a Proposal and submit it to the workspace. For example, module.submit_proposal("I suggest doing X", salience=0.8, context="planning") would internally construct the Proposal (filling in module name as source, etc.) and call workspace.submit_proposal(...).
observe_context(filter=None): Allows a module to read from the global context. This could fetch relevant information the module might need. A filter could be by context tag or type of data. For instance, a memory module might call observe_context(filter="recent_user_input") to get the latest user utterance from the shared context (assuming the input is stored there).
sync_state(): A method that a module can override to synchronize its internal state with the global context. The base implementation might do nothing or copy some common states. For example, if the workspace holds the authoritative “affective_state”, an EmotionModule’s sync_state could pull from that so the module is up-to-date.
contribute() (or run): This could be an abstract method that each module must implement, which defines the module’s behavior in each cycle. The system might call module.contribute() when it’s time for that module to generate a new proposal. Alternatively, modules could run in their own async tasks and use this when they have something to contribute. The base class could provide utility like a background loop that calls contribute() periodically or on certain triggers.
Structure: Likely we will define an abstract base class (using Python’s abc module or simple inheritance) that all modules (e.g., EmotionalCore, MemoryOrchestrator, ReasoningCore, etc.) can extend. Example sketch:
python
Copy
Edit
from abc import ABC, abstractmethod

class WorkspaceModule(ABC):
    def __init__(self, name: str, workspace: GlobalContextWorkspace):
        self.name = name
        self.workspace = workspace
    
    def submit_proposal(self, content: Any, salience: float, context_tag: str) -> None:
        prop = Proposal(source=self.name, content=content, salience=salience, context_tag=context_tag, timestamp=time.time())
        # Fire-and-forget submission (don't await, to not block module if not needed)
        asyncio.create_task(self.workspace.submit_proposal(prop))
    
    def observe_context(self, filter: Optional[str] = None) -> Any:
        # For simplicity, return entire context or filtered part (this could be extended)
        state = {
            "focus": self.workspace.get_focus(),
            "goals": getattr(self.workspace, "active_goals", []),
            # ...other global state as needed
        }
        if filter:
            return state.get(filter)
        return state
    
    def sync_state(self) -> None:
        """Sync module's internal state with global context (override if needed)."""
        return  # default: no action
    
    @abstractmethod
    async def contribute(self):
        """Generate contributions (to be implemented by concrete modules)."""
        pass
In this design, modules subclass WorkspaceModule and implement async def contribute(self). For example, a MemoryModule’s contribute might retrieve recent user input from observe_context, search its memory, and then submit_proposal with a relevant memory retrieval if appropriate. The base class’s submit_proposal ensures all proposals go through the central workspace uniformly. Module Operation: We can run each module’s contribute as a task. The Coordinator or a scheduler might trigger all modules to contribute at certain times (e.g., each turn or whenever new input arrives). Alternatively, modules could internally decide when to contribute (event-driven). The key is they use the provided interface to interact with the global store rather than emitting events or directly calling other modules. By using a base class approach:
We reduce code duplication (common logic for submission and context observation is in one place).
It’s easier to enforce the new communication pattern (no module should bypass the workspace).
We can later enhance the base class to add logging, tracing, or error handling for all modules uniformly.
Replacing the Event Bus and Integration Layer
With this architecture, the old event bus (nyx/core/integration/event_bus.py) and the various integration managers bridging subsystems will be phased out entirely. Modules no longer need to subscribe or publish events by type; instead, they share data through the GlobalContextWorkspace. Here’s how the transition is envisioned:
Direct Communication via Workspace: In the event bus system, a module would fire an Event and any listeners would react. Now, a module can directly submit its information to the workspace. All other components (and modules) have access to this shared info, either by actively observing it or by being invoked through the Coordinator/Attention pipeline. This eliminates duplicate message passing. For example, instead of event_bus.publish(Event("knowledge_updated", ...)) and other modules having subscribed handlers, the Knowledge module might directly update a knowledge entry in the workspace or submit a proposal about new knowledge, which the workspace and attention mechanism then handle globally.
Module Initialization: Previously, an IntegrationManager would subscribe module callbacks to the bus
github.com
. In the new system, during initialization, each module would be provided the global workspace (and possibly references to Coordinator or GoalManager if needed). No subscription calls – just ensure the module is known to the system. We might maintain a registry of modules if needed (for the Coordinator to iterate all modules to trigger contributions, for instance).
Main Loop or Trigger: We will likely implement a main loop or control flow in the new architecture that replaces the ad-hoc event handling. A possible sequence each "cycle" could be:
External input arrives (e.g., user input or environment event) – instead of sending an event, this gets placed into the GlobalContextWorkspace (perhaps as a special proposal or simply updating a context field for “current user input”).
All modules are triggered to run their contribute() once, so they produce any proposals relevant to this input or the current context. (This trigger could be done by a simple loop over modules, or by noticing new input in the workspace.)
AttentionMechanism selects the most salient of the newly submitted (and existing context) proposals.
Coordinator integrates them into a decision or output.
ReflectionLoop checks the output for issues; if okay, the output is emitted; if not, possibly adjust or loop back (maybe choosing the next-best proposal).
GoalManager is consulted throughout (it may have influenced salience in step 3 and final output in step 4).
The cycle ends; possibly clear out used proposals if they’re one-time, or keep some around if still relevant. Increase a cycle count in the workspace (similar to SystemContext.cycle_count
github.com
).
Backwards Compatibility: Initially, we might keep the event bus running in parallel for a short time, but ultimately the goal is to remove it entirely. The new system should cover all communication needs:
The SystemContext (global state) is effectively subsumed by GlobalContextWorkspace (which can hold the same data).
The event subscription patterns (like in NyxEventBusIntegration
github.com
) will be refactored so that, for example, instead of listening for a "decision_required" event, the Decision module’s logic will run during the contribute phase when a decision is needed (triggered by context).
Modules like ReasoningCognitiveBridge that subscribed to events
github.com
 could be refactored into perhaps a single Reasoning module that monitors context for changes (e.g., if knowledge is updated in context, it might then contribute a reasoning outcome). This is facilitated by the observe_context calls or by being directly invoked by the main loop when relevant.
By removing the intermediate event bus, we cut down on a lot of overhead from event object creation, queueing, and callback dispatching. Instead, data flows through shared memory (within the same process) and control is more straightforward (the Coordinator can directly call module methods or use results). This not only improves runtime performance but also simplifies debugging (less indirection than a pub/sub system).
Performance Considerations and Modern Best Practices
The new architecture is built with performance in mind, leveraging modern Python features for asynchronous concurrency and clean code structure:
AsyncIO Concurrency: All components (workspace, attention, coordinator, modules) use async/await for non-blocking operation. This allows the system to handle many tasks seemingly in parallel (e.g., multiple modules computing suggestions) without locking up threads. Unlike multi-threading, async tasks avoid context-switching overhead and shared-state race conditions are easier to manage with awaited locks.
Efficient Locking: We use minimal locking – for example, one lock around the critical section of proposal list manipulation in the workspace
github.com
. Most reads (like get_focus) don’t need locks if we design carefully (since setting focus is atomic assignment, reading it is OK if we accept eventual consistency within a cycle). If needed, a finer-grained lock or read-write lock can be used to allow concurrent reads of context while writes are protected. The goal is to avoid turning the global workspace into a bottleneck; thus, heavy operations are done outside locks.
Data Structures: In-memory Python data structures (lists, dicts, deque for history, etc.) are used for speed. No cross-process communication is involved, and we avoid deep copies where not necessary. For example, the workspace could keep proposals until they’re processed, then either clear them or move them to a history (like the event bus kept history, but we can cap or discard processed ones to free memory).
Modular and Extensible Design: Each component is in its own module file (for clarity and maintainability) and interacts via well-defined interfaces. This means future optimizations (e.g., replacing the AttentionMechanism’s algorithm with a more complex one, or even moving parts to C extensions for speed) can be done in isolation. The code follows Python best practices: type hints, clear class responsibilities, docstrings, and avoiding large monolithic functions.
Testing and Placeholders: Because some of these cognitive functions are complex, we start with simple placeholder implementations that are easy to test. For instance, initial attention selection might just pick the highest salience; we can write tests to ensure that works. As we refine the salience model, those tests guide correctness and performance checks. Similarly, ReflectionLoop can start as a no-op – ensuring it doesn’t break the flow – and later we add actual checks. By keeping these as separate units, we can write unit tests for each (e.g., feed a set of proposals into AttentionMechanism and verify the right one is chosen).
Avoiding Regression: During the transition, we ensure the new system can emulate the old behavior where needed. For example, if certain event types in the old bus had specific effects, we make sure the equivalent is achievable (maybe via proposals or context flags). This way, we can gradually port module by module to the new architecture, verifying performance at each step, rather than a big bang switch.
Gradual Adoption with Placeholder Implementations
Because the full intelligent behavior of attention, coordination, and reflection is complex, the initial implementation will include simplified logic to make the system operational quickly. These placeholders allow existing Nyx modules to start using the new framework without waiting for perfect algorithms. Here’s how each component can start simple:
AttentionMechanism (Placeholder): Initially, trust the module-provided salience scores or even ignore them by just taking the latest proposal as the focus. This ensures something flows through the pipeline. As a test, all modules could submit at least one proposal; the attention mechanism picks one (say, the first or highest salience = 1.0 if assigned) to focus on.
Coordinator (Placeholder): Start with a rule like “take the focus proposal as the decision”. No merging of multiple inputs, no complex conflict resolution. This essentially degrades to a single-winner system which is functionally similar to the event bus case where one module’s event triggers an action. We ensure the plumbing (calls to GoalManager and ReflectionLoop) is in place around this.
ReflectionLoop (Placeholder): Initially perform a trivial check or none at all. For example, just log that reflection was called, and return the decision unchanged. This ensures that adding the ReflectionLoop doesn’t change outputs yet, but the mechanism is ready to be fleshed out.
GoalManager (Placeholder): It can begin by simply holding goals without actively modifying anything. We might manually insert a couple of dummy goals to test the structure. Later, we implement real logic to use those goals. Even in placeholder form, it provides the interface for modules to query goals (even if just returns the list) and for Coordinator to call adjust_decision (which might not change anything initially).
Module Base Class & Modules: Implement the base class and migrate one or two modules as a proof of concept. For example, convert the Goal System or Emotion System to the new interface. In their contribute() methods, have them submit a simple proposal (e.g., Emotion module could always submit a proposal updating the affective state context, with low salience). This tests that modules can post to the workspace and nothing crashes. Over time, increase the complexity of their contributions.
By having these stubbed implementations, we achieve two things:
System Integration Testing: We can run the whole loop with minimal logic to ensure the components connect properly (e.g., a module posts something, attention picks it, coordinator returns it, reflection passes it through). This flushes out any integration bugs early (deadlocks, misordered operations, etc.).
Incremental Enhancement: With the scaffold in place, each part can be improved in isolation. For instance, once the baseline is working, one can focus on the AttentionMechanism – plug in a more sophisticated scoring function (perhaps considering recency and context match), and test it without affecting module code. Similarly, one can implement actual reflection rules (like scanning the output text for first-person consistency or contradictions against known facts) and simply drop that into the ReflectionLoop.
Throughout development, runtime performance will be monitored. Since the architecture emphasizes direct in-memory operations and async concurrency, we expect improvements over the older event bus. As more logic is added, we will profile critical sections (e.g., attention selection if many proposals, or goal evaluation if many goals) to ensure they remain efficient. If any part becomes a bottleneck, we can optimize (for example, if the number of proposals grows, using a priority queue for salience in AttentionMechanism could make selection O(log n) instead of O(n), etc.).
Conclusion
Nyx’s new unified architecture reimagines module interaction by using a Global Workspace model that is both cognitively inspired and performance-oriented. By replacing the event bus with a cohesive set of components – GlobalContextWorkspace, AttentionMechanism, Coordinator, ReflectionLoop, and GoalManager – we achieve a system where modules work in concert more naturally. Each module can contribute its expertise to a shared context, the most important information is automatically brought to the forefront, and decisions are made with holistic awareness of the AI’s state and goals. This design lays a scalable foundation for future enhancements: we can expand on attention scoring, improve consensus algorithms, and integrate learning or meta-reasoning in the ReflectionLoop, all while the core communication infrastructure remains stable. Developers can incrementally port subsystems to this new framework, confident that it will maintain or improve performance and lead to more coherent and goal-aligned AI behavior. Ultimately, this GWT-inspired architecture will make Nyx’s AI brain more unified, intelligent, and efficient, moving beyond the limitations of the old event-driven model into a truly integrated cognitive system.
