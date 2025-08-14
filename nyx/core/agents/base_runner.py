from agents.run import Runner, DEFAULT_MAX_TURNS
from .. import orchestrator


class OrchestratorMixin:
    async def prepare_context(self, ctx: str, user_msg: str) -> str:
        return await orchestrator.prepare_context(ctx, user_msg)

    async def log_and_score(
        self,
        event_type: str,
        payload: dict | None = None,
        user_id: int | None = None,
        conversation_id: int | None = None,
    ) -> float:
        return await orchestrator.log_and_score(
            event_type,
            payload,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def start_background(self) -> None:
        orchestrator.start_background()


class OrchestratorRunner(OrchestratorMixin, Runner):
    """Runner subclass that mixes in orchestrator helpers."""

    @classmethod
    async def run(
        cls,
        starting_agent,
        input,
        *,
        context=None,
        max_turns=DEFAULT_MAX_TURNS,
        hooks=None,
        run_config=None,
        previous_response_id=None,
    ):
        if isinstance(context, str) and isinstance(input, str):
            context = await orchestrator.prepare_context(context, input)
        orchestrator.start_background()
        result = await super().run(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
        )
        return result
