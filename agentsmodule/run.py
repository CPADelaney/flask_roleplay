import types
from .models.interface import ModelResponse

DEFAULT_MAX_TURNS = 10

class Runner:
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
        resp = await starting_agent.model.get_response(
            system_instructions=context,
            input=input,
            model_settings=None,
            tools=None,
            output_schema=None,
            handoffs=None,
            tracing=None,
            previous_response_id=previous_response_id,
        )
        final_output = ""
        if resp.output:
            msg = resp.output[0]
            if hasattr(msg, 'content') and msg.content:
                text_elem = msg.content[0]
                final_output = getattr(text_elem, 'text', '')
        return types.SimpleNamespace(final_output=final_output)
