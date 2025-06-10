class ModelResponse:
    def __init__(self, output=None, usage=None, response_id=None):
        self.output = output or []
        self.usage = usage
        self.response_id = response_id


class Model:
    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *, previous_response_id=None):
        raise NotImplementedError

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError
