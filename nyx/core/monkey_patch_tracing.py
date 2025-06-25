# Monkey patch for OpenAI agents trace metadata
import json
from agents import RunConfig

# Store original __init__
_original_runconfig_init = RunConfig.__init__

def _patched_runconfig_init(self, **kwargs):
    # Stringify trace_metadata if present
    if 'trace_metadata' in kwargs and kwargs['trace_metadata']:
        stringified = {}
        for k, v in kwargs['trace_metadata'].items():
            if isinstance(v, str):
                stringified[k] = v
            elif isinstance(v, (int, float, bool)):
                stringified[k] = str(v)
            elif isinstance(v, (list, dict)):
                stringified[k] = json.dumps(v)
            elif v is None:
                stringified[k] = "null"
            else:
                stringified[k] = str(v)
        kwargs['trace_metadata'] = stringified
    
    # Call original init
    _original_runconfig_init(self, **kwargs)

# Apply monkey patch
RunConfig.__init__ = _patched_runconfig_init

print("Applied OpenAI trace metadata monkey patch")
