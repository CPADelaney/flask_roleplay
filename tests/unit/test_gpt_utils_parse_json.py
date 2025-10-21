import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_chatgpt_stub():
    module_name = "logic.chatgpt_integration"
    if module_name in sys.modules:
        return

    stub_module = ModuleType(module_name)

    async def _fake_get_chatgpt_response(*_args, **_kwargs):
        raise RuntimeError("stub should not be invoked")

    stub_module.get_chatgpt_response = _fake_get_chatgpt_response
    sys.modules[module_name] = stub_module


_install_chatgpt_stub()

from logic.gpt_utils import parse_json_str


def test_parse_json_str_handles_curly_quotes():
    curly_json = '{“name”: “Lena”, “tagline”: “Lena’s Journey”}'

    parsed = parse_json_str(curly_json)

    assert parsed == {"name": "Lena", "tagline": "Lena's Journey"}
