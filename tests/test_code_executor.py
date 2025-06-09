import pytest
from nyx.core.tools import code_executor

@pytest.mark.asyncio
async def test_execute_python_passes():
    code = "def add(a, b):\n    return a + b"
    tests = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    result = await code_executor.execute_python(code, tests)
    assert result['passed']


@pytest.mark.asyncio
async def test_execute_python_fails():
    code = "def mul(a, b):\n    return a * b"
    tests = "from solution import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 5\n"
    result = await code_executor.execute_python(code, tests)
    assert not result['passed']
