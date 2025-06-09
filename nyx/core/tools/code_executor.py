import asyncio
import os
import tempfile
import shutil
from agents import function_tool


@function_tool
async def execute_python(code: str, tests: str, timeout: int = 300) -> dict:
    """Execute Python code and tests using pytest.

    Parameters
    ----------
    code : str
        Source code to execute.
    tests : str
        Pytest-style tests importing the code as ``solution``.
    Returns
    -------
    dict
        Dictionary with 'passed' bool and captured output.
    """
    with tempfile.TemporaryDirectory() as tmp:
        code_file = os.path.join(tmp, 'solution.py')
        test_file = os.path.join(tmp, 'test_solution.py')
        with open(code_file, 'w') as cf:
            cf.write(code)
        with open(test_file, 'w') as tf:
            tf.write(tests)
        cmd = ['python', '-m', 'pytest', '-q', tmp]
        timeout_cmd = shutil.which('timeout')
        if timeout_cmd:
            cmd = [timeout_cmd, str(timeout)] + cmd
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            return {
                'passed': False,
                'stdout': stdout.decode(),
                'stderr': 'Timeout',
                'returncode': -1,
            }
        return {
            'passed': proc.returncode == 0,
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'returncode': proc.returncode,
        }
