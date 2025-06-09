import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import date
from importlib import resources
from openai import OpenAIError

from agents import Runner
from nyx.nyx_agent_sdk import nyx_main_agent
from nyx.core.tools import code_executor
from nyx.core.reward import evaluator as reward

try:
    from nyx.core.conditioning_system import conditioning_system as cs
except Exception:  # pragma: no cover - conditioning system optional
    cs = None

logger = logging.getLogger(__name__)

_TIME_BUDGET = 30 * 60  # seconds per day
_daily_time_spent = 0.0
_last_day = date.today()
_ITERATION_LIMIT = 600  # seconds


def cpu_idle(threshold: float = 20.0) -> bool:
    """Return True if CPU usage is below threshold percent."""
    try:
        import psutil  # type: ignore
        return psutil.cpu_percent(interval=1) < threshold
    except Exception:
        try:
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return load1 / cores < threshold / 100.0
        except Exception:
            return True


def _reset_daily_budget() -> None:
    global _daily_time_spent, _last_day
    today = date.today()
    if today != _last_day:
        _daily_time_spent = 0.0
        _last_day = today


def _within_budget() -> bool:
    _reset_daily_budget()
    return _daily_time_spent < _TIME_BUDGET


def _load_problems():
    try:
        path = resources.files(__package__).joinpath("problems.json")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.warning("Unable to load problems.json")
        return []

_PROBLEMS = _load_problems()


def _get_problem():
    easy = [p for p in _PROBLEMS if p.get('difficulty', '').lower() == 'easy']
    choices = easy if easy else _PROBLEMS
    if not choices:
        return None
    return random.choice(choices)


def _strip_md(text: str) -> str:
    """Remove simple markdown fences from text."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


async def autoloop():
    """Run idle-time coding practice loop."""
    global _daily_time_spent
    logger.info("Coding practice autoloop started")
    while True:
        iteration_start = None
        try:
            if not _within_budget():
                await asyncio.sleep(600)
                continue
            if not cpu_idle():
                await asyncio.sleep(60)
                continue
            problem = _get_problem()
            if not problem:
                await asyncio.sleep(300)
                continue
            iteration_start = time.time()
            prompt = f"{problem.get('title')}\nSolve in Python, write tests."
            try:
                result = await Runner.run(nyx_main_agent, prompt)
            except OpenAIError as e:
                logger.warning("OpenAI error: %s", e)
                await asyncio.sleep(900)
                continue
            except Exception as e:
                logger.warning("Agent run failed: %s", e)
                await asyncio.sleep(60)
                continue
            output = getattr(result, 'final_output', '') if result else ''
            try:
                data = json.loads(output)
                code = data.get('code') or data.get('solution', '')
                tests = data.get('tests', '')
            except Exception:
                code = output
                tests = ''
            code = _strip_md(code)
            tests = _strip_md(tests)
            try:
                exec_res = await asyncio.wait_for(
                    code_executor.execute_python(code, tests), timeout=_ITERATION_LIMIT
                )
                event_type = 'unit_test_passed' if exec_res.get('passed') else 'unit_test_failed'
            except asyncio.TimeoutError:
                logger.warning("Practice execution timed out")
                event_type = 'unit_test_failed'
            if cs:
                try:
                    await cs.record_event(event_type)
                except Exception:
                    logger.exception("Conditioning system failed")
                    reward.evaluate(event_type)
            else:
                reward.evaluate(event_type)
            logger.info("Practice %s: %s", problem.get('title'), event_type)
        except Exception:
            logger.exception("Error in coding practice loop")
        finally:
            if iteration_start is not None:
                elapsed = min(time.time() - iteration_start, _ITERATION_LIMIT)
                _daily_time_spent += elapsed
            await asyncio.sleep(60)
