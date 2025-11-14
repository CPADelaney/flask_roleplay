import importlib.machinery
import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "nyx/nyx_agent/context.py"
NYX_PACKAGE_PATH = MODULE_PATH.parent.parent
NYX_AGENT_PATH = MODULE_PATH.parent

if 'nyx' not in sys.modules:
    nyx_pkg = types.ModuleType('nyx')
    nyx_pkg.__path__ = [str(NYX_PACKAGE_PATH)]
    nyx_pkg.__spec__ = importlib.machinery.ModuleSpec('nyx', loader=None, is_package=True)
    sys.modules['nyx'] = nyx_pkg

nyx_agent_pkg = types.ModuleType('nyx.nyx_agent')
nyx_agent_pkg.__path__ = [str(NYX_AGENT_PATH)]
nyx_agent_pkg.__spec__ = importlib.machinery.ModuleSpec('nyx.nyx_agent', loader=None, is_package=True)
sys.modules['nyx.nyx_agent'] = nyx_agent_pkg

spec = importlib.util.spec_from_file_location('nyx.nyx_agent.context', MODULE_PATH)
context = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = context
assert spec.loader is not None
spec.loader.exec_module(context)

BundleSection = context.BundleSection
ContextBundle = context.ContextBundle
LoreSectionData = context.LoreSectionData
SceneScope = context.SceneScope


def _empty_bundle_section():
    return BundleSection(data={})


def test_lore_compact_preserves_canonical_rules():
    rules = [f"Rule {i}" for i in range(5)]
    data = LoreSectionData(location={'description': 'desc'}, world={'facts': 'something'}, canonical_rules=rules)

    compacted = data.compact()

    assert compacted.canonical_rules == rules


@pytest.mark.parametrize('budget', [16, 128, 1024])
def test_packed_context_always_includes_canonical_rules(budget):
    rules = [f"Non-negotiable rule {i}" for i in range(6)]
    lore_section = BundleSection(
        data=LoreSectionData(
            location={'description': 'A location steeped in history.'},
            world={'factions': ['Guild']},
            canonical_rules=rules,
        )
    )

    bundle = ContextBundle(
        scene_scope=SceneScope(),
        npcs=_empty_bundle_section(),
        memories=_empty_bundle_section(),
        lore=lore_section,
        conflicts=_empty_bundle_section(),
        world=_empty_bundle_section(),
        narrative=_empty_bundle_section(),
    )

    packed = bundle.pack(token_budget=budget)

    assert 'lore_rules' in packed.canonical
    assert packed.canonical['lore_rules']['canonical_rules'] == rules
    packed_dict = packed.to_dict()
    assert packed_dict['lore_rules']['canonical_rules'] == rules
