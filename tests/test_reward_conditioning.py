import pytest
from nyx.core.conditioning_system import ConditioningSystem


@pytest.mark.asyncio
async def test_reward_updates_associations():
    cs = ConditioningSystem()
    for _ in range(60):
        await cs.record_event("unit_test_passed")
    assert cs.classical_associations, "No associations formed"
    assert any(a.association_strength > 0 for a in cs.classical_associations.values())
