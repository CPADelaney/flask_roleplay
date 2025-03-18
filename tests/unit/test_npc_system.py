import pytest
from npcs.npc_creation import NPCCreationHandler
from npcs.npc_agent import NPCAgent
from memory.wrapper import MemorySystem

def test_npc_creation(mock_npc_data):
    """Test NPC creation functionality."""
    handler = NPCCreationHandler()
    npc = handler.create_npc(mock_npc_data)
    
    assert npc is not None
    assert npc.name == mock_npc_data["name"]
    assert npc.personality_traits == mock_npc_data["personality_traits"]
    assert npc.stats["intensity"] == mock_npc_data["stats"]["intensity"]

@pytest.mark.asyncio
async def test_npc_memory_integration(mock_npc_data, mock_memory_data):
    """Test NPC memory system integration."""
    # Create NPC
    handler = NPCCreationHandler()
    npc = handler.create_npc(mock_npc_data)
    
    # Initialize memory system
    memory_system = MemorySystem()
    await memory_system.initialize()
    
    # Create a memory
    memory_id = await memory_system.create_memory(
        content=mock_memory_data["content"],
        importance=mock_memory_data["importance"],
        tags=mock_memory_data["tags"],
        metadata=mock_memory_data["metadata"]
    )
    
    assert memory_id is not None
    
    # Verify NPC can access the memory
    memory = await memory_system.get_memory(memory_id)
    assert memory is not None
    assert memory.content == mock_memory_data["content"]
    
    # Cleanup
    await memory_system.delete_memory(memory_id)

def test_npc_stats_validation(mock_npc_data):
    """Test NPC stats validation."""
    handler = NPCCreationHandler()
    
    # Test valid stats
    npc = handler.create_npc(mock_npc_data)
    assert npc.stats["intensity"] == 50
    
    # Test invalid stats
    invalid_data = mock_npc_data.copy()
    invalid_data["stats"]["intensity"] = 150  # Above max
    
    with pytest.raises(ValueError):
        handler.create_npc(invalid_data)

@pytest.mark.asyncio
async def test_npc_belief_formation(mock_npc_data):
    """Test NPC belief formation system."""
    handler = NPCCreationHandler()
    npc = handler.create_npc(mock_npc_data)
    
    # Create a test belief
    belief = {
        "content": "The world is a simulation",
        "confidence": 0.7,
        "source": "observation",
        "timestamp": "2024-02-20T12:00:00Z"
    }
    
    # Add belief to NPC
    await npc.add_belief(belief)
    
    # Verify belief was added
    npc_beliefs = await npc.get_beliefs()
    assert len(npc_beliefs) > 0
    assert any(b["content"] == belief["content"] for b in npc_beliefs)

def test_npc_relationship_management(mock_npc_data):
    """Test NPC relationship management."""
    handler = NPCCreationHandler()
    npc1 = handler.create_npc(mock_npc_data)
    
    # Create second NPC
    npc2_data = mock_npc_data.copy()
    npc2_data["name"] = "Test NPC 2"
    npc2 = handler.create_npc(npc2_data)
    
    # Create relationship
    relationship = {
        "type": "alliance",
        "strength": 0.6,
        "trust": 0.7
    }
    
    npc1.add_relationship(npc2.id, relationship)
    
    # Verify relationship
    assert npc1.get_relationship(npc2.id) is not None
    assert npc1.get_relationship(npc2.id)["type"] == "alliance" 