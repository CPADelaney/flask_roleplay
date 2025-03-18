import os
import pytest
from flask import Flask
from main import create_flask_app
from db.connection import initialize_connection_pool, close_connection_pool

@pytest.fixture(scope="session")
def app():
    """Create and configure a Flask application for testing."""
    # Set environment to testing
    os.environ["FLASK_ENV"] = "testing"
    
    # Create app instance
    app = create_flask_app()
    
    # Configure app for testing
    app.config.update({
        "TESTING": True,
        "SERVER_NAME": "localhost.localdomain",
        "PREFERRED_URL_SCHEME": "http",
    })
    
    # Initialize test database
    with app.app_context():
        initialize_connection_pool()
    
    yield app
    
    # Cleanup
    with app.app_context():
        close_connection_pool()

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test CLI runner."""
    return app.test_cli_runner()

@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    return {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json"
    }

@pytest.fixture
def mock_npc_data():
    """Sample NPC data for testing."""
    return {
        "name": "Test NPC",
        "personality_traits": ["friendly", "intelligent"],
        "background": "A test NPC for automated testing",
        "goals": ["Help test the application"],
        "relationships": [],
        "stats": {
            "intensity": 50,
            "corruption": 0,
            "dependency": 0
        }
    }

@pytest.fixture
def mock_memory_data():
    """Sample memory data for testing."""
    return {
        "content": "Test memory content",
        "importance": 0.7,
        "emotional_valence": 0.5,
        "tags": ["test", "memory"],
        "metadata": {
            "source": "test",
            "timestamp": "2024-02-20T12:00:00Z"
        }
    } 