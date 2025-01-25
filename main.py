# main.py 

from flask import Flask
# from routes.meltdown import meltdown_bp
from routes.new_game import new_game_bp
from routes.player_input import player_input_bp
from routes.settings_routes import settings_bp
from routes.knowledge_routes import knowledge_bp
from routes.story_routes import story_bp
from logic.memory_logic import memory_bp
from logic.rule_enforcement import rule_enforcement_bp
from db.admin import admin_bp
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Allow cross-origin requests globally

    # If you want a single connection per request:
    # from db.connection import get_db_connection

    # @app.before_request
    # def before_request():
    #     g.db_conn = get_db_connection()

    # @app.teardown_appcontext
    # def teardown_db_context(error):
    #     conn = getattr(g, 'db_conn', None)
    #     if conn:
    #         conn.close()

    # Register your blueprint modules
    app.register_blueprint(meltdown_bp, url_prefix="/meltdown")
    app.register_blueprint(new_game_bp, url_prefix="/new_game")
    app.register_blueprint(player_input_bp, url_prefix="/player")
    app.register_blueprint(settings_bp, url_prefix="/settings")
    app.register_blueprint(memory_bp, url_prefix="/memory")
    app.register_blueprint(rule_enforcement_bp, url_prefix="/rules")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(knowledge_bp, url_prefix="/knowledge")
    app.register_blueprint(story_bp, url_prefix="/story")
    return app

app = create_app()

if __name__ == "__main__":
    my_app = create_app()
    # If you want, set host/port here or rely on environment variables
    my_app.run(host="0.0.0.0", port=5000, debug=True)
