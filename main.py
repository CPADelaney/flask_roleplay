# main.py (or app.py)

from flask import Flask, g
from db.initialization import initialize_database
from routes.admin import admin_bp
from routes.new_game import new_game_bp
from routes.meltdown import meltdown_bp
from routes.settings_routes import settings_bp
from routes.player_input import player_input_bp

def create_app():
    app = Flask(__name__)

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
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(new_game_bp)
    app.register_blueprint(meltdown_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(player_input_bp)

    return app

if __name__ == "__main__":
    my_app = create_app()
    # If you want, set host/port here or rely on environment variables
    my_app.run(host="0.0.0.0", port=5000, debug=True)
