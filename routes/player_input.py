# routes/player_input.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection

player_input_bp = Blueprint('player_input_bp', __name__)

@player_input_bp.route('/player_input', methods=['POST'])
def player_input():
    # ignoring user text if meltdown is active
    ...

