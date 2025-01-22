# routes/settings_routes.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
import random, json

settings_bp = Blueprint('settings_bp', __name__)

@settings_bp.route('/generate_mega_setting', methods=['POST'])
def generate_mega_setting():
    # your code
    ...

@settings_bp.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    ...

