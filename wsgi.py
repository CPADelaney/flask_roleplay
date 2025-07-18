# wsgi.py

import os
import logging
from dotenv import load_dotenv
import config_startup
# Removed unused Quart imports directly here as app comes from main
# from quart import Quart, render_template, session, request, jsonify, redirect
# import socketio
# from quart_cors import cors

import http.server
import threading
import socket
import time

from main import create_quart_app
from nyx.core.orchestrator import start_background

# ensure Hypercorn sees PORT
os.environ.setdefault("PORT", os.getenv("PORT", "8080"))

app = create_quart_app()
start_background()
