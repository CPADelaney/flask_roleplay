import os
import logging
from quart import Quart
import socketio

def create_quart_app():
    app = Quart(__name__)
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        logger=True,
        engineio_logger=True
    )
    app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)
    app.socketio = sio

    @sio.event
    async def connect(sid, environ, auth):
        print(f"Connected: {sid}")
        await sio.emit("response", {"data": "Connected!"}, to=sid)

    @sio.event
    async def disconnect(sid):
        print(f"Disconnected: {sid}")

    @app.route("/")
    async def index():
        return "Memento Mori"

    return app
