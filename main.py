import socketio

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

@sio.event
async def connect(sid, environ, auth):
    print('connect', sid)
    await sio.emit('greeting', {'message': 'hello'}, to=sid)

@sio.event
async def disconnect(sid):
    print('disconnect', sid)

if __name__ == "__main__":
    # run with uvicorn or hypercorn
    import hypercorn
    hypercorn.run(app, host="0.0.0.0", port=8080)
