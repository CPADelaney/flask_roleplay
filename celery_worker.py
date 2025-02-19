import eventlet
eventlet.monkey_patch()
import threading
import http.server
import socketserver

from celery_app import celery_app

def start_dummy_server(port=9000):
    # A simple HTTP server that does nothing, just listens on the port.
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    # Start the dummy HTTP server in a background thread
    dummy_thread = threading.Thread(target=start_dummy_server, kwargs={"port": 9000}, daemon=True)
    dummy_thread.start()
    
    # Now start the Celery worker directly
    celery_app.worker_main(["worker", "--loglevel=INFO", "-P", "eventlet"])
