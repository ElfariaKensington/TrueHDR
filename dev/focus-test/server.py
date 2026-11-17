#!/usr/bin/env python3
"""Static server for the project root + a log sink for the focus test page.

Run from anywhere:  python3 dev/focus-test/server.py [port]
Page:  http://localhost:8765/dev/focus-test/index.html
POST /focus-log  appends JSON lines to dev/focus-test/logs/page.jsonl
GET  /focus-log/state  returns the most recent page state (geometry, focus, rates)
(Plain `python3 -m http.server` from the project root also works; the page then only
logs to its own panel and the console.)
"""
import http.server, json, os, sys, threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = os.path.join(LOGDIR, 'page.jsonl')
lock = threading.Lock()
last_state = {}


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_POST(self):
        if self.path != '/focus-log':
            self.send_error(404); return
        n = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(n)
        try:
            entries = json.loads(body)
        except Exception:
            self.send_error(400); return
        with lock:
            with open(LOGFILE, 'a') as f:
                for e in entries:
                    f.write(json.dumps(e) + '\n')
                    if e.get('kind') == 'state':
                        last_state.clear(); last_state.update(e)
        self.send_response(204); self.end_headers()

    def do_GET(self):
        if self.path == '/focus-log/state':
            with lock:
                data = json.dumps(last_state).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        super().do_GET()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    print(f'serving {ROOT} on http://localhost:{port}/dev/focus-test/index.html', flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
