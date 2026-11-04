"""Preview the self-contained PyRUA-REAL website with video Range support."""
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve()

    def serve(self, head=False):
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/__'):
            self.send_error(404)
            return
        root, relative = SITE, path.lstrip('/') or 'index.html'
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            self.send_error(404)
            return
        size = target.stat().st_size
        kind = {'.m4s':'video/iso.segment', '.mjs':'text/javascript', '.m3u8':'application/vnd.apple.mpegurl'}.get(target.suffix) or mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        source = target.open('rb')
        with source:
            start, end, status = 0, size - 1, 200
            value = self.headers.get('Range')
            if value:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                a, b = match.groups()
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if a and b else size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head:
                return
            source.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = source.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, *args):
        pass

if __name__ == '__main__':
    print('Local preview: http://127.0.0.1:8792/ (no password)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8792), Handler).serve_forever()
