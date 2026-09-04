"""OpenAI-compatible response filter. Public id is raiden-areishtern.

This is a sidecar, not a full vLLM replacement. Point vLLM at
--served-model-name raiden-areishtern and still run this filter so
exception traces and filesystem paths never reach the client.

Internal logs (stdout of vLLM / trainer) MAY contain zai-org/GLM-5.3-Flash.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raiden.identity import PUBLIC_MODEL_ID, PUBLIC_MODEL_ID_ALIASES
from raiden.serving_identity import client_error_payload, redact_mapping, rewrite_models_list


def filter_payload(obj):
    if isinstance(obj, dict):
        if obj.get("object") == "list" and "data" in obj:
            return rewrite_models_list([d.get("id", "") for d in obj.get("data", []) if isinstance(d, dict)])
        return redact_mapping(obj)
    return obj


class Handler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:8000"

    def log_message(self, fmt, *args):
        # Internal log: keep the raw line (may include real model ids).
        super().log_message(fmt, *args)

    def _proxy(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        url = self.upstream.rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
        req = Request(url, data=body if body else None, headers=headers, method=self.command)
        try:
            with urlopen(req, timeout=600) as resp:
                raw = resp.read()
                status = resp.status
                content_type = resp.headers.get("Content-Type", "application/json")
        except HTTPError as exc:
            raw = exc.read()
            status = exc.code
            content_type = "application/json"
            try:
                raw = json.dumps(client_error_payload(exc)).encode("utf-8")
            except Exception:
                raw = json.dumps(client_error_payload(exc)).encode("utf-8")
        except URLError as exc:
            raw = json.dumps(client_error_payload(exc)).encode("utf-8")
            status = 502
            content_type = "application/json"
        if "application/json" in content_type and raw:
            try:
                obj = json.loads(raw.decode("utf-8"))
                raw = json.dumps(filter_payload(obj), ensure_ascii=False).encode("utf-8")
            except Exception:
                raw = json.dumps({"model": PUBLIC_MODEL_ID, "error": {"message": "Request failed."}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--listen", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    args = p.parse_args(argv)
    Handler.upstream = args.upstream
    httpd = ThreadingHTTPServer((args.listen, args.port), Handler)
    print(f"RAIDEN public proxy on {args.listen}:{args.port} -> {args.upstream}")
    print(f"public model ids: {PUBLIC_MODEL_ID_ALIASES}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
