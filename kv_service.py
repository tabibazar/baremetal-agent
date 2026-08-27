#!/usr/bin/env python3
"""A minimal HTTP front for Redis, in the Upstash REST shape.

The BareMetal instances have no writable filesystem, so an agent running there
cannot keep notes locally. It does have TLS and HTTPS, though, so its memory can
live a network round-trip away. This is the other end of that.

Deliberately shaped like Upstash's REST API, so moving to hosted Upstash later
is a change of URL and token and nothing else:

    GET  /set/mykey/myvalue          -> {"result": "OK"}
    GET  /get/mykey                  -> {"result": "myvalue"}
    POST /  ["RPUSH", "k", "v"]      -> {"result": 3}

Auth is a bearer token. Commands are allowlisted: this endpoint is reachable
from the public internet, and a leaked token should not be able to reconfigure
or wipe the server, only touch keys.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import redis

PORT = int(os.environ.get("KV_PORT", "8080"))
TOKEN = os.environ.get("KV_TOKEN") or sys.exit("KV_TOKEN is required")
MAX_BODY = 64 * 1024
MAX_VALUE = 8 * 1024

# Everything the agent needs and nothing that reconfigures, enumerates or
# destroys the server. No FLUSHALL, CONFIG, KEYS, SHUTDOWN, SCRIPT.
ALLOWED = {
    "PING", "GET", "SET", "SETEX", "DEL", "EXISTS", "INCR", "EXPIRE", "TTL",
    "RPUSH", "LPUSH", "LRANGE", "LLEN", "LTRIM", "LPOP", "RPOP",
    "HSET", "HGET", "HGETALL", "HDEL",
    "SADD", "SMEMBERS", "SREM",
    # Sorted sets: reminders are kept scored by when they come due, so "what is
    # due now" is one range query rather than a scan of everything.
    "ZADD", "ZRANGEBYSCORE", "ZRANGE", "ZREM", "ZCARD", "ZSCORE",
}

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)


def decode(v):
    """Redis replies to something JSON can carry."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        return {decode(k): decode(x) for k, x in v.items()}
    return v


class Handler(BaseHTTPRequestHandler):
    server_version = "kv/1.0"

    def log_message(self, fmt, *args):        # one line per request, not three
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        got = self.headers.get("Authorization", "")
        if got.startswith("Bearer ") and got[7:].strip() == TOKEN:
            return True
        self._reply(401, {"error": "unauthorized"})
        return False

    def _run(self, parts):
        if not parts:
            return self._reply(400, {"error": "no command"})
        cmd = str(parts[0]).upper()
        if cmd not in ALLOWED:
            return self._reply(403, {"error": f"command not allowed: {cmd}"})
        args = [str(a) for a in parts[1:]]
        if any(len(a) > MAX_VALUE for a in args):
            return self._reply(413, {"error": "argument too large"})
        try:
            out = decode(r.execute_command(cmd, *args))
        except redis.RedisError as e:
            return self._reply(400, {"error": str(e)})

        # redis-py hands back Python bools where the Redis protocol — and
        # Upstash — return "OK" or 1/0. Normalise, so that swapping this service
        # for hosted Upstash later really is only a change of URL and token.
        if isinstance(out, bool):
            out = "OK" if cmd in {"SET", "SETEX", "LTRIM"} else int(out)
        return self._reply(200, {"result": out})

    def do_GET(self):
        if self.path == "/health":            # unauthenticated, for probes
            try:
                r.ping()
                return self._reply(200, {"result": "ok"})
            except redis.RedisError as e:
                return self._reply(503, {"error": str(e)})
        if not self._authorised():
            return
        parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/") if p]
        self._run(parts)

    def do_POST(self):
        if not self._authorised():
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._reply(400, {"error": "bad content-length"})
        if n > MAX_BODY:
            return self._reply(413, {"error": "body too large"})
        try:
            parts = json.loads(self.rfile.read(n) or b"[]")
        except json.JSONDecodeError:
            return self._reply(400, {"error": "body must be a JSON array"})
        if not isinstance(parts, list):
            return self._reply(400, {"error": "body must be a JSON array"})
        self._run(parts)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
