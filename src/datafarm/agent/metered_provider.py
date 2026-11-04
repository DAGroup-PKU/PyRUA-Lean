"""Trusted, journaled Responses gateway; run outside the author sandbox.

The metered model id and its price table are explicit deployment configuration with no
built-in default: load_pricing() reads them from a JSON object or file, and a Meter
admits only that model.
"""

import datetime, http.server, json, os, threading, time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import requests
from datafarm.common import sha256_bytes, sha256_text

MAX_INPUT = 1050000
MAX_OUTPUT = 4096
PRICE_FIELDS = ("input", "cached_input", "cache_write", "output")


@dataclass(frozen=True)
class Pricing:
    """USD per token for the one metered model, with an optional long-context tier."""

    model: str
    input: Decimal
    cached_input: Decimal
    cache_write: Decimal
    output: Decimal
    long_context_above: int | None = None  # input tokens; a larger request bills all tokens at the factors below
    long_context_input_factor: Decimal = Decimal(1)
    long_context_output_factor: Decimal = Decimal(1)
    source: str = ""


def load_pricing(value):
    """Pricing from a JSON object or file path; a missing or partial table is an error.

    {"model": "<provider model id>", "source": "<price page>",
     "usd_per_million_tokens": {"input": 10, "cached_input": 1, "cache_write": 12.5, "output": 50},
     "long_context": {"above_input_tokens": 272000, "input_factor": 2, "output_factor": 1.5}}  # optional
    """
    if isinstance(value, Pricing):
        return value
    if not value:
        raise ValueError(
            "metered pricing is not configured: give a JSON object or file with the metered model id "
            "and usd_per_million_tokens (input, cached_input, cache_write, output); there is no default"
        )
    if isinstance(value, (str, os.PathLike)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict) or not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError('metered pricing must name the metered model id ("model")')
    prices = value.get("usd_per_million_tokens") or {}
    missing = [k for k in PRICE_FIELDS if k not in prices]
    if missing:
        raise ValueError(f"metered pricing lacks usd_per_million_tokens {missing}")
    per_token = {k: Decimal(str(prices[k])) / Decimal(1000000) for k in PRICE_FIELDS}
    if min(per_token.values()) < 0:
        raise ValueError("metered prices must be non-negative")
    tier = value.get("long_context") or {}
    above = tier.get("above_input_tokens")
    factors = (Decimal(str(tier.get("input_factor", 1))), Decimal(str(tier.get("output_factor", 1))))
    # Factors >= 1 keep usd() monotone in total input, which bound() relies on.
    if (above is not None and (not isinstance(above, int) or above <= 0)) or min(factors) < 1:
        raise ValueError("long_context needs a positive integer above_input_tokens and factors >= 1")
    return Pricing(
        value["model"],
        **per_token,
        long_context_above=above,
        long_context_input_factor=factors[0],
        long_context_output_factor=factors[1],
        source=str(value.get("source", "")),
    )


def _check_usage(usage):
    n = int(usage["input_tokens"])
    out = int(usage["output_tokens"])
    d = usage.get("input_tokens_details") or {}
    cached = int(d.get("cached_tokens", 0))
    writes = int(d.get("cache_write_tokens", 0))
    if min(n, out, cached, writes) < 0 or cached + writes > n:
        raise ValueError("invalid usage")
    return n, out, cached, writes


def usd(usage, pricing):
    n, out, cached, writes = _check_usage(usage)
    p = pricing
    long_context = p.long_context_above is not None and n > p.long_context_above
    fi = p.long_context_input_factor if long_context else 1
    fo = p.long_context_output_factor if long_context else 1
    return (
        Decimal(n - cached - writes) * p.input * fi
        + Decimal(cached) * p.cached_input * fi
        + Decimal(writes) * p.cache_write * fi
        + Decimal(out) * p.output * fo
    )


def bound(payload, pricing):
    # Text tokens cannot exceed UTF-8 bytes. For vision, bound original
    # patches before any downscaling; the patch-based vision models reject
    # >30k patches and have a 1.2 multiplier. Add overhead and rounding margin, never alter images.
    # https://developers.openai.com/api/docs/guides/images-vision
    import base64, io, math

    image_tokens = 0
    unknown_media = False

    def clean(value):
        nonlocal image_tokens, unknown_media
        if isinstance(value, list):
            return [clean(x) for x in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") in ("input_audio", "input_file"):
            unknown_media = True
        if value.get("type") == "input_image":
            maximum = 36001
            url = value.get("image_url", "")
            if isinstance(url, str) and url.startswith("data:image/"):
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
                        maximum = min(
                            maximum, math.ceil(math.ceil(im.width / 32) * math.ceil(im.height / 32) * 1.2) + 1
                        )
                except Exception:
                    pass
            if value.get("detail") == "high":
                maximum = min(maximum, 3001)
            if value.get("detail") == "low":
                maximum = min(maximum, 309)
            image_tokens += maximum + 512
            return {"type": "input_image", "detail": value.get("detail", "auto")}
        return {k: clean(v) for k, v in value.items()}

    text = clean(payload)
    maximum = (
        MAX_INPUT
        if unknown_media
        else min(MAX_INPUT, len(json.dumps(text, ensure_ascii=False).encode()) + 16384 + image_tokens)
    )
    # Use the same context tier as settlement. All input charged as cache writes
    # is the worst allowed mixture; usd() is monotone in total input, including
    # its long-context threshold. Keep the byte/media upper bound unchanged.
    return usd(
        {"input_tokens": maximum, "output_tokens": MAX_OUTPUT, "input_tokens_details": {"cache_write_tokens": maximum}},
        pricing,
    )


def completed_response_events(response):
    """Frame a fully received upstream response without changing its items or usage."""
    events = [{"type": "response.created", "response": dict(response, output=[], status="in_progress")}]
    for index, item in enumerate(response.get("output", [])):
        initial = dict(item, status="in_progress")
        for field in ("arguments", "input"):
            if field in initial:
                initial[field] = ""
        if "content" in initial:
            initial["content"] = []
        events.append({"type": "response.output_item.added", "output_index": index, "item": initial})
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    status = response.get("status")
    if status not in ("completed", "incomplete", "failed"):
        raise ValueError("buffered upstream response is not terminal")
    events.append({"type": "response." + status, "response": response})
    return b"".join(
        ("event: " + event["type"] + "\ndata: " + json.dumps(dict(event, sequence_number=i)) + "\n\n").encode()
        for i, event in enumerate(events)
    )


def openai_compatible_receipt(endpoint, key, request_id, model):
    """Retry only read-only receipt transport failures, never generation calls."""
    for attempt in range(3):
        try:
            return _openai_compatible_receipt_once(endpoint, key, request_id, model)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 2:
                raise


def _openai_compatible_receipt_once(endpoint, key, request_id, model):
    """Read one exact server-request receipt; never infer identity from timing.

    The OpenAI-compatible (new-api style) token log is read from the generation endpoint's own
    origin, so the key never reaches another host. Opt-in: provider_receipt_lookup.
    """
    from urllib.parse import urlsplit

    origin = urlsplit(endpoint)
    if origin.scheme != "https" or not origin.hostname:
        return None
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            f"https://{origin.netloc}/api/log/token", headers={"Authorization": "Bearer " + key}, timeout=(10, 15)
        )
        response.raise_for_status()
        body = response.json()
        if body.get("success") is not True or not isinstance(body.get("data"), list):
            return None
        rows = [row for row in body["data"] if row.get("request_id") == request_id]
        if len(rows) != 1:
            return None
        row = rows[0]
        if row.get("model_name") != model or row.get("type") != 2:
            return None
        extra = json.loads(row.get("other") or "{}")
        if extra.get("matched_tier") not in ("standard", "long_context"):
            return None
        # This integration only recognizes the observed cache-read receipt schema.
        if any(v for k, v in extra.items() if "cache" in k and k not in ("cache_tokens", "cache_ratio")):
            return None
        usage = {
            "input_tokens": int(row["prompt_tokens"]),
            "output_tokens": int(row["completion_tokens"]),
            "input_tokens_details": {"cached_tokens": int(extra.get("cache_tokens", 0))},
        }
        _check_usage(usage)  # validate usage before accepting a receipt
        return {
            "usage": usage,
            "request_id": request_id,
            "provider_row_sha256": sha256_text(json.dumps(row, sort_keys=True)),
        }
    finally:
        session.close()


def tracking_adapter(progress):
    """Track entry to HTTP writes, after TLS, without recording headers or body.

    A failure before send() starts is provably pre-request. Once a write starts,
    even a socket exception remains an unknown liability. No transport retries.
    """
    from urllib3.connection import HTTPSConnection
    from urllib3.connectionpool import HTTPSConnectionPool

    class Connection(HTTPSConnection):
        def connect(self):
            progress["connect_started"] = True
            super().connect()
            progress["connect_complete"] = True

        def send(self, data):
            if self.sock is None:
                self.connect()
            progress["http_write_started"] = True
            return super().send(data)

    class Pool(HTTPSConnectionPool):
        ConnectionCls = Connection

    class Adapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme, https=Pool)

    return Adapter(max_retries=0)


class Meter:
    def __init__(
        self, path, limit, *, pricing=None, buffered=False, receipt_lookup=False, read_timeout_s=900, stream_max_s=None
    ):
        self.pricing = load_pricing(pricing)  # model id and prices: explicit configuration, no default
        self.path = Path(path)
        self.limit = Decimal(str(limit))
        self.spent = Decimal(0)
        self.unknown = False
        self.lock = threading.Lock()
        self.calls = 0
        if self.path.exists():
            raise ValueError("never reset an existing driver journal")
        self.log({"event": "meter-start", "at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        self.denied = False
        self.buffered = buffered
        self.receipt_lookup = receipt_lookup
        self.read_timeout_s = float(read_timeout_s)
        if not 30 <= self.read_timeout_s <= 900:
            raise ValueError("read timeout must be30..900seconds")
        self.stream_max_s = None if stream_max_s is None else float(stream_max_s)
        if self.stream_max_s is not None and not 30 <= self.stream_max_s <= 900:
            raise ValueError("stream duration must be30..900seconds")
        self._session = None
        self._transport_progress = {}
        self.not_sent_calls = 0

    def log(self, item):
        with self.path.open("a") as f:
            f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def request(self, payload, endpoint, key, emit=None):
        with self.lock:
            if payload.get("model") != self.pricing.model:
                return 400, {"error": {"message": "Only the configured metered model is allowed"}}
            if payload.get("previous_response_id"):
                return 400, {"error": {"message": "Server-side context is not metered by this driver"}}
            if any(t.get("type") not in ("function", "custom") for t in payload.get("tools", [])):
                return 400, {"error": {"message": "Unpriced hosted tools are disabled"}}
            payload["max_output_tokens"] = MAX_OUTPUT
            payload["service_tier"] = "default"
            payload["store"] = False
            client_stream = bool(payload.get("stream"))
            if self.buffered:
                payload["stream"] = False
            reserve = bound(payload, self.pricing)
            if self.unknown or self.spent + reserve > self.limit:
                self.denied = True
                self.log(
                    {
                        "event": "denied-local",
                        "reason": "unknown-liability" if self.unknown else "request-bound",
                        "required_usd": str(reserve),
                        "remaining_usd": str(self.limit - self.spent),
                        "provider_calls": self.calls,
                        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                )
                return 402, {
                    "error": {"message": "Task cost reservation unavailable; stop model work and preserve results"}
                }
            # Establish TLS with a read-only request before sending a billable body.
            # Keep the connection pool across calls rather than reconnecting each time.
            if self._session is None:
                candidate = requests.Session()
                candidate.trust_env = False
                if hasattr(candidate, "mount") and endpoint.startswith("https://"):
                    candidate.mount("https://", tracking_adapter(self._transport_progress))
                    self._transport_progress["installed"] = True
                try:
                    probe = candidate.get(
                        endpoint.rsplit("/", 1)[0] + "/models",
                        headers={"Authorization": "Bearer " + key},
                        timeout=(30, 30),
                    )
                    if probe.status_code != 200:
                        raise RuntimeError("Read-only preflight HTTP " + str(probe.status_code))
                    probe.close()
                except Exception as e:
                    close = getattr(candidate, "close", None)
                    if close:
                        close()
                    self.log(
                        {
                            "event": "preflight-failed",
                            "error_type": type(e).__name__,
                            "error_detail": str(e).replace(key, "[redacted]")[:1000],
                            "provider_calls": self.calls,
                            "billable_request_sent": False,
                        }
                    )
                    return 503, {
                        "error": {"message": "Read-only connection preflight failed; no generation request sent"}
                    }
                self._session = candidate
                self.log({"event": "preflight-passed", "billable_request_sent": False})
            self.calls += 1
            number = self.calls
            self.log(
                {
                    "event": "reserve",
                    "call": number,
                    "usd": str(reserve),
                    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "request_sha256": sha256_text(json.dumps(payload, sort_keys=True)),
                    "reasoning_effort": (payload.get("reasoning") or {}).get("effort"),
                    "stream": bool(payload.get("stream")),
                    "max_output_tokens": payload["max_output_tokens"],
                }
            )
            self.unknown = True
            session = self._session
            tracked = self._transport_progress.get("installed", False)
            self._transport_progress.clear()
            self._transport_progress.update(installed=tracked, http_write_started=False)
            started = time.monotonic()
            stage = "awaiting_headers"
            upstream_ids = {}
            response = None
            try:
                response = session.post(
                    endpoint,
                    headers={"Authorization": "Bearer " + key},
                    json=payload,
                    timeout=(30, self.read_timeout_s),
                    allow_redirects=False,
                    stream=True,
                )
                stage = "reading_body"
                usage = None
                rid = None
                chunks = []
                upstream_ids = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    in (
                        "x-request-id",
                        "x-one-api-request-id",
                        "x-oneapi-request-id",
                        "x-new-api-request-id",
                        "x-newapi-request-id",
                    )
                }
                self.log(
                    {
                        "event": "upstream-headers",
                        "call": number,
                        "http_status": response.status_code,
                        "request_ids": upstream_ids,
                    }
                )
                if response.status_code == 200 and payload.get("stream"):
                    if emit:
                        emit(None)
                    for line in response.iter_lines(chunk_size=1):
                        if self.stream_max_s is not None and time.monotonic() - started > self.stream_max_s:
                            stage = "stream_duration_exceeded"
                            raise requests.exceptions.ReadTimeout("Stream duration exceeded despite incoming data")
                        packet = line + b"\n"
                        chunks.append(packet)
                        if emit:
                            emit(packet)
                        if not line.startswith(b"data:"):
                            continue
                        try:
                            event = json.loads(line[5:].lstrip())
                        except ValueError:
                            continue
                        obj = event.get("response", {})
                        if obj.get("id"):
                            rid = obj["id"]
                        if obj.get("usage"):
                            usage = obj["usage"]
                        if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                            if emit:
                                emit(b"\n")
                            chunks.append(b"\n")
                            break
                    content = b"".join(chunks)
                else:
                    content = response.content
                    if response.status_code == 200:
                        obj = response.json()
                        usage = obj.get("usage")
                        rid = obj.get("id")
                        if self.buffered and client_stream:
                            content = completed_response_events(obj)
                            if emit:
                                emit(None)
                                emit(content)
                response.close()
                self.log(
                    {
                        "event": "upstream-response",
                        "call": number,
                        "http_status": response.status_code,
                        "request_ids": upstream_ids,
                        "response_id": rid,
                        "response_sha256": sha256_bytes(content),
                    }
                )
                if usage is not None:
                    actual = usd(usage, self.pricing)
                    self.spent += actual
                    self.unknown = actual > reserve
                    self.log(
                        {
                            "event": "settled",
                            "call": number,
                            "response_id": rid,
                            "usage": usage,
                            "usd": str(actual),
                            "total_usd": str(self.spent),
                            "over_reservation": actual > reserve,
                        }
                    )
                else:
                    self.log(
                        {
                            "event": "unknown",
                            "call": number,
                            "http_status": response.status_code,
                            "reserved_usd": str(reserve),
                            "response_id": rid,
                            "request_ids": upstream_ids,
                        }
                    )
                return response.status_code, content
            except Exception as e:
                if response is not None:
                    response.close()
                progress = dict(self._transport_progress)
                if progress.get("installed") and not progress.get("http_write_started"):
                    self.unknown = False
                    self.not_sent_calls += 1
                    self.log(
                        {
                            "event": "not-sent",
                            "call": number,
                            "cost_usd": "0",
                            "error_type": type(e).__name__,
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "transport": progress,
                            "released_reservation_usd": str(reserve),
                        }
                    )
                    return 503, {
                        "error": {"message": "Connection failed before HTTP write; no generation request sent"}
                    }
                if self.receipt_lookup:
                    rid_for_lookup = next(
                        (v for k, v in upstream_ids.items() if k.lower() == "x-oneapi-request-id"), None
                    )
                    if rid_for_lookup:
                        try:
                            receipt = openai_compatible_receipt(endpoint, key, rid_for_lookup, payload["model"])
                        except Exception as lookup_error:
                            receipt = None
                            self.log(
                                {
                                    "event": "receipt-lookup-failed",
                                    "call": number,
                                    "error_type": type(lookup_error).__name__,
                                }
                            )
                        if receipt:
                            actual = usd(receipt["usage"], self.pricing)
                            self.spent += actual
                            self.unknown = actual > reserve
                            self.log(
                                {
                                    "event": "settled",
                                    "call": number,
                                    "response_id": None,
                                    **receipt,
                                    "usd": str(actual),
                                    "total_usd": str(self.spent),
                                    "over_reservation": self.unknown,
                                    "cost_source": "exact-provider-request-receipt",
                                    "output_received": False,
                                    "transport_error_type": type(e).__name__,
                                }
                            )
                            return 503, {
                                "error": {
                                    "message": "503 Service Unavailable: output interrupted; exact provider usage reconciled"
                                }
                            }
                self.log(
                    {
                        "event": "unknown",
                        "call": number,
                        "error_type": type(e).__name__,
                        "reserved_usd": str(reserve),
                        "stage": stage,
                        "transport": progress,
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "error_detail": str(e).replace(key, "[redacted]")[:1000],
                    }
                )
                return 502, {"error": {"message": "Upstream interrupted; liability unknown, further calls blocked"}}


class MeteredGateway:
    def __init__(self, meter, endpoint, key):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = b'{"models":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path not in ("/v1/responses", "/responses"):
                    self.send_error(404)
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                except Exception:
                    self.send_error(400)
                    return
                started = False
                done = threading.Event()
                write_lock = threading.Lock()
                heartbeat = None

                def emit(packet):
                    nonlocal started, heartbeat
                    if packet is None:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True

                        def keepalive():
                            while not done.wait(10):
                                emit(b": keepalive\n\n")

                        heartbeat = threading.Thread(target=keepalive, daemon=True)
                        heartbeat.start()
                        return
                    with write_lock:
                        try:
                            self.wfile.write(packet)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            pass

                try:
                    status, body = meter.request(payload, endpoint, key, emit=emit)
                    body = body if isinstance(body, bytes) else json.dumps(body).encode()
                    if started:
                        if status != 200:
                            emit(b"event: error\ndata: " + body + b"\n\n")
                        self.close_connection = True
                    else:
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        try:
                            self.wfile.write(body)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                finally:
                    done.set()
                    if heartbeat:
                        heartbeat.join(timeout=1)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = False  # Drain every metered request before writing the final receipt.
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.server = server
        self.thread = thread
        self.meter = meter
        self.url = f"http://127.0.0.1:{server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        close = getattr(self.meter._session, "close", None)
        if close:
            close()
