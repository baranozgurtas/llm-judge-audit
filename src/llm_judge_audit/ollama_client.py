"""Minimal local Ollama client (stdlib only). One HTTP request per call; never retries."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from llm_judge_audit.parser import OUTPUT_SCHEMA


@dataclass
class ChatResult:
    """Outcome of one judge call. `status` is ok, runtime_error, or timeout."""

    status: str
    content: str | None
    latency_s: float
    done_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    connection_error: bool = False


class JudgeClient(Protocol):
    def chat(self, prompt: str) -> ChatResult: ...


def _base_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}"


def _get_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_version(endpoint: str) -> str:
    return str(_get_json(_base_url(endpoint) + "/api/version")["version"])


def model_digest(endpoint: str, tag: str) -> str | None:
    for m in _get_json(_base_url(endpoint) + "/api/tags")["models"]:
        if m.get("name") == tag or m.get("model") == tag:
            return str(m["digest"])
    return None


def model_details(endpoint: str, tag: str) -> dict[str, Any]:
    """Model metadata from /api/show (details, default parameters, template hash)."""
    from llm_judge_audit.io_utils import sha256_text

    info = _get_json(_base_url(endpoint) + "/api/show", {"model": tag})
    ctx = {k: v for k, v in info.get("model_info", {}).items() if k.endswith("context_length")}
    return {
        "details": info.get("details", {}),
        "default_parameters": info.get("parameters", ""),
        "template_sha256": sha256_text(info.get("template", "")),
        "capabilities": info.get("capabilities", []),
        "context_length": ctx,
    }


class OllamaJudge:
    def __init__(self, judge_cfg: dict[str, Any]) -> None:
        self.endpoint: str = judge_cfg["endpoint"]
        self.model: str = judge_cfg["model_tag"]
        self.options: dict[str, Any] = dict(judge_cfg["options"])
        self.keep_alive: str = judge_cfg["keep_alive"]
        self.timeout: float = float(judge_cfg["request_timeout_s"])

    def request_body(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }

    def chat(self, prompt: str) -> ChatResult:
        body = json.dumps(self.request_body(prompt)).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            return ChatResult("timeout", None, time.perf_counter() - start, error=repr(exc))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            return ChatResult(
                "runtime_error",
                None,
                time.perf_counter() - start,
                error=f"HTTP {exc.code}: {detail}",
            )
        except urllib.error.URLError as exc:
            latency = time.perf_counter() - start
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return ChatResult("timeout", None, latency, error=repr(exc.reason))
            return ChatResult(
                "runtime_error", None, latency, error=repr(exc.reason), connection_error=True
            )
        except (OSError, ValueError) as exc:
            return ChatResult("runtime_error", None, time.perf_counter() - start, error=repr(exc))
        latency = time.perf_counter() - start
        message = payload.get("message") or {}
        # Only the visible content is kept; any hidden "thinking" field is discarded.
        content = message.get("content")
        runtime = {
            k: payload.get(k)
            for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
        }
        runtime["model"] = payload.get("model")
        return ChatResult(
            status="ok" if isinstance(content, str) else "runtime_error",
            content=content if isinstance(content, str) else None,
            latency_s=latency,
            done_reason=payload.get("done_reason"),
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
            runtime=runtime,
            error=None if isinstance(content, str) else "response has no message content",
        )
