"""Shared HTTP client for EDGAR network calls.

Provides:
- Session reuse (connection pooling)
- Conservative retry/backoff defaults
- Bounded connect/read timeouts
"""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    HTTP_ADAPTIVE_RATE_LIMIT,
    HTTP_BACKOFF_FACTOR,
    HTTP_CONNECT_TIMEOUT,
    HTTP_MAX_RETRIES,
    HTTP_MIN_INTERVAL_DATA_SEC,
    HTTP_MIN_INTERVAL_WWW_SEC,
    HTTP_READ_TIMEOUT,
)

_THREAD_LOCAL = threading.local()
_HOST_LIMITER_LOCK = threading.Lock()
_HOST_LIMITERS: dict[str, "_HostLimiter"] = {}


class _HostLimiter:
    def __init__(self, min_interval_seconds: float):
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval_seconds <= 0:
            return

        sleep_seconds = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_seconds = self._next_allowed - now
            schedule_from = max(now, self._next_allowed)
            self._next_allowed = schedule_from + self._min_interval_seconds

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _host_interval_seconds(host: str) -> float:
    host_lower = host.lower()
    if host_lower == "data.sec.gov":
        return HTTP_MIN_INTERVAL_DATA_SEC
    if host_lower == "www.sec.gov":
        return HTTP_MIN_INTERVAL_WWW_SEC
    # Keep conservative default for other hosts using SEC fetches.
    return max(HTTP_MIN_INTERVAL_DATA_SEC, HTTP_MIN_INTERVAL_WWW_SEC)


def _wait_for_host_slot(url: str) -> None:
    if not HTTP_ADAPTIVE_RATE_LIMIT:
        return

    host = urlparse(url).netloc
    if not host:
        return

    with _HOST_LIMITER_LOCK:
        limiter = _HOST_LIMITERS.get(host)
        if limiter is None:
            limiter = _HostLimiter(_host_interval_seconds(host))
            _HOST_LIMITERS[host] = limiter

    limiter.wait()


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        connect=HTTP_MAX_RETRIES,
        read=HTTP_MAX_RETRIES,
        status=HTTP_MAX_RETRIES,
        allowed_methods=frozenset(["GET", "HEAD"]),
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=HTTP_BACKOFF_FACTOR,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = _build_session()
        _THREAD_LOCAL.session = session
    return session


def get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: tuple[float, float] | None = None,
    rate_limited: bool = True,
    **kwargs,
):
    """Issue a GET request using shared settings.

    Args:
        url: Request URL.
        headers: Optional request headers.
        timeout: Optional (connect, read) timeout tuple.
        **kwargs: Additional requests keyword args.
    """
    if rate_limited:
        _wait_for_host_slot(url)
    effective_timeout = timeout if timeout is not None else (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
    return _get_session().get(url, headers=headers, timeout=effective_timeout, **kwargs)
