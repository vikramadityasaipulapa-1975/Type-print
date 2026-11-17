from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status


class InMemoryRateLimiter:
    """Per-process sliding window limiter for local deployments."""

    def __init__(self) -> None:
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = monotonic()
        with self._lock:
            requests = self._entries[key]
            while requests and requests[0] <= now - window_seconds:
                requests.popleft()
            if len(requests) >= limit:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests. Try again shortly.")
            requests.append(now)


limiter = InMemoryRateLimiter()


def rate_limit(request: Request, action: str, limit: int) -> None:
    ip = request.client.host if request.client else "unknown"
    limiter.check(f"{action}:{ip}", limit)
