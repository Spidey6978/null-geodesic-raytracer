"""
Module: api.rate_limiter
Sliding-window IP rate-limiting middleware for public API protection.
Prevents request spam and denial-of-service on local GPU/CPU rendering resources.
"""

import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 20, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.ip_history = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only rate-limit POST render job submissions
        if request.method == "POST" and "/renders" in request.url.path:
            client_ip = request.client.host if request.client else "127.0.0.1"
            now = time.time()

            # Clean history outside window
            history = [t for t in self.ip_history[client_ip] if now - t < self.window_seconds]
            self.ip_history[client_ip] = history

            if len(history) >= self.max_requests:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Maximum {self.max_requests} render requests per {self.window_seconds}s allowed per IP.",
                        "error_code": "RATE_LIMIT_EXCEEDED"
                    }
                )

            self.ip_history[client_ip].append(now)

        return await call_next(request)
