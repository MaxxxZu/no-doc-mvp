"""Safe HTTP errors and bounded request body middleware."""
from starlette.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


def error_response(status, code, message, headers=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}},
                        headers=headers)


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            event = await receive()
            if event["type"] != "http.request":
                return
            chunk = event.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return await error_response(413, "body_too_large", "Request body is too large.")(
                    scope, receive, send)
            if chunk:
                chunks.append(chunk)
            if not event.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
