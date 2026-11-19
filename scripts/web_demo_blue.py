"""Run the local Blue Team chat demo web server."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB = ROOT / "web"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from agents.agent import create_blue_agent
from assignment.pipeline import build_observability, build_production_plugins
from core.config import get_openrouter_api_key
from core.utils import chat_with_agent
from guardrails.input_guardrails import (
    detect_balance_action,
    detect_hacking_intent,
    detect_injection,
    detect_toxicity,
    topic_filter,
)

PLUGINS = build_production_plugins(max_requests=10, window_seconds=60)
RATE_LIMITER, INPUT_GUARDRAIL, OUTPUT_GUARDRAIL = PLUGINS
AUDIT, MONITOR = build_observability()
BLUE_AGENT, BLUE_RUNNER = create_blue_agent(PLUGINS)
REQUEST_LOCK = threading.Lock()


def _reason_for_input(prompt: str) -> str:
    if detect_injection(prompt) == "BLOCK":
        return "Input guardrail phát hiện dấu hiệu prompt injection."
    if detect_toxicity(prompt) == "BLOCK":
        return "Input guardrail chặn lời lẽ xúc phạm hoặc toxic."
    if detect_balance_action(prompt) == "BLOCK":
        return "Chatbot không được phép thao tác làm thay đổi số dư tài khoản."
    if detect_hacking_intent(prompt) == "BLOCK":
        return "Yêu cầu hack hoặc xâm nhập bị chặn bởi input guardrail."
    if topic_filter(prompt) == "BLOCK":
        return "Câu hỏi nằm ngoài phạm vi dịch vụ ngân hàng."
    return "Yêu cầu bị chặn bởi input guardrail."


def _chat(prompt: str) -> dict:
    request_id = uuid4().hex
    user_id = "web-demo"
    AUDIT.record_input(user_id=user_id, text=prompt, request_id=request_id)

    rate_before = RATE_LIMITER.blocked_count
    input_before = INPUT_GUARDRAIL.blocked_count
    output_blocked_before = OUTPUT_GUARDRAIL.blocked_count
    output_redacted_before = OUTPUT_GUARDRAIL.redacted_count

    try:
        response, _ = asyncio.run(chat_with_agent(BLUE_AGENT, BLUE_RUNNER, prompt))
        response = response or ""
    except Exception as exc:
        if type(exc).__name__ == "RateLimitError":
            reason = (
                "Nhà cung cấp AI đang giới hạn tốc độ hoặc quota (HTTP 429). "
                "Đợi một lúc rồi thử lại; đây không phải thao tác tài khoản đã được thực hiện."
            )
        else:
            reason = (
                f"Không gọi được model ({type(exc).__name__}). "
                "Kiểm tra OPENROUTER_API_KEY trong .env và kết nối mạng."
            )
        AUDIT.record_output(
            user_id=user_id,
            text=reason,
            blocked=False,
            layer="model_error",
            request_id=request_id,
        )
        return {"status": "error", "answer": "", "reason": reason, "issues": []}

    if RATE_LIMITER.blocked_count > rate_before:
        status = "blocked"
        reason = "Rate limit: đã vượt quá 10 yêu cầu trong 60 giây."
        layer = "rate_limit"
    elif INPUT_GUARDRAIL.blocked_count > input_before:
        status = "blocked"
        reason = _reason_for_input(prompt)
        layer = "input_guardrail"
    elif OUTPUT_GUARDRAIL.blocked_count > output_blocked_before:
        status = "blocked"
        reason = "Output guardrail chặn phản hồi không phù hợp."
        layer = "output_guardrail"
    elif OUTPUT_GUARDRAIL.redacted_count > output_redacted_before:
        status = "blocked"
        issues = OUTPUT_GUARDRAIL.last_issues
        reason = "Phát hiện và mask thông tin nhạy cảm" + (
            f" ({', '.join(issues)})." if issues else "."
        )
        layer = "output_guardrail_redacted"
    else:
        status = "pass"
        reason = "Input và output đều vượt qua các lớp guardrail."
        layer = None

    AUDIT.record_output(
        user_id=user_id,
        text=response,
        blocked=status == "blocked",
        layer=layer,
        request_id=request_id,
    )
    MONITOR.total_requests += 1
    MONITOR.blocked_requests += int(status == "blocked")
    MONITOR.rate_limit_hits += int(layer == "rate_limit")
    return {
        "status": status,
        "answer": response,
        "reason": reason,
        "issues": list(OUTPUT_GUARDRAIL.last_issues),
    }


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        page = (WEB / "blue_chat.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 20000:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            prompt = body.get("message", "")
            if not isinstance(prompt, str) or not prompt.strip():
                self._send_json(400, {"error": "Nhập câu hỏi trước khi gửi."})
                return
            if len(prompt) > 4000:
                self._send_json(400, {"error": "Câu hỏi quá dài (tối đa 4000 ký tự)."})
                return
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Yêu cầu không hợp lệ."})
            return

        with REQUEST_LOCK:
            result = _chat(prompt.strip())
        code = 502 if result["status"] == "error" else 200
        self._send_json(code, result)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    if not get_openrouter_api_key():
        print("Cảnh báo: OPENROUTER_API_KEY chưa được cấu hình; chỉ ca bị chặn chạy được.")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), ChatHandler)
    print("Blue chat demo đang chạy: http://127.0.0.1:8000")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Blue chat demo...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
