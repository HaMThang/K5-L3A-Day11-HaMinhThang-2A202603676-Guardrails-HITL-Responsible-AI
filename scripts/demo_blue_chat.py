"""Interactive terminal demo for the Blue Team banking assistant."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from agents.agent import create_blue_agent
from assignment.pipeline import build_production_plugins
from core.config import get_openrouter_api_key
from core.utils import chat_with_agent
from guardrails.input_guardrails import (
    detect_balance_action,
    detect_injection,
    detect_toxicity,
    topic_filter,
)


async def main() -> None:
    plugins = build_production_plugins()
    rate_limiter, input_guardrail, output_guardrail = plugins
    agent, runner = create_blue_agent(plugins)

    print("\n=== VinBank Blue Team demo ===")
    print("Hỏi về tài khoản, giao dịch, tiết kiệm, khoản vay hoặc thẻ.")
    print("Gõ /exit để kết thúc. Giới hạn mặc định: 10 câu/phút.\n")

    while True:
        try:
            prompt = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc demo.")
            return

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Đã kết thúc demo.")
            return

        rate_before = rate_limiter.blocked_count
        input_before = input_guardrail.blocked_count
        redacted_before = output_guardrail.redacted_count
        output_blocked_before = output_guardrail.blocked_count

        if (
            not get_openrouter_api_key()
            and detect_injection(prompt) == "ALLOW"
            and topic_filter(prompt) == "ALLOW"
            and detect_balance_action(prompt) == "ALLOW"
        ):
            print(
                "[CHƯA CÓ API KEY] Câu hỏi hợp lệ nhưng cần OPENROUTER_API_KEY "
                "trong .env để gọi Blue.\n"
            )
            continue

        try:
            response, _ = await chat_with_agent(agent, runner, prompt)
        except Exception as exc:
            print(
                f"[LỖI MODEL] Không gọi được Blue ({type(exc).__name__}). "
                "Kiểm tra OPENROUTER_API_KEY trong .env và kết nối mạng.\n"
            )
            continue

        if rate_limiter.blocked_count > rate_before:
            print("[BLUE ĐÃ CHẶN] Bạn đã vượt giới hạn yêu cầu. Hãy thử lại sau.\n")
        elif input_guardrail.blocked_count > input_before:
            if detect_injection(prompt) == "BLOCK":
                print("[BLUE ĐÃ CHẶN] Phát hiện dấu hiệu prompt injection.\n")
            elif detect_toxicity(prompt) == "BLOCK":
                print("[BLUE ĐÃ CHẶN] Vui lòng giữ nội dung hội thoại tôn trọng.\n")
            elif detect_balance_action(prompt) == "BLOCK":
                print(
                    "[BLUE ĐÃ CHẶN] Chatbot không được phép thao tác làm thay đổi "
                    "số dư tài khoản.\n"
                )
            elif topic_filter(prompt) == "BLOCK":
                print("[BLUE ĐÃ CHẶN] Câu hỏi ngoài phạm vi dịch vụ ngân hàng.\n")
            else:
                print("[BLUE ĐÃ CHẶN] Câu hỏi không vượt qua input guardrail.\n")
        elif output_guardrail.blocked_count > output_blocked_before:
            print("[BLUE ĐÃ CHẶN] Phản hồi có ngôn từ toxic hoặc xúc phạm.\n")
        elif output_guardrail.redacted_count > redacted_before:
            print("[BLUE ĐÃ BẢO VỆ] Phát hiện và đã che thông tin nhạy cảm trong câu trả lời.")
            print(f"Blue: {response}\n")
        else:
            print(f"Blue: {response or '(không có phản hồi)'}\n")


if __name__ == "__main__":
    asyncio.run(main())
