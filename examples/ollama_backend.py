"""Ollama 后端参考实现。

这是 privllm 之外的东西——privllm 只认 ChatBackend 协议，不认 Ollama。
把它当模板：换成 vLLM、OpenAI、或公司内部网关，只需要重写 chat() 一个方法。

接一个新后端时最容易踩的两个坑：

1. 异常类型。必须把底层异常统一转成 BackendError。privllm 靠它区分
   "模型说不了"（可降级）和"程序有 bug"（该炸）。
2. 流式。非流式下读超时卡在整个生成上，本地 CPU 推理出长回答必然触发；
   流式下超时按分块算，只要还在吐字就不会断。
"""

import json
from typing import Sequence

import requests

from privllm.backend import BackendError, Message


class OllamaBackend:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        think: bool | None = None,
    ):
        """think:
            None  = 不下发该参数（默认）。Ollama 只对支持的模型认它，
                    对不支持的模型传了会报错，所以默认不传最省事。
            False = 关掉思考。qwen3 这类思考模型建议显式关掉，省得输出  thinking。
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        timeout: int = 180,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
        }
        if self.think is not None:
            payload["think"] = self.think
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
                stream=True,
            ) as resp:
                if resp.status_code != 200:
                    raise BackendError(
                        f"Ollama 返回 HTTP {resp.status_code}：{resp.text[:200]}"
                    )
                parts = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parts.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        break
                return "".join(parts)
        except requests.RequestException as exc:
            raise BackendError(f"Ollama 请求失败：{exc}") from exc
