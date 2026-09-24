# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""LLM 后端接口。

privllm 不绑定任何模型服务——它只需要一个"发消息、收文本"的东西。
是 Ollama、vLLM、OpenAI 还是内部网关，由调用方注入。

这样切分的好处不只是解耦：privllm 里因此不出现任何 URL、API key、
模型名，也就不存在"密钥不小心跟着请求发出去"的路径。密钥和数据留在
这一侧，只有脱敏后的文本跨过后端边界。
"""

from typing import Protocol, Sequence, runtime_checkable

# {"role": "system" | "user" | "assistant", "content": str}
Message = dict[str, str]


class BackendError(RuntimeError):
    """后端调用失败。

    privllm 只认这一种异常来决定降级。后端实现请把所有底层错误
    （连接失败、超时、HTTP 5xx、鉴权失败）都转成它，不要把
    requests / httpx / SDK 的原生异常泄漏上来。
    """


@runtime_checkable
class ChatBackend(Protocol):
    """privllm 对后端的全部要求：送一轮对话，拿回助手文本。"""

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        timeout: int = 180,
    ) -> str:
        """执行一轮对话，返回助手回复的纯文本。

        实现要注意两点：

        1. 失败时抛 BackendError。privllm 靠它区分"模型说不了"和"程序有 bug"——
           前者可以降级，后者应该炸出来。

        2. 长回答建议用流式。非流式下读超时卡在整个生成上，本地 CPU 推理出长回答
           必然触发；流式下超时按分块算，只要还在吐字就不会断。

        temperature 为 None 时表示"由后端决定"，不要强行套一个默认值。
        """
        ...
