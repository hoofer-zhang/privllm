# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""privllm —— 给任意 LLM 连接套一层可逆脱敏。

   明文 -> 实体识别 -> 替换 -> [ 交给后端 ] -> 还原 -> 明文

密钥和数据始终留在这一侧，只有脱敏后的文本跨过 ChatBackend 边界。

最小用法：

    from privllm import PrivacyProxy, load_master
    from examples.ollama_backend import OllamaBackend

    proxy = PrivacyProxy(load_master(), OllamaBackend(model="gemma4:e2b"))
    result = proxy.chat("联系人张伟，手机 13812345678")
    print(result["sent"])      # 发给模型的内容（已脱敏）
    print(result["restored"])  # 还原后的回答

注意：这只挡得住结构化 PII 和命名实体。自由文本的"内容本身即机密"
挡不住，详见 README 的「已知限制」。
"""

from . import ner
from .backend import BackendError, ChatBackend, Message
from .ff1 import FF1, IDCARD_ALPHABET, minlen_for
from .keys import derive, derive_all, load_master
from .pseudo import Pseudonymizer
from .proxy import Entry, PrivacyProxy, Redaction, load_vault, save_vault

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "ChatBackend",
    "Entry",
    "FF1",
    "IDCARD_ALPHABET",
    "Message",
    "PrivacyProxy",
    "Pseudonymizer",
    "Redaction",
    "derive",
    "derive_all",
    "load_master",
    "load_vault",
    "minlen_for",
    "ner",
    "save_vault",
    "__version__",
]
