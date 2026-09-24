# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""密钥管理：一个主密钥，用 HKDF 派生出各用途的子密钥。

绝不要用一个密钥干所有事。FF1 的 key、假名派生的 key、映射表加密的 key
必须相互独立——否则某个用途上的泄露会连累其它用途。
"""

import os
import secrets
import stat
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MASTER_KEY_BYTES = 32
DEFAULT_KEY_PATH = Path.home() / ".privllm" / "master.key"

# 子密钥用途。新增用途时加在这里，不要复用已有名字。
PURPOSES = ("ff1", "pseudo", "vault")


def derive(master: bytes, purpose: str, length: int = 32) -> bytes:
    """从主密钥派生指定用途的子密钥。"""
    if len(master) != MASTER_KEY_BYTES:
        raise ValueError(f"主密钥必须为 {MASTER_KEY_BYTES} 字节")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=b"privllm/v1/" + purpose.encode(),
    ).derive(master)


def derive_all(master: bytes) -> dict[str, bytes]:
    return {p: derive(master, p) for p in PURPOSES}


def load_master(path: Path | None = None, create: bool = True) -> bytes:
    """按 环境变量 > 密钥文件 的顺序取主密钥，都没有则生成一个。

    环境变量 PRIVLLM_MASTER_KEY 存 64 位十六进制。
    """
    env = os.environ.get("PRIVLLM_MASTER_KEY")
    if env:
        raw = bytes.fromhex(env.strip())
        if len(raw) != MASTER_KEY_BYTES:
            raise ValueError(f"PRIVLLM_MASTER_KEY 必须是 {MASTER_KEY_BYTES * 2} 位十六进制")
        return raw

    path = path or DEFAULT_KEY_PATH
    if path.exists():
        raw = bytes.fromhex(path.read_text().strip())
        if len(raw) != MASTER_KEY_BYTES:
            raise ValueError(f"{path} 内容不是 {MASTER_KEY_BYTES} 字节密钥")
        return raw

    if not create:
        raise FileNotFoundError(f"主密钥不存在：{path}")

    raw = secrets.token_bytes(MASTER_KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw.hex())
    # 仅属主可读。Windows 上 chmod 基本无效，靠目录 ACL。
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return raw
