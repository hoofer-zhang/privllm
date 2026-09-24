# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""FF1 格式保持加密 (NIST SP 800-38G)。

只实现 FF1，刻意不实现 FF3：FF3 已被 Durak & Vaudenay (2017) 攻破并由 NIST
撤销，替代品 FF3-1 把 tweak 限制到 64 位。FF1 目前无已知实际攻击。

FF1 的性质（正是脱敏场景需要的）：
  - 保持格式：radix=10 的 11 位手机号加密后仍是 11 位数字
  - 确定性：同一 (key, tweak, 明文) 永远得到同一密文 → 模型能做指代消解
  - 可逆且无状态：不需要存映射表，重启不失效
  - 安全域下限：radix^minlen >= 1,000,000，否则算法不保证安全
"""

import math

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# NIST 向量使用 0-9a-z 的字母表，大小写敏感
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"

# 中国身份证：末位可能是 X，用自定义字母表覆盖
IDCARD_ALPHABET = "0123456789X"

_MIN_DOMAIN = 1_000_000  # SP 800-38G 规定的最小定义域


def _num(alphabet: str, s: str) -> int:
    """NUM_radix：数字串 -> 整数。"""
    radix = len(alphabet)
    n = 0
    for ch in s:
        i = alphabet.find(ch)
        if i < 0:
            raise ValueError(f"字符 {ch!r} 不在字母表 {alphabet!r} 内")
        n = n * radix + i
    return n


def _str(alphabet: str, m: int, n: int) -> str:
    """STR_radix^m：整数 -> 定长 m 的数字串。"""
    radix = len(alphabet)
    out = []
    for _ in range(m):
        out.append(alphabet[n % radix])
        n //= radix
    return "".join(reversed(out))


def minlen_for(radix: int) -> int:
    """满足 radix^m >= 10^6 的最小 m。低于此长度的输入不可用 FF1。"""
    m = 1
    while radix**m < _MIN_DOMAIN:
        m += 1
    return m


class FF1:
    """FF1 密码对象。key 为 16/24/32 字节 AES 密钥，tweak 为任意字节串。"""

    def __init__(
        self,
        key: bytes,
        radix: int = 10,
        tweak: bytes = b"",
        alphabet: str | None = None,
    ):
        if alphabet is not None:
            if len(alphabet) != radix:
                raise ValueError(f"字母表长度 {len(alphabet)} 与 radix={radix} 不符")
            if len(set(alphabet)) != len(alphabet):
                raise ValueError("字母表含重复字符")
            self.alphabet = alphabet
        else:
            if radix < 2 or radix > 36:
                raise ValueError("未指定字母表时 radix 必须在 [2, 36] 内")
            self.alphabet = DIGITS[:radix]

        if len(key) not in (16, 24, 32):
            raise ValueError("AES 密钥必须为 16/24/32 字节")
        self.key = key
        self.radix = radix
        self.tweak = tweak
        self._enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()

    def _ciph(self, block: bytes) -> bytes:
        """单分组 AES 加密（CIPH）。ECB 无状态，复用 encryptor 是安全的。"""
        return self._enc.update(block)

    def _prf(self, data: bytes) -> bytes:
        """PRF = AES-CBC-MAC，IV 固定为全零，取最后一个分组。"""
        enc = Cipher(algorithms.AES(self.key), modes.CBC(b"\x00" * 16)).encryptor()
        return (enc.update(data) + enc.finalize())[-16:]

    def _check(self, x: str) -> None:
        if len(x) < minlen_for(self.radix):
            raise ValueError(
                f"输入长度 {len(x)} 低于 radix={self.radix} 的安全下限 "
                f"{minlen_for(self.radix)}（radix^len 必须 >= 10^6）"
            )

    def _setup(self, x: str):
        """两个方向共用的输入展开：返回 (A, B, P, b, d, u, v)。"""
        radix = self.radix
        n = len(x)
        self._check(x)

        u = n // 2
        v = n - u
        A, B = x[:u], x[u:]

        b = math.ceil(math.ceil(v * math.log2(radix)) / 8)
        d = 4 * math.ceil(b / 4) + 4

        # P 固定 16 字节：算法标识 + radix + 轮数 + u + n + tweak 长度
        P = (
            bytes([1, 2, 1])
            + radix.to_bytes(3, "big")
            + bytes([10, u % 256])
            + n.to_bytes(4, "big")
            + len(self.tweak).to_bytes(4, "big")
        )
        return A, B, P, b, d, u, v

    def _round(self, i: int, X: str, P: bytes, b: int, d: int) -> int:
        """第 i 轮的 S 值。X 是参与 PRF 的那一半：加密用 B，解密用 A。"""
        pad = (-len(self.tweak) - b - 1) % 16
        Q = self.tweak + b"\x00" * pad + bytes([i]) + _num(self.alphabet, X).to_bytes(b, "big")

        R = self._prf(P + Q)

        # S = R || CIPH(R xor [1]^16) || CIPH(R xor [2]^16) || ... ，截断到 d 字节。
        # 注意 [j]^16 是「整数 j 的大端 16 字节编码」（0x00..01），不是 0x01 重复 16 次。
        S = bytearray(R)
        j = 1
        while len(S) < d:
            ctr = j.to_bytes(16, "big")
            S += self._ciph(bytes(a ^ c for a, c in zip(R, ctr)))
            j += 1

        # S 是字节串，NUM 按 256 进制解释
        return int.from_bytes(bytes(S[:d]), "big")

    def encrypt(self, x: str) -> str:
        radix = self.radix
        alpha = self.alphabet
        A, B, P, b, d, u, v = self._setup(x)

        for i in range(10):
            y = self._round(i, B, P, b, d)
            m = u if i % 2 == 0 else v
            c = (_num(alpha, A) + y) % radix**m
            A, B = B, _str(alpha, m, c)

        return A + B

    def decrypt(self, x: str) -> str:
        radix = self.radix
        alpha = self.alphabet
        A, B, P, b, d, u, v = self._setup(x)

        # 解密是加密的严格逆序：轮次倒着走、Q 取 A、从 B 里减、交换方向相反
        for i in reversed(range(10)):
            y = self._round(i, A, P, b, d)
            m = u if i % 2 == 0 else v
            c = (_num(alpha, B) - y) % radix**m
            A, B = _str(alpha, m, c), A

        return A + B
