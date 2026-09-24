# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""脱敏代理：客户端与 LLM 之间的可逆隐私层。

   明文 -> 实体识别 -> 替换 -> [ 发给 provider ] -> 还原 -> 明文

这层必须跑在客户端信任域内。密钥绝不到达 provider——这是整个方案的前提，
其它所有设计都建立在这条之上。
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import keys as keymod
from . import ner
from .ff1 import DIGITS, IDCARD_ALPHABET, FF1, minlen_for
from .pseudo import strip_prefix

if TYPE_CHECKING:
    from .backend import ChatBackend, Message

# 保格式加密的类别。其余类别走可读假名。
FPE_KINDS = ("PHONE", "ID_CARD", "BANK_CARD", "SSN", "EMAIL")


@dataclass(frozen=True)
class Entry:
    placeholder: str  # 裸值，不含界定标记。标记只在收发边界上施加。
    original: str  # 真实的
    type: str


@dataclass
class Redaction:
    original: str
    redacted: str
    entities: list[ner.Entity]
    entries: list[Entry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PrivacyProxy:
    """可逆脱敏代理。线程不安全——每个会话一个实例。"""

    def __init__(
        self,
        master_key: bytes,
        backend: "ChatBackend",
        structured_mode: str = "fpe",
        mark: str | None = None,
    ):
        """master_key:
            32 字节主密钥。所有子密钥由它派生，绝不要传给后端。
        backend:
            实现 ChatBackend 协议的对象。privllm 只通过它收发文本，
            不关心对方是 Ollama、vLLM 还是云端 API。
        structured_mode:
        - "fpe"   : 结构化字段用 FF1 保格式加密，密文看起来仍是手机号/身份证。
                    适合数据要过校验、要落库的场景。
        - "token" : 换成 [PHONE_1] 这类占位符，还原时无歧义。
                    适合纯对话，不care形态的场景。
        """
        if structured_mode not in ("fpe", "token"):
            raise ValueError("structured_mode 只能是 'fpe' 或 'token'")

        # 占位符外套一层界定标记（如 ⟦刘瑶⟧）。
        #
        # 好处：匹配变成精确的——⟦刘瑶⟧ 只可能是我们自己放进去的，模型自己写出
        # 的同名人名（真实的「刘瑶」）不会被误认成映射表里的实体。
        #
        # 为什么默认关闭：实测 gemma4:e2b 和 qwen3:8b **都会把这对括号当格式清理掉**
        # （⟦云启材料⟧ 写成"云启材料"）。两个模型都不保留标记，标记就只剩复杂度，
        # 还给出一种虚假的安全感——反正最后都要走降级路径。
        # 换成明确会保留结构化边界的模型（Claude/GPT 一级）时再打开才有意义。
        #
        # 打开时 restore 有降级路径：逐条检测标记是否还在，不在就退回裸串匹配并告警，
        # 绝不静默失败。
        self.mark = mark
        self._marker_re = (
            re.compile(re.escape(mark.split("{}")[0]) + r".+?" + re.escape(mark.split("{}")[1]))
            if mark
            else None
        )

        self.keys = keymod.derive_all(master_key)
        from .pseudo import Pseudonymizer

        self._pseudo = Pseudonymizer(self.keys["pseudo"])
        self.backend = backend
        self.structured_mode = structured_mode

        # 同一密钥 + 不同 tweak：把手机号、身份证、卡号隔到互不相通的定义域里。
        # 否则同一串数字在不同字段间会算出相同密文，泄露"这两个字段值相同"。
        self._ff1 = {
            "PHONE": FF1(self.keys["ff1"], radix=10, tweak=b"privllm/phone"),
            "BANK_CARD": FF1(self.keys["ff1"], radix=10, tweak=b"privllm/bankcard"),
            "SSN": FF1(self.keys["ff1"], radix=10, tweak=b"privllm/ssn"),
            "ID_CARD": FF1(
                self.keys["ff1"], radix=11, tweak=b"privllm/idcard", alphabet=IDCARD_ALPHABET
            ),
            # email 统一 lower 后只含小写字母数字，radix=36（DIGITS 字母表）。
            # 密文仍是合法的 email 形态，@ 和 . 位置不变。
            "EMAIL": FF1(
                self.keys["ff1"], radix=36, tweak=b"privllm/email", alphabet=DIGITS
            ),
        }
        self._counters: dict[str, int] = {}

    # ---------- 脱敏 ----------

    def _placeholder(self, ent: ner.Entity, canonical: str, cjk_ctx: bool = False) -> str:
        """canonical 是归并后的写法：别名共用同一个占位符，模型才能认出是同一实体。"""
        if ent.type in FPE_KINDS:
            if self.structured_mode == "fpe":
                return self._fpe(ent.type, canonical)
            self._counters[ent.type] = self._counters.get(ent.type, 0) + 1
            return f"{ent.type}_{self._counters[ent.type]}"
        return self._pseudo(ent.type, canonical, cjk_ctx)

    def _wrap(self, raw: str) -> str:
        return self.mark.format(raw) if self.mark else raw

    def _fpe(self, kind: str, value: str) -> str:
        cipher = self._ff1[kind]
        if kind == "ID_CARD":
            value = value.upper()
            if len(value) < 6:  # 达不到 FF1 的安全下限
                return f"[{kind}]"
            return cipher.encrypt(value)

        if kind == "EMAIL":
            # email 大小写不敏感，统一 lower 保证同一邮箱永远得到同一密文。
            # 只加密 ASCII 字母数字，@ 和 .（以及 - _ +）原样保留，密文仍是
            # email 形态。radix=36 的下限是 4 个可加密字符，太短就退回占位符。
            value = value.lower()
            alnum = "".join(c for c in value if c.isascii() and c.isalnum())
            if len(alnum) < minlen_for(36):
                return f"[{kind}]"
            enc = iter(cipher.encrypt(alnum))
            return "".join(
                next(enc) if (c.isascii() and c.isalnum()) else c for c in value
            )

        # 其余结构化类型只加密数字，非数字字符（+、-、空格、括号）原样保留。
        # 这样 +1-555-123-4567 加密后还是 +X-XXX-XXX-XXXX，123-45-6789 还是
        # XXX-XX-XXXX。中文连续号码没有分隔符，走这条路径与旧逻辑完全一致。
        digits = "".join(c for c in value if "0" <= c <= "9")
        if len(digits) < 6:
            return f"[{kind}]"
        enc = iter(cipher.encrypt(digits))
        return "".join(next(enc) if "0" <= c <= "9" else c for c in value)

    def redact(
        self,
        text: str,
        use_llm: bool = True,
        on_ner_failure: str = "raise",
        merge_aliases: bool = True,
    ) -> Redaction:
        if use_llm:
            entities, notes = ner.extract(
                text, self.backend, on_llm_failure=on_ner_failure
            )
        else:
            entities, notes = ner.extract_local_only(text), []

        canon = ner.canonical_map(entities) if merge_aliases else {e.text: e.text for e in entities}
        merged = {t: c for t, c in canon.items() if t != c}
        if merged:
            notes.append(
                "以下写法被判定为同一实体并归并（还原时会统一成最完整的写法）："
                + "、".join(f"{t}→{c}" for t, c in merged.items())
            )

        # 从后往前替换，避免前面的替换挪动了后面实体的偏移。
        # 这里显式排序，不依赖上游保证——偏移错位的结果是整段文本被搅碎，
        # 而且看起来仍然"像"脱敏成功，极难发现。
        out = text
        entries: list[Entry] = []
        seen: dict[tuple[str, str], str] = {}
        # 中性实体（项目代号/地址）的语言跟随整段文本，而非实体本身：「CRP2」
        # 出现在中文文本里该用「项目-」，不能因为它是拉丁就跳成英文「Project-」。
        text_cjk = any("一" <= c <= "鿿" for c in text)

        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            target = canon.get(ent.text, ent.text)
            ck = (ent.type, target)
            if ck not in seen:
                seen[ck] = self._placeholder(ent, target, text_cjk)
                entries.append(Entry(seen[ck], target, ent.type))
            out = out[: ent.start] + self._wrap(seen[ck]) + out[ent.end :]

        entries.reverse()
        return Redaction(
            original=text, redacted=out, entities=entities, entries=entries, notes=notes
        )

    # ---------- 还原 ----------

    def restore(self, text: str, redaction: Redaction) -> tuple[str, list[str]]:
        """把占位符换回真值。返回 (还原后文本, 告警列表)。

        模型可能编造出映射表里没有的号码。这类情况只告警，不静默丢弃——
        静默处理会让人误以为输出是干净的。
        """
        # 告警必须扫"还原之前"的模型输出——还原之后真值当然会出现，那样全是误报。
        warnings: list[str] = []
        if self.structured_mode == "fpe":
            known = {e.placeholder for e in redaction.entries}
            for ent in ner.extract_local_only(text):
                # 已知占位符本身可能被正则重新识别成子串（如 "+7-572-405-3824"
                # 里的 "572-405-3824"）。只要这个数是某个已知占位符的子串，就说明
                # 它不是模型编造的，跳过，避免误报。
                if ent.type in FPE_KINDS and ent.text not in known and not any(
                    ent.text in k for k in known
                ):
                    warnings.append(
                        f"输出中出现未经映射的 {ent.type}：{ent.text!r} —— 模型编造的数字，"
                        f"还原时无从对应"
                    )

        # 键用带标记的形态，匹配才是精确的——⟦刘瑶⟧ 只可能是我们自己放进去的。
        rmap = {self._wrap(e.placeholder): e.original for e in redaction.entries}
        if not rmap:
            return text, warnings

        # 必须单遍替换。用连续的 str.replace 会让前一次替换的产物被后一条规则再次命中：
        # 若映射里有 {李白->王强, 王强->刘瑶}，逐条替换会把「李白」变成「刘瑶」——
        # 把 B 的话安到 A 头上。re.sub 一次性扫描原文，替换结果不再重扫。
        # 长键优先，避免短键吃掉长键的前缀。
        pattern = re.compile("|".join(re.escape(k) for k in sorted(rmap, key=len, reverse=True)))
        out = pattern.sub(lambda m: rmap[m.group(0)], text)

        # 降级路径：模型若把界定标记当格式清理掉了（gemma4:e2b 就会），带标记的精确
        # 匹配会一个都命中不了，还原等于没做——而且是静默的。这里逐条检测标记是否
        # 还存在于模型输出中，缺失的退回裸串匹配，并把精度损失明确报出来。
        if self.mark:
            lost = [
                e for e in redaction.entries
                if self._wrap(e.placeholder) not in text and e.placeholder in out
            ]
            if lost:
                warnings.append(
                    f"模型未保留界定标记，{len(lost)} 条已降级为裸串匹配"
                    f"（自然碰撞风险回归）：{', '.join(e.placeholder for e in lost)}"
                )
                bare = {e.placeholder: e.original for e in lost}
                pat2 = re.compile(
                    "|".join(re.escape(k) for k in sorted(bare, key=len, reverse=True))
                )
                out = pat2.sub(lambda m: bare[m.group(0)], out)

        # 降级路径：模型可能把假名前缀剥掉——人名用的名字样标记会偶尔被剥（实测
        # gemma4:e2b：「翎羽·刘瑶」只写「刘瑶」），机构名的「假名·」同理。带前缀的
        # 占位符没出现在模型输出里、但剥掉前缀的裸名出现了，就退回裸名匹配并告警
        # ——这会重新引入自然碰撞风险，所以必须明确说出来，而不是静默降级。
        lost_prefixed: list[Entry] = []
        for e in redaction.entries:
            bare = strip_prefix(e.placeholder)
            if bare and e.placeholder not in text and bare in out:
                lost_prefixed.append(e)
        if lost_prefixed:
            warnings.append(
                f"模型剥掉了假名前缀，{len(lost_prefixed)} 条已降级为裸名匹配"
                f"（自然碰撞风险回归）：{', '.join(e.placeholder for e in lost_prefixed)}"
            )
            bare_map = {strip_prefix(e.placeholder): e.original for e in lost_prefixed}
            pat3 = re.compile(
                "|".join(re.escape(k) for k in sorted(bare_map, key=len, reverse=True))
            )
            out = pat3.sub(lambda m: bare_map[m.group(0)], out)

        # 还原后不应再有残留的占位符。有的话说明模型把标记写坏了或拼错，
        # 那条信息还原不回来，必须让人看见而不是悄悄放过。
        leftovers = set(self._marker_re.findall(out)) if self._marker_re else set()
        for leftover in leftovers:
            warnings.append(f"输出中残留未能还原的占位符：{leftover!r}")
        return out, warnings

    # ---------- 端到端 ----------

    def chat(
        self,
        user_text: str,
        system: str | None = None,
        use_llm_ner: bool = True,
        on_ner_failure: str = "raise",
        **backend_options,
    ) -> dict:
        """跑完整条链路，返回每一阶段的中间结果便于审计。

        关键不变量：跨过 backend 边界的只有 red.redacted。明文 user_text
        从头到尾没有离开过这个进程。
        """
        red = self.redact(user_text, use_llm=use_llm_ner, on_ner_failure=on_ner_failure)

        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": red.redacted})

        raw_reply = self.backend.chat(messages, **backend_options)

        restored, warnings = self.restore(raw_reply, red)
        return {
            "sent": red.redacted,
            "entities": red.entities,
            "entries": red.entries,
            "raw_reply": raw_reply,
            "restored": restored,
            "notes": red.notes,  # 过程性说明，无需处理
            "warnings": warnings,  # 需要人看一眼的异常
        }


# ---------- 映射表的加密落盘 ----------
#
# 假名派生是无状态的，理论上不需要存映射表。但"理论上"不等于"运维上"：
# 出问题时要能审计"当时到底把谁替换成了谁"。落盘必须加密，且用独立的子密钥。


def save_vault(path: str, redaction: Redaction, vault_key: bytes) -> None:
    """把映射表加密落盘。vault_key 传 proxy.keys["vault"]。

    这里刻意收子密钥而不是主密钥：落盘逻辑只需要 vault 这一支，
    给它主密钥等于让它有能力推出 ff1（反解所有脱敏字段）和 pseudo
    （字典攻击所有假名）。最小权限在这一层最容易被顺手破坏。
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    payload = json.dumps(
        [{"p": e.placeholder, "o": e.original, "t": e.type} for e in redaction.entries],
        ensure_ascii=False,
    ).encode("utf-8")

    nonce = os.urandom(12)
    blob = nonce + AESGCM(vault_key).encrypt(nonce, payload, None)
    with open(path, "wb") as fh:
        fh.write(blob)


def load_vault(path: str, vault_key: bytes) -> list[Entry]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    with open(path, "rb") as fh:
        blob = fh.read()
    plain = AESGCM(vault_key).decrypt(blob[:12], blob[12:], None)
    return [Entry(d["p"], d["o"], d["t"]) for d in json.loads(plain.decode("utf-8"))]
