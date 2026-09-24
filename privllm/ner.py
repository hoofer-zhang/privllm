# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""实体识别：正则管结构化字段，LLM 管自由文本里的命名实体。

两级设计的原因：手机号/身份证有确定形态，正则又快又准，没必要动用模型；
而人名、公司名在中文里没有可靠的正则，只能用模型的语义理解。

正则优先于 LLM——同一段文字上两者冲突时，形态明确的赢。
"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import ChatBackend

# 结构化 PII。前后的 (?<!\d)/(?!\d) 防止从长数字串中间截取。
#
# 美式号码的关键取舍：格式化电话强制要求分隔符。因为中国手机号（11 位，
# 1[3-9] 开头）和"带国家码 1 的美式号码"在纯数字上是同构的——「13812345678」
# 也能被读成 1+381+234+5678。只有强制分隔符才能把两者切开。
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ID_CARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)(?:\d{4}[\s-]){3}\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[. -])?\(?\d{3}\)?[. -]\d{3}[. -]\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)[2-9]\d{9}(?!\d)")),
    ("SSN", re.compile(r"(?<!\d)\d{3}[\s-]\d{2}[\s-]\d{4}(?!\d)")),
    ("EMAIL", re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])")),
]

# 需要模型识别的类别。用户名/密码没有可靠正则，只能靠语义（"用户名是 xxx"）。
LLM_KINDS = ("PERSON", "ORG", "ADDRESS", "PROJECT", "USERNAME", "PASSWORD")

# 有"缩写指代"概念的类别，回扫补漏和别名归并才对它们生效。结构化字段
# （手机号/卡号/SSN/email）和登录凭证（用户名/密码）没有缩写指代——把密码
# 「abc123」切成「abc」去全文回扫只会误伤，所以不进这个集合。
ALIAS_KINDS = ("PERSON", "ORG", "ADDRESS", "PROJECT")

# 形态明确的类别优先，避免和模型结果打架
_PRIORITY = {"ID_CARD": 0, "BANK_CARD": 1, "PHONE": 2, "SSN": 3, "EMAIL": 4, "PERSON": 5, "ORG": 6, "ADDRESS": 7, "PROJECT": 8, "USERNAME": 9, "PASSWORD": 10}

NER_PROMPT = """你是文本脱敏系统的实体识别模块。从文本中找出所有需要脱敏的实体。

类别：
- PERSON：人名
- ORG：公司、机构、组织、部门名
- ADDRESS：地址、地点
- PROJECT：项目名、产品名、内部代号
- USERNAME：用户名、登录账号
- PASSWORD：密码、口令

要求：
1. 只输出 JSON 数组，不要任何解释或 markdown 代码块。
2. 每项格式：{"text": "原文中的片段", "type": "类别"}
3. text 必须是原文里逐字出现的片段，不要改写、不要补全。
4. 没有实体则输出 []

文本：
"""


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    start: int = -1
    end: int = -1


def regex_entities(text: str) -> list[Entity]:
    found = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            found.append(Entity(m.group(), kind, m.start(), m.end()))
    return found


def _parse_json_array(raw: str) -> list[dict]:
    """模型偶尔会套上 ```json 围栏或加前缀，这里做一次容错提取。"""
    s = raw.strip()
    if "```" in s:
        # 取第一个代码块的内容
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                s = p
                break
    lo, hi = s.find("["), s.rfind("]")
    if lo < 0 or hi < lo:
        return []
    try:
        data = json.loads(s[lo : hi + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def llm_entities(
    text: str,
    backend: "ChatBackend",
    timeout: int = 180,
) -> list[Entity]:
    """用一个语言模型抽命名实体，失败时抛 BackendError。

    调用方必须显式决定失败策略。默认（见 extract）是 fail-closed——
    NER 挂掉时若继续把原文发给 provider，脱敏层就成了摆设。
    超时给得宽是因为模型冷加载 5 GB 权重可能要几十秒。

    temperature 压到 0：抽取任务要的是稳定复现，不是创造力。
    """
    raw = backend.chat(
        [{"role": "user", "content": NER_PROMPT + text}],
        temperature=0.0,
        timeout=timeout,
    )

    found = []
    for item in _parse_json_array(raw):
        frag, kind = item.get("text"), item.get("type")
        if not isinstance(frag, str) or not frag or kind not in LLM_KINDS:
            continue
        # 模型可能改写原片，这里在所有出现位置上锚定
        for m in re.finditer(re.escape(frag), text):
            found.append(Entity(frag, kind, m.start(), m.end()))
    return found


def _resolve(candidates: list[Entity], text: str) -> list[Entity]:
    """解决重叠：按 (起点, 优先级, 长度) 排序后贪心取不冲突的。

    重叠是常态——模型会把「北京市朝阳区」整个报成 ADDRESS，同时把「朝阳区」
    也报一遍。取覆盖更全的那个。
    """
    ordered = sorted(
        candidates,
        key=lambda e: (e.start, _PRIORITY.get(e.type, 99), -(e.end - e.start)),
    )
    chosen: list[Entity] = []
    cursor = -1
    for ent in ordered:
        if ent.start >= cursor:
            chosen.append(ent)
            cursor = ent.end
    return chosen


def extract(
    text: str,
    backend: "ChatBackend",
    use_llm: bool = True,
    on_llm_failure: str = "raise",
) -> tuple[list[Entity], list[str]]:
    """完整抽取流程。返回 (按位置排序且互不重叠的实体表, 告警)。

    on_llm_failure:
      - "raise"      : 模型不可用时直接失败（默认，fail-closed）
      - "regex_only" : 降级为只用正则——手机号/身份证仍被脱敏，
                       但人名公司名会被原样放过。必须让调用方知道这一点。
    """
    from .backend import BackendError

    candidates = regex_entities(text)
    notes: list[str] = []

    if use_llm:
        try:
            candidates += llm_entities(text, backend)
        except BackendError as exc:
            if on_llm_failure == "raise":
                raise
            if on_llm_failure != "regex_only":
                raise ValueError(f"未知的 on_llm_failure：{on_llm_failure!r}") from exc
            notes.append(
                f"模型实体识别不可用（{exc}），已降级为纯正则脱敏："
                f"手机号/身份证/银行卡仍受保护，但人名、公司名等未被替换。"
            )

    return expand_aliases(_resolve(candidates, text), text), notes


def extract_local_only(text: str) -> list[Entity]:
    """只用正则，不联网。给离线测试和降级路径用。"""
    return expand_aliases(_resolve(regex_entities(text), text), text)


def _anchor(frag: str) -> str:
    """给拉丁片段加词边界，防止回扫出子词。

    中文没有词边界概念，而且 isascii() 为 False 直接跳过——中文行为零变化。
    拉丁片段如果不加边界，「Smith」的前缀「Smit」「mith」「th」会在
    「Smithsonian」中间匹配，把别人名字的字头搅进脱敏替换，还原时静默丢字。
    """
    if not (frag.isascii() and any(c.isalnum() for c in frag)):
        return re.escape(frag)
    left = r"(?<![A-Za-z0-9])" if frag[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if frag[-1].isalnum() else ""
    return left + re.escape(frag) + right


def expand_aliases(
    entities: list[Entity], text: str, min_ratio: float = 0.5
) -> list[Entity]:
    """回扫：已识别实体的缩略写法若仍以明文存在，一并纳管。

    NER 召回不完美是常态——模型报了「星海动力集团」，却漏了后文的「星海动力」。
    对脱敏来说漏检不是"少脱一点"，而是那段文字以明文原样发出。这一步拿已知实体的
    前缀/后缀去全文回扫，把漏网的缩写找回来。

    宁可过度脱敏（把「北极」也换掉）也不能漏——这是隐私工具唯一正确的偏向。
    """
    covered = [(e.start, e.end) for e in entities]
    extra: list[Entity] = []

    for ent in entities:
        # 结构化字段（手机号/身份证/卡号/SSN）没有"缩写指代"概念，跳过。
        # 否则一段数字会被切成子串，去误伤文本里别的数字。
        if ent.type not in ALIAS_KINDS:
            continue
        t = ent.text
        lowest = max(2, int(len(t) * min_ratio))
        for k in range(len(t) - 1, lowest - 1, -1):
            for frag in {t[:k], t[-k:]}:
                # 多词实体（John Smith）的前缀/后缀切片会带出边界空格（"John "、
                # " Smith"）。带边界空格的片段不是有意义的指代，还会在文本里匹配到
                # "空格+Smith" 这种幽灵实体，归并不了、派生独立假名。直接跳过。
                if frag != frag.strip():
                    continue
                for m in re.finditer(_anchor(frag), text):
                    span = (m.start(), m.end())
                    # 跳过已被现有实体覆盖的位置，避免制造重叠
                    if any(s <= span[0] and span[1] <= e for s, e in covered):
                        continue
                    extra.append(Entity(frag, ent.type, m.start(), m.end()))
                    covered.append(span)

    # 必须消重并按位置排序返回。调用方（如 proxy.redact）依赖"从后往前"遍历来保护
    # 偏移，顺序一乱，先替换的实体会把后面实体的 span 全部推歪，整段文本被搅碎。
    # _resolve 同时做了"重叠取最长"和"按位置排序"两件事，作为最后的兜底。
    return _resolve(entities + extra, text)


def canonical_map(entities: list[Entity]) -> dict[str, str]:
    """把同一实体的不同写法归并到最完整的那种写法。

    「星海动力」和「星海动力集团」是同一家公司。不归并就会派生出两个假名，
    模型会当成两家公司在谈——这比不脱敏还糟，因为它给出了错误的事实。

    规则保守：短串是长串的前缀（中文缩写），或短串是长串的末词（英文姓氏指代，
    边界检查让后缀规则只对拉丁文本生效）、类别相同、且长度比 >= 0.5 时归并。
    已知误判：「中国」与「中国银行」长度比 0.5，会被当成同一实体；英文里
    「John Smith」与「Alice Smith」各自的「Smith」会归到先出现的那个全名。
    宁可漏并（保守，只是少归并一些）也不要错并。
    """
    by_type: dict[str, set[str]] = {}
    for e in entities:
        # 结构化字段和登录凭证没有别名/缩写概念，跳过归并——否则两个用户名
        # 「admin」与「admin1」会被前缀规则误并成同一实体。
        if e.type not in ALIAS_KINDS:
            continue
        by_type.setdefault(e.type, set()).add(e.text)

    canon = {e.text: e.text for e in entities}
    for texts in by_type.values():
        ordered = sorted(texts, key=len, reverse=True)  # 最长优先
        for short in ordered:
            for long in ordered:
                if len(long) <= len(short) or len(short) / len(long) < 0.5:
                    continue
                # 前缀归并：中文的「星海动力」→「星海动力集团」。
                if long.startswith(short):
                    canon[short] = long
                    break
                # 后缀归并：英文的姓氏指代，「John Smith」后文的「Smith」是同一人。
                # 边界检查（后缀前一字符非 alnum）让它只对拉丁文本生效——中文里
                # 「集团」这类后缀前一个字符必是汉字（alnum），永不触发，也就不会
                # 把常见名词误并到「某某集团」。
                if long.endswith(short) and not long[len(long) - len(short) - 1].isalnum():
                    canon[short] = long
                    break
    return canon
