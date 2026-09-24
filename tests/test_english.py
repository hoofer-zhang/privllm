"""英文环境的脱敏自检（离线，不联网）。

覆盖：英文正则（美式电话/SSN/带空格卡号）、英文可读假名、格式保持、
姓氏指代归并、以及两个关键回归——子词切碎（英文最危险的一类静默损坏）
和无陷阱英文段落的严格无损往返。

运行（在仓库根目录）：python tests/test_english.py
"""

import pathlib
import re
import secrets
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import privllm.ner as nermod  # noqa: E402
from privllm.backend import BackendError  # noqa: E402
from privllm.ner import Entity, extract_local_only  # noqa: E402
from privllm.pseudo import EN_MARKERS, Pseudonymizer  # noqa: E402
from privllm.proxy import PrivacyProxy  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8") # pyright: ignore[reportAttributeAccessIssue]


class _NoBackend:
    """离线测试不该发生任何模型调用，真被调到时立即抛错而不是悄悄连网。"""

    def chat(self, messages, **kwargs):
        raise BackendError("离线测试不应调用后端")


NO_BACKEND = _NoBackend()

# 含美式电话、SSN、带空格卡号、人名、公司、地址、项目代号、以及一处姓氏指代。
TEXT = (
    "On Thursday, John Smith (phone +1-555-123-4567, SSN 123-45-6789) "
    "met Mary Johnson at Acme Corporation's office at 221B Baker Street, London. "
    "Smith will lead Project Falcon. Their corporate card is 4111 1111 1111 1111."
)


def _ent(src: str, frag: str, kind: str, nth: int = 0) -> Entity:
    """用 index 算偏移，别手数——数错一个字，测试就在验证错误的东西。"""
    i = -1
    for _ in range(nth + 1):
        i = src.index(frag, i + 1)
    return Entity(frag, kind, i, i + len(frag))


# 模拟英文 NER 输出：只报全称，故意漏掉后文单独的 "Smith"，
# 让回扫补漏 + 姓氏归并在真实路径上被触发。
def _sim_ner(text: str, backend, timeout: int = 180) -> list[Entity]:
    return [
        _ent(text, "John Smith", "PERSON"),
        _ent(text, "Mary Johnson", "PERSON"),
        _ent(text, "Acme Corporation", "ORG"),
        _ent(text, "221B Baker Street, London", "ADDRESS"),
        _ent(text, "Project Falcon", "PROJECT"),
    ]


nermod.llm_entities = _sim_ner

_checks = 0
_failed = 0


def check(label: str, cond: bool, detail: str = ""):
    global _checks, _failed
    _checks += 1
    if cond:
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         {detail}")


def main() -> int:
    proxy = PrivacyProxy(secrets.token_bytes(32), NO_BACKEND, structured_mode="fpe")

    print("1. 英文正则识别")
    ents = extract_local_only(TEXT)
    kinds = sorted({e.type for e in ents})
    check("抓到 PHONE / SSN / BANK_CARD", kinds == ["BANK_CARD", "PHONE", "SSN"], str(kinds))
    got = {e.text for e in ents}
    check("美式电话被识别", "+1-555-123-4567" in got)
    check("SSN 被识别", "123-45-6789" in got)
    check("带空格卡号被识别", "4111 1111 1111 1111" in got)

    print("\n2. 英文可读假名")
    pz = Pseudonymizer(secrets.token_bytes(32))
    p_en = pz("PERSON", "John Smith")
    o_en = pz("ORG", "Acme Corporation")
    check("人名假名是「名字样标记 + 真实姓氏」两词",
          bool(re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", p_en))
          and any(p_en.startswith(m + " ") for m in EN_MARKERS), repr(p_en))
    check("英文人名保留真实姓氏、只换名字", p_en.split(" ")[-1] == "Smith", repr(p_en))
    check("公司假名是「前缀 + 后缀」两词，不再带 alias",
          bool(re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", o_en))
          and not o_en.startswith("alias "), repr(o_en))
    check("中文实体仍走中文池", bool(re.search(r"[一-鿿]", pz("PERSON", "张伟"))),
          repr(pz("PERSON", "张伟")))
    # 假名与原文不同（不会把真名当假名发出）
    check("假名不等于原文", p_en not in ("John Smith", "Mary Johnson"))

    print("\n3. 端到端：完整英文段落")
    red = proxy.redact(TEXT, use_llm=True)
    by_type = {e.type: e.placeholder for e in red.entries}
    print(f"   脱敏 {red.redacted}")

    print("\n4. 格式保持")
    check("电话保持 +X-XXX-XXX-XXXX",
          bool(re.fullmatch(r"\+\d-\d{3}-\d{3}-\d{4}", by_type["PHONE"])),
          repr(by_type["PHONE"]))
    check("SSN 保持 XXX-XX-XXXX",
          bool(re.fullmatch(r"\d{3}-\d{2}-\d{4}", by_type["SSN"])),
          repr(by_type["SSN"]))
    check("卡号保持 XXXX XXXX XXXX XXXX",
          bool(re.fullmatch(r"\d{4} \d{4} \d{4} \d{4}", by_type["BANK_CARD"])),
          repr(by_type["BANK_CARD"]))

    print("\n5. 姓氏指代归并")
    personals = [e for e in red.entries if e.type == "PERSON"]
    check("两个人 -> 两条 PERSON 映射", len(personals) == 2, str([e.original for e in personals]))
    check("没有独立的 Smith 映射", all(e.original != "Smith" for e in red.entries))
    check("John Smith 是映射之一", any(e.original == "John Smith" for e in personals))
    check("共 8 条映射（5 命名实体 + 3 结构化）", len(red.entries) == 8,
          str([(e.type, e.original) for e in red.entries]))

    restored, warns = proxy.restore(red.redacted, red)
    check("后文的 Smith 还原成全名", "John Smith will lead Project Falcon" in restored, restored)
    check("还原无告警", warns == [], str(warns))

    print("\n6. 子词切碎回归（英文最危险的静默损坏）")
    src_shred = "Smith visited the Smithsonian."
    nermod.llm_entities = lambda text, backend, timeout=180: [_ent(src_shred, "Smith", "PERSON")]
    red_s = proxy.redact(src_shred, use_llm=True)
    back_s, w_s = proxy.restore(red_s.redacted, red_s)
    check("Smithsonian 未被搅碎（严格无损）", back_s == src_shred, repr(back_s))
    check("无告警", w_s == [], str(w_s))

    print("\n7. 无陷阱英文段落严格无损")
    src_plain = "John Smith met Mary Johnson at Acme Corporation."
    nermod.llm_entities = lambda text, backend, timeout=180: [
        _ent(src_plain, "John Smith", "PERSON"),
        _ent(src_plain, "Mary Johnson", "PERSON"),
        _ent(src_plain, "Acme Corporation", "ORG"),
    ]
    red_p = proxy.redact(src_plain, use_llm=True)
    back_p, _ = proxy.restore(red_p.redacted, red_p)
    check("往返逐字相等", back_p == src_plain, repr(back_p))

    print()
    if _failed:
        print(f"{_failed}/{_checks} 项失败")
        return 1
    print(f"全部 {_checks} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
