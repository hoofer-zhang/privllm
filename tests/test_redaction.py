"""脱敏链路自检（离线，不联网）。

覆盖：FF1 往返、假名确定性、跨调用一致性、还原歧义、编造号码告警。
运行（在仓库根目录）：python tests/test_redaction.py
"""

import pathlib
import secrets
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from privllm import ner  # noqa: E402
from privllm.backend import BackendError  # noqa: E402
from privllm.ner import Entity, extract_local_only  # noqa: E402
from privllm.pseudo import Pseudonymizer  # noqa: E402
from privllm.proxy import PrivacyProxy  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TEXT = (
    "负责人张伟的手机是13812345678，身份证110101199003074512X，"
    "银行卡6222021234567890123。备用联系人李静的号码是13987654321。"
)


class _NoBackend:
    """本文件全部为离线测试，不该发生任何模型调用。

    真被调到时立即抛错，而不是悄悄连网——那样测试会变成慢且不确定的东西。
    """

    def chat(self, messages, **kwargs):
        raise BackendError("离线测试不应调用后端")


NO_BACKEND = _NoBackend()

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
    master = secrets.token_bytes(32)
    proxy = PrivacyProxy(master, NO_BACKEND, structured_mode="fpe")

    print("1. 正则识别")
    ents = extract_local_only(TEXT)
    kinds = sorted({e.type for e in ents})
    check("抓到 ID_CARD / BANK_CARD / PHONE", kinds == ["BANK_CARD", "ID_CARD", "PHONE"], str(kinds))
    check("抓到 4 个实体", len(ents) == 4, f"实际 {len(ents)}：{[e.text for e in ents]}")

    print("\n2. FF1 保格式脱敏")
    red = proxy.redact(TEXT, use_llm=False)
    print(f"   原文 {TEXT}")
    print(f"   脱敏 {red.redacted}")
    check("原文的手机号已消失", "13812345678" not in red.redacted)
    check("身份证已消失", "110101199003074512X" not in red.redacted)

    by_type = {e.type: e.placeholder for e in red.entries}
    check("手机号密文仍是 11 位数字",
          len(by_type.get("PHONE", "")) == 11 and by_type.get("PHONE", "").isdigit(),
          repr(by_type.get("PHONE")))
    check("身份证密文仍是 18 位", len(by_type.get("ID_CARD", "")) == 18, repr(by_type.get("ID_CARD")))
    check("银行卡密文仍是 19 位", len(by_type.get("BANK_CARD", "")) == 19, repr(by_type.get("BANK_CARD")))

    print("\n3. 还原")
    restored, warns = proxy.restore(red.redacted, red)
    check("脱敏->还原 得到原文", restored == TEXT, f"得到 {restored!r}")
    check("无告警", warns == [], str(warns))

    print("\n4. 确定性 / 跨调用一致性")
    red2 = proxy.redact(TEXT, use_llm=False)
    check("两次脱敏结果一致", red2.redacted == red.redacted)
    check("换密钥后结果不同",
          PrivacyProxy(secrets.token_bytes(32), NO_BACKEND, structured_mode="fpe").redact(
              TEXT, use_llm=False).redacted != red.redacted)

    print("\n5. 假名确定性（不用模型，直接测派生）")
    pz = Pseudonymizer(secrets.token_bytes(32))
    a1, a2 = pz("PERSON", "张伟"), pz("PERSON", "张伟")
    b = pz("PERSON", "李静")
    check("同一实体 -> 同一假名", a1 == a2, f"{a1} vs {a2}")
    check("不同实体 -> 不同假名", a1 != b, f"{a1} vs {b}")
    check("归一化生效（全角/大小写）",
          pz("ORG", "ABC 公司") == pz("ORG", "ａｂｃ 公司"),
          f"{pz('ORG', 'ABC 公司')} vs {pz('ORG', 'ａｂｃ 公司')}")

    print("\n6. 模型输出里的编造号码应告警")
    fake_reply = f"请联系 {by_type['PHONE']}，备用号码 13700000000。"
    _, warns2 = proxy.restore(fake_reply, red)
    check("检出未映射的号码", any("13700000000" in w for w in warns2), str(warns2))

    print("\n7. FF1 定义域下限")
    try:
        proxy._ff1["PHONE"].encrypt("12345")
        check("过短输入应报错", False, "没有抛异常")
    except ValueError:
        check("过短输入报错", True)

    print("\n8. 别名归并")
    al = ner.canonical_map([
        ner.Entity("星海动力集团", "ORG"),
        ner.Entity("星海动力", "ORG"),
        ner.Entity("张伟", "PERSON"),
    ])
    check("短写法归并到长写法", al["星海动力"] == "星海动力集团", str(al))
    check("无关实体不受影响", al["张伟"] == "张伟", str(al))

    # 「中国」是「中国银行」的前缀但长度比 0.5，按规则会归并——已知误判，这里固化行为
    risky = ner.canonical_map([ner.Entity("中国银行", "ORG"), ner.Entity("中国", "ORG")])
    check("已知误判：中国/中国银行 会被归并", risky["中国"] == "中国银行", str(risky))

    print("\n9. 回扫补漏（NER 漏检的缩写不能以明文发出）")
    src = "我们收购星海动力集团。星海动力的电池业务不错，星海动力集团表示同意。"
    # 模拟 NER 只报了全称，漏掉了两处缩写
    partial = [Entity("星海动力集团", "ORG", 3, 9)]
    filled = ner.expand_aliases(partial, src)
    texts = sorted({e.text for e in filled})
    check("缩写作被回扫找回", "星海动力" in texts, str(texts))
    check("回扫结果与全称归并到同一实体",
          ner.canonical_map(filled)["星海动力"] == "星海动力集团")

    # 两字实体不做前缀回扫，否则「张伟」会退化到单字
    two = ner.expand_aliases([Entity("张伟", "PERSON", 0, 2)], "张伟和张三")
    check("两字实体不退化", {e.text for e in two} == {"张伟"}, str({e.text for e in two}))

    print("\n10. 还原不得误伤")
    from privllm.proxy import Entry, Redaction

    # 标记默认关闭（实测两个模型都会吞掉），所以这里显式开一个来测标记路径
    marked = PrivacyProxy(master, NO_BACKEND, structured_mode="fpe", mark="⟦{}⟧")
    red_a = Redaction("", "", [], entries=[Entry("刘瑶", "张伟", "PERSON")])

    # 模型写出了与假名同名的真人：带标记才区分得开
    got, _ = marked.restore("建议由刘瑶牵头，另外⟦刘瑶⟧女士也有类似经验。", red_a)
    check("带标记时同名人名不受影响",
          got == "建议由刘瑶牵头，另外张伟女士也有类似经验。", got)

    # 链式替换：A 的还原结果不得被 B 的规则再次命中
    red_b = Redaction("", "", [], entries=[
        Entry("李白", "王强", "PERSON"),
        Entry("王强", "刘瑶", "PERSON"),
    ])
    got2, _ = proxy.restore("李白负责技术，王强负责财务。", red_b)
    check("链式替换已消除（裸串模式）", got2 == "王强负责技术，刘瑶负责财务。", got2)

    _, w3 = marked.restore("备用⟦未知名⟧。", red_a)
    check("编造的占位符被告警", any("未知名" in x for x in w3), str(w3))

    # 模型把界定标记当格式清理掉（实测 gemma4:e2b / qwen3:8b 都这样）：
    # 必须降级还原并告警，不能静默地什么都不做
    got3, w4 = marked.restore("建议由刘瑶牵头。", red_a)
    check("标记被吞时降级还原", got3 == "建议由张伟牵头。", got3)
    check("降级有明确告警", any("界定标记" in x for x in w4), str(w4))

    print("\n11. 实体乱序不得搅碎文本（回归）")
    import privllm.proxy as _pm

    src_ord = "我们收购星海动力集团。星海动力的电池业务不错，星海动力集团表示同意。"

    def _span(frag, nth=0):
        """用 find 算偏移，别手数——数错一个字，测试就在验证错误的东西。"""
        i = -1
        for _ in range(nth + 1):
            i = src_ord.index(frag, i + 1)
        return i, i + len(frag)

    _real = _pm.ner.extract
    # 故意返回乱序实体表，复现旧版 expand_aliases 把回扫结果追加在末尾的行为
    _pm.ner.extract = lambda *a, **k: ([
        Entity("星海动力集团", "ORG", *_span("星海动力集团", 0)),
        Entity("星海动力", "ORG", *_span("星海动力", 1)),      # 位置靠前却排在中间
        Entity("星海动力集团", "ORG", *_span("星海动力集团", 1)),
    ], [])
    try:
        red_o = proxy.redact(src_ord, use_llm=True)
    finally:
        _pm.ner.extract = _real

    back, _ = proxy.restore(red_o.redacted, red_o)
    # 不是严格恒等：别名归并会把缩写统一成最完整的写法，这是既定行为（见 red.notes）。
    # 这里要验的是"文本没被搅碎"，所以逐字比对，只允许缩写作那处不同。
    expect = src_ord.replace("星海动力的电池", "星海动力集团的电池")
    check("乱序输入下文本不被搅碎（缩写作按归并规则统一）", back == expect, repr(back))
    check("原文实体未以明文残留",
          "星海动力" not in red_o.redacted, repr(red_o.redacted))
    check("expand_aliases 输出按位置排序",
          (lambda es: [e.start for e in es] == sorted(e.start for e in es))(
              ner.expand_aliases([Entity("星海动力集团", "ORG", 3, 9)], src_ord)))

    print("\n12. 假名前缀（让名字自带假名身份）")
    from privllm.pseudo import (
        EN_PROJECT_WORDS, EN_USERNAME_WORDS, MARKERS, PASSWORD_CHARS,
        PROJECT_WORDS, USERNAME_WORDS, strip_prefix,
    )

    pz2 = Pseudonymizer(secrets.token_bytes(32))
    p_person = pz2("PERSON", "张伟")
    marker, sep, rest = p_person.partition("·")
    check("人名假名是「名字样标记·名」",
          sep == "·" and marker in MARKERS and len(rest) == 2, repr(p_person))
    check("机构假名带「假名·」前缀", pz2("ORG", "星海动力").startswith("假名·"),
          repr(pz2("ORG", "星海动力")))
    check("strip_prefix 剥掉词前缀", strip_prefix("假名·刘瑶") == "刘瑶", str(strip_prefix("假名·刘瑶")))
    check("strip_prefix 剥掉名字样标记", strip_prefix(p_person) == rest, str(strip_prefix(p_person)))
    check("无前缀返回 None", strip_prefix("刘瑶") is None, str(strip_prefix("刘瑶")))

    # 模型把前缀剥掉只写「名」：降级还原 + 告警，而不是静默丢
    red_pfx = Redaction("", "", [], entries=[Entry(p_person, "张伟", "PERSON")])
    got_pfx, w_pfx = proxy.restore(f"建议由{p_person}牵头。", red_pfx)
    check("前缀完整时正常还原", got_pfx == "建议由张伟牵头。", got_pfx)
    check("前缀完整时无告警", w_pfx == [], str(w_pfx))

    got_pfx2, w_pfx2 = proxy.restore(f"建议由{rest}牵头。", red_pfx)
    check("前缀被剥时降级还原", got_pfx2 == "建议由张伟牵头。", got_pfx2)
    check("降级有明确告警", any("假名前缀" in x for x in w_pfx2), str(w_pfx2))

    print("\n13. 中性实体（项目代号）的语言跟随上下文")
    check("中文语境下拉丁项目代号用中文代号词",
          pz2("PROJECT", "CRP2", cjk_ctx=True) in PROJECT_WORDS,
          repr(pz2("PROJECT", "CRP2", cjk_ctx=True)))
    check("英文语境下拉丁项目代号用英文代号词",
          pz2("PROJECT", "CRP2", cjk_ctx=False) in EN_PROJECT_WORDS,
          repr(pz2("PROJECT", "CRP2", cjk_ctx=False)))
    check("实体本身是中文时无视语境仍用中文代号词",
          pz2("PROJECT", "北极星", cjk_ctx=False) in PROJECT_WORDS,
          repr(pz2("PROJECT", "北极星", cjk_ctx=False)))

    # 端到端：中文文本里脱敏拉丁项目代号，sent 里应是「项目-」前缀（走 redact 的
    # text_cjk 传递路径，而不是 Pseudonymizer 默认的 False）。
    import privllm.proxy as _pm
    src_proj = "代号 CRP2 的项目由张伟负责。"
    _real_ner = _pm.ner.extract
    _pm.ner.extract = lambda *a, **k: ([
        Entity("CRP2", "PROJECT", src_proj.index("CRP2"), src_proj.index("CRP2") + 4),
        Entity("张伟", "PERSON", src_proj.index("张伟"), src_proj.index("张伟") + 2),
    ], [])
    try:
        red_proj = proxy.redact(src_proj, use_llm=True)
    finally:
        _pm.ner.extract = _real_ner
    proj_ph = next(e.placeholder for e in red_proj.entries if e.original == "CRP2")
    check("中文文本里拉丁项目代号脱敏为中文代号词", proj_ph in PROJECT_WORDS, proj_ph)

    print("\n14. Email 保格式脱敏 + 用户名/密码假名")
    EMAIL_TEXT = "联系邮箱 zhang.wei@example.com，备用 john+tag@sub.example.co.uk。"
    ents_email = extract_local_only(EMAIL_TEXT)
    kinds_email = sorted({e.type for e in ents_email})
    check("正则抓到 EMAIL", kinds_email == ["EMAIL"], str(kinds_email))
    check("抓到 2 个邮箱", len(ents_email) == 2,
          f"实际 {len(ents_email)}：{[e.text for e in ents_email]}")

    red_em = proxy.redact(EMAIL_TEXT, use_llm=False)
    print(f"   原文 {EMAIL_TEXT}")
    print(f"   脱敏 {red_em.redacted}")
    check("原文邮箱已消失", "zhang.wei@example.com" not in red_em.redacted)
    check("原文第二个邮箱已消失", "john+tag@sub.example.co.uk" not in red_em.redacted)
    em_ph = next(e.placeholder for e in red_em.entries if e.type == "EMAIL")
    check("邮箱密文仍是 email 形态（一个 @，含 .）",
          em_ph.count("@") == 1 and "." in em_ph, repr(em_ph))
    check("邮箱密文已 lower 规范化", em_ph == em_ph.lower(), repr(em_ph))

    restored_em, warns_em = proxy.restore(red_em.redacted, red_em)
    check("邮箱脱敏->还原 得到原文", restored_em == EMAIL_TEXT, f"得到 {restored_em!r}")
    check("邮箱无告警", warns_em == [], str(warns_em))

    # 大小写不敏感：同一邮箱不同大小写 -> 同一密文
    a_up = proxy._fpe("EMAIL", "John.Doe@Example.COM")
    b_lo = proxy._fpe("EMAIL", "john.doe@example.com")
    check("同一邮箱大小写不同 -> 同一密文", a_up == b_lo, f"{a_up} vs {b_lo}")

    # 用户名/密码没有可靠正则，走 LLM 类别；假名派生离线可测
    pz3 = Pseudonymizer(secrets.token_bytes(32))
    check("中文语境用户名假名是拼音+数字",
          pz3("USERNAME", "alice_wang", cjk_ctx=True).startswith(USERNAME_WORDS),
          repr(pz3("USERNAME", "alice_wang", cjk_ctx=True)))
    check("密码假名是 10 位字母数字符号串",
          len(pz3("PASSWORD", "s3cret!pass", cjk_ctx=True)) == 10
          and all(c in PASSWORD_CHARS for c in pz3("PASSWORD", "s3cret!pass", cjk_ctx=True)),
          repr(pz3("PASSWORD", "s3cret!pass", cjk_ctx=True)))
    check("英文语境用户名假名是英文名+数字",
          pz3("USERNAME", "alice_wang", cjk_ctx=False).startswith(EN_USERNAME_WORDS),
          repr(pz3("USERNAME", "alice_wang", cjk_ctx=False)))

    # 用户名/密码没有"缩写指代"概念：不归并、不回扫
    um = ner.canonical_map([Entity("admin", "USERNAME"), Entity("admin1", "USERNAME")])
    check("用户名不归并（admin 不被 admin1 吞并）", um["admin"] == "admin", str(um))
    pw_nosweep = ner.expand_aliases(
        [Entity("abc123", "PASSWORD", 0, 6)], "密码是 abc123，另一处 abc 只是缩写。")
    check("密码不回扫子串", {e.text for e in pw_nosweep} == {"abc123"},
          str({e.text for e in pw_nosweep}))

    # 端到端：用户名/密码脱敏->还原（模拟 LLM 报出这两类实体）
    src_cred = "我的用户名是 alice_wang，密码是 s3cret!pass。"
    import privllm.proxy as _pm
    _real_ner2 = _pm.ner.extract
    _pm.ner.extract = lambda *a, **k: ([
        Entity("alice_wang", "USERNAME",
               src_cred.index("alice_wang"), src_cred.index("alice_wang") + len("alice_wang")),
        Entity("s3cret!pass", "PASSWORD",
               src_cred.index("s3cret!pass"), src_cred.index("s3cret!pass") + len("s3cret!pass")),
    ], [])
    try:
        red_cred = proxy.redact(src_cred, use_llm=True)
    finally:
        _pm.ner.extract = _real_ner2
    back_cred, w_cred = proxy.restore(red_cred.redacted, red_cred)
    check("用户名/密码脱敏->还原 得到原文", back_cred == src_cred, repr(back_cred))
    check("用户名/密码无告警", w_cred == [], str(w_cred))

    print()
    if _failed:
        print(f"{_failed}/{_checks} 项失败")
        return 1
    print(f"全部 {_checks} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
