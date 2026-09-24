"""端到端演示：明文 -> 脱敏 -> LLM -> 还原。

运行（在仓库根目录）：
    python examples/demo_ollama.py
    OLLAMA_URL=http://localhost:11434 MODEL=qwen3:8b python examples/demo_ollama.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from privllm import PrivacyProxy, load_master  # noqa: E402

from examples.ollama_backend import OllamaBackend  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.1.21:11434")
MODEL = os.environ.get("MODEL", "gemma4:e2b")

TEXT = """我们计划以 3.2 亿元收购星海动力集团的电池业务。这次交易由投资部的张伟负责，
他的手机号是 13812345678，身份证 110101199003074512。对方对接人是星海动力的
财务总监李静，公司在上海市浦东新区张江路 88 号。项目内部代号叫"北极星"。
资料已发到李静的邮箱 jing.li@starpower.com，OA 登录名 lijing，初始密码 St@rPower2024。
请帮我把这桩交易的主要风险点列一下。"""


def rule(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    backend = OllamaBackend(base_url=OLLAMA_URL, model=MODEL)
    proxy = PrivacyProxy(load_master(), backend)

    rule("① 原始明文（只应存在于客户端）")
    print(TEXT)

    # 只跑一次 NER：chat() 的返回值里带着每一阶段的中间结果，够本节全部展示用。
    try:
        result = proxy.chat(
            TEXT,
            system="你是并购顾问。回答时先完整复述对方的关键信息（手机号、身份证、"
                   "公司地址、邮箱、OA 登录名、初始密码、项目代号），再列出主要风险点。",
            use_llm_ner=True,
        )
    except Exception as exc:
        print(f"\n调用失败：{exc}")
        return 1

    if result["notes"]:
        rule("提示")
        for n in result["notes"]:
            print(f"  {n}")

    rule("② 实体识别结果")
    if not result["entities"]:
        print("（未识别到实体）")
    for e in result["entities"]:
        print(f"  {e.type:10s} {e.text}")

    rule("③ 实际发给 provider 的内容 —— 对照上面看泄露了什么")
    print(result["sent"])

    rule("④ 映射表（留在本地，绝不上传）")
    for e in result["entries"]:
        print(f"  {e.original:28s} -> {e.placeholder}   [{e.type}]")

    rule("⑤ 模型的原始回复（provider 视角的输入输出）")
    print(result["raw_reply"])

    rule("⑥ 还原后的回复（客户端视角）")
    print(result["restored"])

    if result["warnings"]:
        rule("⚠ 告警")
        for w in result["warnings"]:
            print(f"  {w}")

    rule("⑦ 残余泄露分析 —— 这一步才是重点")
    print("provider 虽然看不到姓名、号码、邮箱和登录凭证，但仍能完整读出：")
    print("  · 这是一起 3.2 亿元的并购")
    print("  · 标的是「星海动力集团」的电池业务（ORG 替换后仍是同一个实体）")
    print("  · 交易正在进行中，需要风险评估")
    print()
    print("即：结构化 PII 挡住了，但「内容本身即机密」的部分挡不住。")
    print("这类文本若真不能外发，只能本地推理（TEE 或直接跑本地模型）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
