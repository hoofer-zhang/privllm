# privllm

给任意 LLM 连接套一层**可逆脱敏**。

明文进，明文出；中间跨过网络的那一段，姓名、手机号、身份证、公司名、地址已经换成了别的值。

```python
from privllm import PrivacyProxy, load_master
from examples.ollama_backend import OllamaBackend

proxy = PrivacyProxy(load_master(), OllamaBackend(model="gemma4:e2b"))
result = proxy.chat("负责人张伟，手机 13812345678")

print(result["sent"])      # 实际发出去的：负责人翎羽·刘瑶，手机 76601293345
print(result["restored"])  # 拿回来的：张伟的手机号是 13812345678
```

`privllm` 本身不连模型。它只认一个 `ChatBackend` 协议——送一轮对话，拿回助手文本。接 Ollama、vLLM、OpenAI 还是公司内部网关，由你注入。仓库里的 [examples/ollama_backend.py](examples/ollama_backend.py) 是一个完整实现，可以直接当模板改。

---

## 目录

- [应用场景](#应用场景)
- [为什么不能用"加密 prompt"解决](#为什么不能用加密-prompt-解决)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [工作流程](#工作流程)
- [API](#api)
- [接入自己的后端](#接入自己的后端)
- [已知限制](#已知限制)
- [测试](#测试)
- [目录结构](#目录结构)

更多文档：[DESIGN.md](DESIGN.md)（核心设计） · [NOTES.md](NOTES.md)（注意事项）

---

## 应用场景

**问题：调用外部大模型 API 时，数据会完整地离开你的信任边界。**

把 prompt 发给 OpenAI、Claude、通义、DeepSeek 这类第三方服务，等于把内容交给一个你看不见、无法审计的对方：

- 请求会被记录、可能被用于监控或训练——姓名、手机号、身份证、银行卡、内部项目代号，一旦发出就收不回；
- 《个人信息保护法》《数据安全法》、GDPR 都要求对个人信息最小化处理，把客户手机号直接发给第三方模型，等于把合规责任外包给一个黑盒；
- 员工图方便把合同、财报、病历、代码贴进对话框，内容就此落到服务方服务器上。

**两难**：想用大模型的能力，又不想把敏感数据交出去。

`privllm` 解决的就是这个两难：数据离开你的机器之前，先把敏感实体（姓名、电话、证件号、邮箱、地址、内部代号、登录凭证）替换成无法反解的假名或密文，服务方看到的文本本身就不含机密；拿到回答后再换回真值，你这边全程无感。

典型场景：

- **客服 / CRM**：把客户咨询喂给外部模型做摘要、意图识别，但客户姓名、电话不能外发。
- **财务 / 人事 / 法务**：合同、报表、简历里的个人信息要模型帮忙处理，又受合规约束。
- **研发**：把内部日志、报错、代码贴给模型排查，代码里可能嵌着内网地址、密钥、客户数据。
- **医疗 / 金融**：受最强监管的行业，任何个人信息流出都要能说清"发给谁、能不能还原"。

---

## 为什么不能用"加密 prompt"解决

第一直觉是：把 prompt 加密发过去，让模型解密、回答、再加密回来。

这条路走不通，而且不是工程问题，是数学问题。模型要生成下一个 token，就必须对明文概率分布做 softmax。密文上算出的 softmax 和明文上算出的不是同一个东西。想让模型"在密文上计算"，只有三条路：

| 方案 | 现状 |
|---|---|
| 全同态加密（CKKS / TFHE） | 需要把 GELU、softmax 换成多项式近似，实测比明文慢 100–1000 倍。7B 模型上不可用 |
| TEE（Intel TDX / NVIDIA H100 CC） | 唯一今天就能跑 7B 级模型的方案。但它防的是宿主机，不防模型服务方——服务方在 TEE 内部依然看得见明文 |
| MPC（安全多方计算） | 每生成一个 token 要跨方通信几十轮。吞吐低到不适合交互 |

所以**如果威胁模型是 LLM 服务提供方**，纯密码学没有答案。这也是为什么业界实际在用的是另一条路：**在客户端把敏感信息替换掉，让服务方看到的文本本身就不含机密；拿到回答后再换回来**。

`privllm` 做的是这一层。它不假装能保护全部内容——能保护什么、不能保护什么，见[已知限制](#已知限制)和 [NOTES.md](NOTES.md)。

---

## 特性

- **可逆**。脱敏后的文本发出去，回答回来时自动还原成真值。
- **保格式**。手机号密文还是 11 位数字，身份证还是 18 位，银行卡还是 19 位，邮箱还是 `xxx@xxx.xxx` 的形态。数据要过校验、要落库、要喂给别的系统时不会崩。
- **可读假名**。人名换成"翎羽·刘瑶"而不是 `PERSON_a3f2`。模型的推理质量不会因为看到怪 token 而下降，而且「翎羽·」这种名字样标记让模型把整个占位符当一个名字原样保留。
- **中英双语**。假名按实体脚本自动切换——`John Smith → Sparrow Smith`、`Acme Corporation → Titan Dynamics`，中文实体仍走中文池。结构化字段同时覆盖中国大陆格式（手机号/身份证）和美式格式（`+1-555-123-4567`、SSN `123-45-6789`、`4111 1111 1111 1111`）。
- **确定性**。同一实体在同一密钥下永远映射到同一个假名，所以模型能正确做指代消解——它知道第二段的"刘瑶"就是第一段的"刘瑶"。
- **两级识别**。手机号/身份证/银行卡/邮箱走正则（快、准、不花钱），人名/公司名/地址/项目代号/用户名/密码走模型（中文里没有可靠的正则）。
- **失败关闭**。实体识别挂掉时默认直接抛错，不会静默地把明文原样转发出去。
- **无状态**。假名由 HMAC 派生，不需要预先建表。换台机器、重开会话，只要密钥相同，映射就相同。

---

## 安装

### 环境要求

- **Python 3.10+**。代码用了 `X | Y` 联合类型注解语法（PEP 604），3.9 及以下运行不了。
- **操作系统**：Windows / macOS / Linux 均可，无系统级依赖。注意 Windows 上密钥文件的保护依赖目录 ACL（`chmod` 基本无效），详见 [NOTES.md](NOTES.md)。
- **依赖**：
  - `cryptography` —— 必需。用于 FF1 的 AES、HKDF 子密钥派生、映射表的 AES-GCM 落盘。
  - `requests` —— 仅 `examples/ollama_backend.py` 需要；库本身（`privllm/`）不依赖它，只做脱敏不调模型就不必装。

```bash
pip install cryptography requests
```

```bash
git clone https://github.com/hoofer-zhang/privllm.git
cd privllm
```

也可以作为库安装（已带 `pyproject.toml`）：

```bash
pip install .             # 常规安装
pip install -e .          # 开发态：改代码即时生效
pip install .[examples]   # 连同 Ollama 示例后端需要的 requests 一起装
```

或者不安装，直接从仓库目录 import。

首次运行会自动生成主密钥到 `~/.privllm/master.key`。也可以用环境变量指定：

```bash
export PRIVLLM_MASTER_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
```

> 主密钥是这套方案的全部安全性所在。它一旦泄露，脱敏字段可以被反解、假名可以被字典攻击。**绝不要把它发给模型服务方，也不要提交进 git。**

---

## 快速开始

### 端到端跑一遍

```bash
# 默认连内网 Ollama 的 gemma4:e2b
python examples/demo_ollama.py

# 换成你自己的
OLLAMA_URL=http://localhost:11434 MODEL=qwen3:8b python examples/demo_ollama.py
```

这个 demo 会打印七个阶段：原文 → 识别到的实体 → **实际发出去的内容** → 本地映射表 → 模型的原始回复 → 还原后的回复 → 残余泄露分析。

第 ③ 段和第 ⑦ 段值得对着看：前者告诉你挡住了什么，后者告诉你没挡住什么。

### 只用 `redact` / `restore`（提示词脱敏，结果解密）

整个用法就两步，一条线走完：

```
明文提示词 ── redact() ──► 脱敏提示词 ── 发给 LLM ──► LLM 返回的回复
                                                          │
明文回复 ◄── restore() ───────────────────────────────────┘
```

1. **提示词脱敏**：`redact()` 把要发给 LLM 的提示词里的敏感信息替换成假名/密文；
2. **结果解密**：`restore()` 把 LLM 返回结果里的假名/密文换回真值。

```python
from privllm import PrivacyProxy, load_master
from examples.ollama_backend import OllamaBackend

# backend 只用于 redact 里的「实体识别」(NER)，不用于生成回复。
# 结构化字段（手机号/身份证/邮箱）走正则、不碰 backend；人名/公司名才需要 NER。
proxy = PrivacyProxy(load_master(), OllamaBackend(model="qwen3:8b"))

# ① 提示词脱敏
red = proxy.redact("负责人张伟，手机 13812345678")
print(red.redacted)   # 负责人翎羽·刘瑶，手机 76601293345 —— 把这个作为提示词发给 LLM

# ② 把脱敏后的提示词发给 LLM（任何 SDK、任何参数）
raw = your_llm_client.generate(red.redacted, model="...", temperature=0.7, stream=True)

# ③ 结果解密
reply, warnings = proxy.restore(raw, red)
print(reply)          # 张伟的手机号是 13812345678
```

`restore()` 完全在本地、不碰网络。第二个返回值 `warnings` 务必处理——LLM 改写占位符导致解不回来的地方都会列在这里，绝不静默失败。

仓库自带的 `chat()` 只是把这三步封装成一个方法；不用它，就用上面两个函数。

只处理结构化字段、不想为 NER 配模型？见下一节，`use_llm=False` 全程不联网。

### 纯结构化字段（正则，不联网）

```python
from privllm import PrivacyProxy, load_master

proxy = PrivacyProxy(load_master(), backend=None)

red = proxy.redact("手机 13812345678，身份证 110101199003074512", use_llm=False)
print(red.redacted)   # 手机 76601293345，身份证 544069721742615292
print(red.entries)    # 本地映射表

back, warnings = proxy.restore(red.redacted, red)
assert back == "手机 13812345678，身份证 110101199003074512"
```

`use_llm=False` 时完全不联网，只用正则——适合结构化数据管道。注意这个模式下**人名不会被替换**（"张伟"原样留下），因为中文人名没有可靠的正则，只能靠模型。要处理自由文本就得上模型：

```python
red = proxy.redact("负责人张伟，手机 13812345678")   # use_llm 默认为 True
```

### 落盘审计

假名派生是无状态的，理论上不需要存映射表。但"理论上"不等于"运维上"——出事时要能查"当时到底把谁换成了谁"。

```python
from privllm import save_vault, load_vault

save_vault("audit.vault", red, proxy.keys["vault"])
entries = load_vault("audit.vault", proxy.keys["vault"])
```

映射表用 AES-GCM 加密，且刻意只接受 `vault` 这一支子密钥而不是主密钥——落盘逻辑不需要有能力反解脱敏字段或攻击假名。

---

## 工作流程

```
                      客户端信任域                               │      服务方
                                                               │
  明文 ──► 实体识别 ──► 替换 ──► 脱敏文本 ──────────────────────┼──►   LLM
            │            │                                     │       │
            │            └──► 映射表（留在本地）                │       │
            │                                                  │       │
  明文 ◄── 还原 ◄──────────────────────────────────────────────┼──◄ 回复
                                                               │
                                   密钥和数据从不跨越这条边界 ──┘
```

跨过 `ChatBackend` 边界的只有脱敏后的文本。明文和主密钥从头到尾没有离开过这个进程。

---

## API

### `PrivacyProxy(master_key, backend, structured_mode="fpe", mark=None)`

| 参数 | 说明 |
|---|---|
| `master_key` | 32 字节主密钥 |
| `backend` | 实现 `ChatBackend` 的对象；只做脱敏时传 `None` |
| `structured_mode` | `"fpe"` = 密文保持原格式；`"token"` = 换成 `[PHONE_1]` 占位符 |
| `mark` | 占位符界定标记，如 `"⟦{}⟧"`。默认关闭——默认信号是假名自带的前缀 |

### 方法

```python
proxy.redact(text, use_llm=True, on_ner_failure="raise", merge_aliases=True) -> Redaction
proxy.restore(text, redaction) -> (str, list[str])   # (还原后文本, 告警)
proxy.chat(user_text, system=None, use_llm_ner=True, **backend_options) -> dict
```

`chat()` 返回每一阶段的中间结果，便于审计：

```python
{
    "sent":      str,        # 实际发给服务方的文本
    "entities":  list[Entity],
    "entries":   list[Entry],   # 本地映射表
    "raw_reply": str,        # 模型原样回复（含假名）
    "restored":  str,        # 还原后的回复
    "notes":     list[str],  # 过程性说明，无需处理
    "warnings":  list[str],  # 需要人看一眼的异常
}
```

**`warnings` 不要忽略。** 它包含几类真实问题：模型编造了映射表里没有的号码、模型把占位符写坏了导致还原不回来、界定标记或假名前缀被吞掉导致匹配精度下降（会退回裸串匹配）。

### 密钥与落盘

```python
from privllm import derive, derive_all, load_master, save_vault, load_vault

master = load_master()               # 环境变量 > 文件 > 生成
keys = derive_all(master)            # {"ff1": b"...", "pseudo": b"...", "vault": b"..."}
```

---

## 接入自己的后端

实现一个方法就行：

```python
from privllm import BackendError, Message


class MyBackend:
    def chat(self, messages: list[Message], *, temperature=None, timeout=180) -> str:
        try:
            resp = my_sdk.generate(messages, temperature=temperature, timeout=timeout)
        except MySDKError as exc:
            raise BackendError(f"调用失败：{exc}") from exc
        return resp.text
```

两条要求：

1. **失败时抛 `BackendError`。** privllm 靠它区分"模型说不了"（可降级）和"程序有 bug"（该炸出来）。别把 SDK 的原生异常泄漏上来。
2. **长回答用流式。** 非流式下读超时卡在整个生成上——本地 CPU 推理出一段长回答必然触发；流式下超时按分块算，只要还在吐字就不会断。这个坑在 [examples/ollama_backend.py](examples/ollama_backend.py) 里有完整处理。

`ChatBackend` 是 `runtime_checkable` 的 Protocol，不需要继承什么：

```python
isinstance(MyBackend(), ChatBackend)  # True
```

---

## 已知限制

脱敏不是加密。最要紧的一条：**还原不是万无一失的**。

模型把占位符**改写**掉时（「青丘市扶摇路70号」简写成「青丘市」、「翎羽·刘瑶」写成「刘瑶」），精确匹配对不上，那条信息就还原不回来。

对应策略：`restore()` 会把所有还原不回来的地方列进 `warnings`，**绝不静默失败**——所以务必处理 `warnings`，不要忽略它。换模型时也留意：实测 gemma4:e2b / qwen3:8b 会吞掉 `⟦⟧` 界定标记、偶尔剥掉中文人名的名字样前缀，这些都会触发降级告警。

其余限制——自由文本的语义不受保护、假名可被字典攻击、英文姓氏归并歧义、邮箱大小写被规范化等——完整清单见 [NOTES.md](NOTES.md)。

---

## 测试

```bash
python tests/test_ff1.py         # 对照 NIST SP 800-38G 官方向量验证 FF1
python tests/test_redaction.py   # 中文离线自检，不联网
python tests/test_english.py     # 英文离线自检，不联网
```

`test_ff1.py` 对照标准向量验证 FF1 的加解密、长度保持、以及 radix=10/36 下的正确性。

`test_redaction.py` 覆盖：正则识别、保格式、往返还原、确定性与跨调用一致性、假名派生与归一化、编造号码告警、FF1 定义域下限、别名归并与已知误判、回扫补漏、还原不得误伤（含链式替换、标记降级、假名前缀降级）、邮箱保格式与用户名/密码假名、以及一个"实体乱序不得搅碎文本"的回归测试。

`test_english.py` 覆盖：英文正则（美式电话/SSN/带空格卡号）、英文可读假名、格式保持、姓氏指代归并（`John Smith` 后文的 `Smith` 归并到同一假名）、以及两个关键回归——子词切碎（英文最危险的一类静默损坏）和无陷阱英文段落的严格无损往返。

所有文件都用纯 `assert` 风格，不依赖 pytest，退出码即结果。

---

## 目录结构

```
.
├── privllm/              # 库本身：只管脱敏与还原，不含任何模型连接代码
│   ├── __init__.py       # 公开 API
│   ├── backend.py        # ChatBackend 协议 + BackendError
│   ├── ff1.py            # FF1 保格式加密（NIST SP 800-38G）
│   ├── keys.py           # HKDF 子密钥派生
│   ├── ner.py            # 两级实体识别、回扫补漏、别名归并
│   ├── pseudo.py         # 可读假名派生
│   └── proxy.py          # 脱敏代理、单遍还原、映射表加密落盘
├── examples/             # 具体 LLM 连接，不属于库
│   ├── ollama_backend.py # ChatBackend 的 Ollama 实现
│   └── demo_ollama.py    # 端到端演示
├── tests/
│   ├── test_ff1.py
│   ├── test_redaction.py
│   └── test_english.py
├── README.md             # 概览 + 外部可用 API（中文）
├── README.en.md          # 概览 + 外部可用 API（英文）
├── DESIGN.md             # 核心设计与内部开发事项
└── NOTES.md              # 注意事项（完整已知限制）
```

`privllm/` 里刻意不出现任何 URL、API key、模型名——所以也不存在"密钥不小心跟着请求发出去"的路径。

---

## 许可

MIT
© 上海铁锹信息科技有限公司（Shanghai Spade-Tech Information Technology Co., Ltd.）
联络：zhang@spade-tec.com
