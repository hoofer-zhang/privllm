# privllm

A **reversible redaction** layer for any LLM connection.

Plain text in, plain text out; across the network hop in between, names, phone numbers, ID numbers, company names, and addresses have already been swapped for something else.

```python
from privllm import PrivacyProxy, load_master
from examples.ollama_backend import OllamaBackend

proxy = PrivacyProxy(load_master(), OllamaBackend(model="gemma4:e2b"))
result = proxy.chat("Contact: John Smith, phone 13812345678")

print(result["sent"])      # what actually goes out: Contact: Sparrow Smith, phone 76601293345
print(result["restored"])  # what comes back: John Smith's phone number is 13812345678
```

`privllm` itself never talks to a model. It only knows one `ChatBackend` protocol — send a turn of conversation, get the assistant's text back. Whether that's Ollama, vLLM, OpenAI, or an internal gateway is yours to plug in. [examples/ollama_backend.py](examples/ollama_backend.py) is a complete implementation you can use as a template.

---

## Contents

- [Use cases](#use-cases)
- [Why not "encrypt the prompt"](#why-not-encrypt-the-prompt)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [API](#api)
- [Bring your own backend](#bring-your-own-backend)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [Layout](#layout)

More docs: [DESIGN.md](DESIGN.md) (core design, Chinese) · [NOTES.md](NOTES.md) (caveats, Chinese)

---

## Use cases

**The problem: calling an external LLM API means your data leaves your trust boundary.**

Sending a prompt to a third-party service — OpenAI, Claude, Qwen, DeepSeek — is handing your content to a party you can't see or audit:

- Requests get logged, and may be used for monitoring or training. Names, phone numbers, ID numbers, bank cards, internal project codenames — once sent, you don't get them back.
- Privacy laws (GDPR, PIPL, and equivalents) require data minimization. Forwarding a customer's phone number to a third-party model outsources your compliance obligations to a black box.
- Employees casually paste contracts, financials, medical records, and code into a chat box, and that content lands on the provider's servers.

**The dilemma**: you want the model's capabilities without handing over sensitive data.

`privllm` resolves that dilemma: before data leaves your machine, it replaces sensitive entities (names, phones, ID numbers, emails, addresses, internal codenames, login credentials) with pseudonyms or ciphertext that no one else can reverse. The provider only ever sees text that contains no secrets; after the reply comes back, the true values are swapped back in, invisibly on your side.

Typical scenarios:

- **Support / CRM**: summarize or classify customer queries with an external model, without sending customer names or phone numbers out.
- **Finance / HR / Legal**: contracts, reports, and résumés need a model's help, but are bound by compliance rules.
- **Engineering**: paste internal logs, stack traces, and code to a model for debugging — code that may embed internal addresses, keys, and customer data.
- **Healthcare / Finance**: the most regulated industries, where any personal-data outflow must be able to answer "who did we send it to, and can it be reversed".

---

## Why not "encrypt the prompt"

The first instinct is: encrypt the prompt, have the model decrypt it, answer, and encrypt back.

This path doesn't work — and it's not an engineering problem, it's math. To generate the next token, a model must compute a softmax over the plaintext probability distribution. The softmax computed over ciphertext is not the same thing as the one over plaintext. Getting the model to "compute on ciphertext" leaves exactly three routes:

| Approach | Status |
|---|---|
| Fully homomorphic encryption (CKKS / TFHE) | Requires replacing GELU/softmax with polynomial approximations, empirically 100–1000× slower than plaintext. Unusable on 7B models |
| TEE (Intel TDX / NVIDIA H100 CC) | The only option that runs 7B-scale models today. But it defends against the host, not the model provider — the provider still sees plaintext inside the TEE |
| MPC (secure multi-party computation) | Each generated token needs dozens of cross-party communication rounds. Throughput too low for interaction |

So **if the threat model is the LLM provider**, pure cryptography has no answer. That's why the industry actually uses the other path: **replace the sensitive information on the client side, so the text the provider sees contains no secrets to begin with; swap it back after the reply returns**.

`privllm` is exactly that layer. It doesn't pretend to protect everything — see [Known limitations](#known-limitations) and [NOTES.md](NOTES.md) for what it can and cannot protect.

---

## Features

- **Reversible.** The redacted text goes out; when the reply comes back, true values are restored automatically.
- **Format-preserving.** An encrypted phone number is still 11 digits, an ID number still 18, a bank card still 19, an email still looks like `xxx@xxx.xxx`. Data passing through validation, being stored, or fed to other systems won't break.
- **Readable pseudonyms.** Names become "翎羽·刘瑶" instead of `PERSON_a3f2`. The model's reasoning quality doesn't degrade from seeing weird tokens, and name-like markers such as 「翎羽·」 make the model keep the whole placeholder as one name.
- **Chinese & English.** Pseudonyms switch with the entity's script — `John Smith → Sparrow Smith`, `Acme Corporation → Titan Dynamics`; Chinese entities still use the Chinese pools. Structured fields cover both mainland-China formats (phone / ID) and US formats (`+1-555-123-4567`, SSN `123-45-6789`, `4111 1111 1111 1111`).
- **Deterministic.** The same entity maps to the same pseudonym under the same key, so the model resolves coreference correctly — it knows the "刘瑶" in the second paragraph is the same "刘瑶" from the first.
- **Two-tier recognition.** Phone / ID / bank card / email use regex (fast, accurate, free); names / companies / addresses / project codenames / usernames / passwords use the model (Chinese has no reliable regex for these).
- **Fail-closed.** If entity recognition fails, it raises by default rather than silently forwarding plaintext.
- **Stateless.** Pseudonyms are derived via HMAC — no prebuilt table needed. A different machine or a fresh session yields the same mapping as long as the key matches.

---

## Installation

### Requirements

- **Python 3.10+**. The code uses the `X | Y` union-type annotation syntax (PEP 604); 3.9 and below will not run it.
- **OS**: Windows / macOS / Linux all work, with no system-level dependencies. Note that on Windows, key-file protection relies on directory ACLs (`chmod` is effectively a no-op); see [NOTES.md](NOTES.md).
- **Dependencies**:
  - `cryptography` — required. Used for FF1's AES, HKDF subkey derivation, and AES-GCM persistence of the mapping table.
  - `requests` — only needed by `examples/ollama_backend.py`; the library itself (`privllm/`) doesn't depend on it, so redact-only usage doesn't need it.

```bash
pip install cryptography requests
```

```bash
git clone https://github.com/hoofer-zhang/privllm.git
cd privllm
```

You can also install it as a library (a `pyproject.toml` is included):

```bash
pip install .             # regular install
pip install -e .          # editable: code changes take effect immediately
pip install .[examples]   # also installs requests for the Ollama example backend
```

Or skip installation and import straight from the repo directory.

On first run a master key is generated at `~/.privllm/master.key`. You can also supply one via the environment:

```bash
export PRIVLLM_MASTER_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
```

> The master key is the entire security of this scheme. If it leaks, redacted fields can be decrypted and pseudonyms dictionary-attacked. **Never send it to the model provider, and never commit it to git.**

---

## Quick start

### End-to-end run

```bash
# defaults to the on-prem Ollama model gemma4:e2b
python examples/demo_ollama.py

# point it at your own
OLLAMA_URL=http://localhost:11434 MODEL=qwen3:8b python examples/demo_ollama.py
```

The demo prints seven stages: original text → recognized entities → **what actually gets sent** → the local mapping table → the model's raw reply → the restored reply → residual-leak analysis.

Stages ③ and ⑦ are worth reading side by side: the former shows what you blocked, the latter what you didn't.

### Just `redact` / `restore` (redact the prompt, decrypt the reply)

The whole usage is two steps, one straight line:

```
plaintext prompt ── redact() ──► redacted prompt ── send to LLM ──► LLM's reply
                                                                      │
plaintext reply ◄── restore() ────────────────────────────────────────┘
```

1. **Redact the prompt**: `redact()` swaps the sensitive bits of the prompt you're about to send for pseudonyms/ciphertext;
2. **Decrypt the reply**: `restore()` swaps the pseudonyms/ciphertext in the LLM's reply back to the real values.

```python
from privllm import PrivacyProxy, load_master
from examples.ollama_backend import OllamaBackend

# The backend is only used for entity recognition (NER) inside redact() — not for
# generating the reply. Structured fields (phone / ID / email) go through regex and
# never touch the backend; free-text entities like names/companies need NER.
proxy = PrivacyProxy(load_master(), OllamaBackend(model="qwen3:8b"))

# ① Redact the prompt
red = proxy.redact("负责人张伟，手机 13812345678")
print(red.redacted)   # 负责人翎羽·刘瑶，手机 76601293345 — send this as the prompt

# ② Send the redacted prompt to the LLM (any SDK, any parameters)
raw = your_llm_client.generate(red.redacted, model="...", temperature=0.7, stream=True)

# ③ Decrypt the reply
reply, warnings = proxy.restore(raw, red)
print(reply)          # 张伟的手机号是 13812345678
```

`restore()` is fully local and never touches the network. Always handle its second return value, `warnings` — every placeholder the LLM mangled and thus couldn't be decrypted is listed there, and it never fails silently.

The bundled `chat()` just wraps these three steps into one method; if you don't want it, use the two functions above.

Only handling structured fields and don't want to set up a model for NER? See the next section — `use_llm=False` runs entirely offline.

### Structured fields only (regex, offline)

```python
from privllm import PrivacyProxy, load_master

proxy = PrivacyProxy(load_master(), backend=None)

red = proxy.redact("手机 13812345678，身份证 110101199003074512", use_llm=False)
print(red.redacted)   # 手机 76601293345，身份证 544069721742615292
print(red.entries)    # local mapping table

back, warnings = proxy.restore(red.redacted, red)
assert back == "手机 13812345678，身份证 110101199003074512"
```

With `use_llm=False` no network is touched — regex only, for structured data pipelines. Note that **names are not replaced** in this mode ("张伟" passes through untouched), because Chinese names have no reliable regex and need the model. For free text, use the model:

```python
red = proxy.redact("负责人张伟，手机 13812345678")   # use_llm defaults to True
```

### Persisting an audit trail

Pseudonym derivation is stateless, so in theory you don't need to store the mapping. But "in theory" isn't "in operations" — when something goes wrong, you need to answer "who was mapped to what".

```python
from privllm import save_vault, load_vault

save_vault("audit.vault", red, proxy.keys["vault"])
entries = load_vault("audit.vault", proxy.keys["vault"])
```

The mapping table is AES-GCM-encrypted, and deliberately accepts only the `vault` subkey rather than the master key — the persistence layer has no business being able to decrypt redacted fields or attack pseudonyms.

---

## How it works

```
                    client trust domain                          │      provider
                                                                 │
  plaintext ──► NER ──► replace ──► redacted text ──────────────┼──►   LLM
                │          │                                     │       │
                │          └──► mapping table (stays local)      │       │
                │                                                │       │
  plaintext ◄── restore ◄───────────────────────────────────────┼──◄ reply
                                                                 │
                             keys and data never cross ──────────┘
```

Only the redacted text crosses the `ChatBackend` boundary. Plaintext and the master key never leave this process.

---

## API

### `PrivacyProxy(master_key, backend, structured_mode="fpe", mark=None)`

| Parameter | Description |
|---|---|
| `master_key` | 32-byte master key |
| `backend` | An object implementing `ChatBackend`; pass `None` to redact only |
| `structured_mode` | `"fpe"` = ciphertext keeps its original format; `"token"` = replace with `[PHONE_1]` placeholders |
| `mark` | Placeholder delimiter, e.g. `"⟦{}⟧"`. Off by default — the default signal is the pseudonym's own prefix |

### Methods

```python
proxy.redact(text, use_llm=True, on_ner_failure="raise", merge_aliases=True) -> Redaction
proxy.restore(text, redaction) -> (str, list[str])   # (restored text, warnings)
proxy.chat(user_text, system=None, use_llm_ner=True, **backend_options) -> dict
```

`chat()` returns every intermediate stage for auditing:

```python
{
    "sent":      str,        # text actually sent to the provider
    "entities":  list[Entity],
    "entries":   list[Entry],   # local mapping table
    "raw_reply": str,        # the model's raw reply (with pseudonyms)
    "restored":  str,        # the restored reply
    "notes":     list[str],  # informational notes, no action needed
    "warnings":  list[str],  # anomalies that need a human look
}
```

**Do not ignore `warnings`.** They carry several real problems: the model fabricated a number that isn't in the mapping table, the model mangled a placeholder so it can't be restored, or the delimiter/pseudonym prefix was dropped so matching precision degraded (falling back to bare-string matching).

### Keys and persistence

```python
from privllm import derive, derive_all, load_master, save_vault, load_vault

master = load_master()               # env var > file > generate
keys = derive_all(master)            # {"ff1": b"...", "pseudo": b"...", "vault": b"..."}
```

---

## Bring your own backend

Implement one method:

```python
from privllm import BackendError, Message


class MyBackend:
    def chat(self, messages: list[Message], *, temperature=None, timeout=180) -> str:
        try:
            resp = my_sdk.generate(messages, temperature=temperature, timeout=timeout)
        except MySDKError as exc:
            raise BackendError(f"call failed: {exc}") from exc
        return resp.text
```

Two requirements:

1. **Raise `BackendError` on failure.** privllm relies on it to distinguish "the model can't answer" (degradable) from "there's a bug" (should blow up). Don't leak your SDK's native exceptions.
2. **Stream long replies.** Non-streaming read timeouts block on the whole generation — local CPU inference on a long reply will trip it; streaming timeouts apply per chunk, so it won't cut out while tokens are still flowing. This is handled fully in [examples/ollama_backend.py](examples/ollama_backend.py).

`ChatBackend` is a `runtime_checkable` Protocol — no inheritance needed:

```python
isinstance(MyBackend(), ChatBackend)  # True
```

---

## Known limitations

Redaction is not encryption. The one that matters most: **restoration is not guaranteed**.

When the model **rewrites** a placeholder ("青丘市扶摇路70号" shortened to "青丘市", or "翎羽·刘瑶" written as "刘瑶"), exact matching no longer lines up and that piece of information cannot be restored.

The corresponding strategy: `restore()` lists every unrecoverable spot in `warnings` and **never fails silently** — so always handle `warnings`, don't ignore them. When switching models, note that gemma4:e2b / qwen3:8b drop `⟦⟧` delimiters and occasionally strip the name-like prefix off Chinese names; both trigger the degraded warning.

Other limitations — free-text semantics are unprotected, pseudonyms are dictionary-attackable, English surname-merge ambiguity, email case normalization — are enumerated in [NOTES.md](NOTES.md).

---

## Tests

```bash
python tests/test_ff1.py         # verify FF1 against NIST SP 800-38G official vectors
python tests/test_redaction.py   # offline self-checks, Chinese; no network
python tests/test_english.py     # offline self-checks, English; no network
```

`test_ff1.py` verifies FF1 encryption/decryption, length preservation, and correctness at radix=10/36 against the standard vectors.

`test_redaction.py` covers: regex recognition, format preservation, round-trip restoration, determinism and cross-call consistency, pseudonym derivation and normalization, fabricated-number warnings, the FF1 domain minimum, alias merging and known false positives, sweep-back recall, restore-must-not-overreach (chained replacement, marker degradation, prefix degradation), email format preservation and username/password pseudonyms, and a regression for "entity reordering must not mangle text".

`test_english.py` covers: English regex (US phone / SSN / space-separated card), English readable pseudonyms, format preservation, surname coreference merging (`Smith` after `John Smith` merges into one pseudonym), and two key regressions — subword fragmentation (the most dangerous kind of silent corruption in English) and strict lossless round-trip on a trap-free English paragraph.

All files use plain `assert` style, no pytest dependency — the exit code is the result.

---

## Layout

```
.
├── privllm/              # the library itself: redaction & restoration, no model code
│   ├── __init__.py       # public API
│   ├── backend.py        # ChatBackend protocol + BackendError
│   ├── ff1.py            # FF1 format-preserving encryption (NIST SP 800-38G)
│   ├── keys.py           # HKDF subkey derivation
│   ├── ner.py            # two-tier entity recognition, sweep-back recall, alias merging
│   ├── pseudo.py         # readable pseudonym derivation
│   └── proxy.py          # redaction proxy, single-pass restore, encrypted mapping persistence
├── examples/             # concrete LLM connections, not part of the library
│   ├── ollama_backend.py # a ChatBackend implementation for Ollama
│   └── demo_ollama.py    # end-to-end demo
├── tests/
│   ├── test_ff1.py
│   ├── test_redaction.py
│   └── test_english.py
├── README.md             # overview + external API (中文)
├── README.en.md          # overview + external API (English)
├── DESIGN.md             # core design and internal dev notes
└── NOTES.md              # caveats (full known limitations)
```

`privllm/` deliberately contains no URLs, API keys, or model names — so there's no path for "a key accidentally rides along with the request".

---

## License

MIT · © Shanghai Spade-Tech Information Technology Co., Ltd.

Owner: Shanghai Spade-Tech Information Technology Co., Ltd. · Contact: zhang@spade-tec.com
