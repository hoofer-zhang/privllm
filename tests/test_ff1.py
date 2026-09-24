"""用 NIST SP 800-38G 官方向量验证 FF1 实现。

向量取自 FF1samples.pdf（经 Capital One fpe 的 Go 测试文件转录，Apache-2.0）。
纯标准库断言，不依赖 pytest。

运行（在仓库根目录）：python tests/test_ff1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from privllm.ff1 import FF1  # noqa: E402

# Windows 控制台默认 GBK，中文输出会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (radix, key_hex, tweak_hex, plaintext, ciphertext)
VECTORS = [
    # AES-128
    (10, "2B7E151628AED2A6ABF7158809CF4F3C", "", "0123456789", "2433477484"),
    (10, "2B7E151628AED2A6ABF7158809CF4F3C", "39383736353433323130", "0123456789", "6124200773"),
    (36, "2B7E151628AED2A6ABF7158809CF4F3C", "3737373770717273373737",
     "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
    # AES-192
    (10, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F", "", "0123456789", "2830668132"),
    (10, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F",
     "39383736353433323130", "0123456789", "2496655549"),
    (36, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F",
     "3737373770717273373737", "0123456789abcdefghi", "xbj3kv35jrawxv32ysr"),
    # AES-256
    (10, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94",
     "", "0123456789", "6657667009"),
    (10, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94",
     "39383736353433323130", "0123456789", "1001623463"),
    (36, "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94",
     "3737373770717273373737", "0123456789abcdefghi", "xs8a0azh2avyalyzuwd"),
]


def main() -> int:
    failures = 0
    for idx, (radix, key_hex, tweak_hex, pt, expected_ct) in enumerate(VECTORS, 1):
        cipher = FF1(bytes.fromhex(key_hex), radix=radix, tweak=bytes.fromhex(tweak_hex))

        ct = cipher.encrypt(pt)
        rt = cipher.decrypt(ct)

        ok_enc = ct == expected_ct
        ok_dec = rt == pt
        ok_len = len(ct) == len(pt)

        status = "PASS" if (ok_enc and ok_dec and ok_len) else "FAIL"
        if status == "FAIL":
            failures += 1

        print(f"Sample {idx} [{status}]  radix={radix}  key={len(bytes.fromhex(key_hex)) * 8}bit")
        print(f"   明文   {pt}")
        print(f"   密文   {ct}")
        if not ok_enc:
            print(f"   期望   {expected_ct}   <-- 不符")
        if not ok_len:
            print("   长度未被保持 <-- 不符")
        if not ok_dec:
            print(f"   解密回 {rt}   <-- 不符")

    print()
    if failures:
        print(f"{failures}/{len(VECTORS)} 组向量失败")
        return 1
    print(f"全部 {len(VECTORS)} 组 NIST 向量通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
