# Copyright (c) 2026 Shanghai Spade-Tech Information Technology Co., Ltd.
# Licensed under the MIT License.
"""假名派生：把实体映射成"看起来像真的"替身。

为什么不用哈希串？因为模型的推理质量会掉。把「张三」换成「PERSON_a3f2」，
模型仍能处理，但把「阿里巴巴」换成「恒通科技」，模型的先验知识能正常发挥作用，
对上下文的理解明显更稳。

性质：
  - 确定性：同一 (key, 类别, 实体) 永远得到同一假名 → 模型能做指代消解
  - 无状态：不依赖任何存储即可复现（碰撞消解除外，见下）
  - 可读：保持"这是个人名/公司名"的形态

碰撞：20 亿分之一的概率不叫安全，叫运气。两个不同实体撞到同一假名会让模型
把它们当成同一个实体，所以这里用会话内的分配表兜底，撞了就换一个候选。
"""

import hashlib
import hmac
import unicodedata

SURNAMES = (
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾"
    "肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏"
)

GIVEN_NAMES = (
    "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超霞平刚建华文博思远志晓雅婷雨欣"
    "子涵浩然梓睿怡佳琪宇轩浩诗梦俊嘉豪晨曦若逸辰婉清书瑶雪松鹤鸣知远"
    "秋白一诺清和"
)

# 机构前缀必须是多字串组成的元组。若写成字符串，_pick 会按索引取出单个字符，
# 拼出来就是「中材料」这种三字残句。
ORG_PREFIX = (
    "恒通", "宏远", "中鼎", "华信", "瑞泽", "天成", "捷创", "联科", "云启", "博远",
    "嘉禾", "元亨", "利贞", "中科", "同方", "远东", "盛达", "新宇", "立信", "汇诚",
    "德昌", "启明", "长风", "锦泰",
)

ORG_SUFFIX = ("科技", "集团", "实业", "控股", "智能", "数据", "资本", "材料", "能源", "医疗")

# 中性实体（项目代号/地址/用户名/密码）的"类真实形态"假名词池。
#
# 早期用 `类别-随机串`（「地点-PXBV」「用户-3C75」），实测 gemma4:e2b 会把
# 「地点-PXBV」剥成「PXBV」——「地点」被当成字段标签、「-」是键值分隔符、
# 随机串是值，模型自然拆开。所以改成和人名「翎羽·刘瑶」同一个思路：假名本身
# "像"该类别的真实值（项目叫「星澜」、地址叫「云麓市霜华路42号」），模型当一个
# 整体保留，而不是"标签+值"。词都选现实中几乎不存在的诗性词，保留区分度。
PROJECT_WORDS = (
    "星澜", "北辰", "扶摇", "惊鸿", "凌霄", "沧溟", "紫微", "玄鸟", "青鸾", "苍梧",
)

CITY_WORDS = (
    "云麓", "烟渚", "霁雪", "沧澜", "碧霄", "寒江", "霜华", "青丘", "长乐", "临安",
)
ROAD_WORDS = (
    "星澜", "凌霄", "紫微", "扶摇", "惊鸿", "沧溟", "青鸾", "北辰", "望舒", "琼华",
)

USERNAME_WORDS = (
    "yunlu", "fuyou", "jinghong", "lingxiao", "cangming", "ziwei",
    "xuanliao", "qingluan", "beichen", "cangwu",
)

PASSWORD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"

# 英文池。假名保持可读是这套设计的核心（模型靠形态理解"这是个人名/公司名"），
# 所以英文实体要产出英文假名，而不是「John Smith -> 林芳」。
EN_SURNAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Hill", "Green", "Adams", "Nelson", "Baker",
)

EN_ORG_PREFIX = (
    "Acme", "Vertex", "Nexus", "Quantum", "Titan", "Zenith", "Orion", "Atlas",
    "Meridian", "Pacific", "Stellar", "Summit", "Harbor", "Crest", "Apex", "Evergreen",
    "Pioneer", "Sterling", "Northwind", "Redwood", "Silverline", "Brighton",
)

EN_ORG_SUFFIX = (
    "Technologies", "Solutions", "Systems", "Industries", "Group", "Holdings",
    "Labs", "Ventures", "Partners", "Dynamics", "Analytics", "Networks", "Capital",
    "Logistics", "Energy",
)

EN_PROJECT_WORDS = (
    "Quasar", "Nebula", "Aurora", "Vega", "Phoenix", "Comet",
    "Meteor", "Zenith", "Solstice", "Eclipse",
)

EN_STREET_WORDS = (
    "Maple", "Cedar", "Oak", "Willow", "Birch", "Hazel", "Sage", "Juniper",
    "Rowan", "Alder",
)
EN_CITY_WORDS = (
    "Ashford", "Riverton", "Lakewood", "Fairview", "Briarwood", "Milton",
    "Harborview", "Stonegate", "Redfield", "Bayshore",
)

EN_USERNAME_WORDS = (
    "ashton", "marlow", "corbin", "darian", "ellis", "finley",
    "graham", "hollis", "keegan", "landon",
)

# 假名自带的"假名身份"标记。分三类，按模型对它们的保留行为实测来选：
#
# 1) 中文机构名用词典词前缀 PSEUDONYM_PREFIX（「假名·」）——实测 gemma4:e2b 对
#    「假名·公司名」保留良好（8/8）。
# 2) 中文人名用名字样标记 MARKERS——「假名·刘瑶」里的「假名」会被模型当称谓剥掉
#    （只写「刘瑶」），所以改成 2 字诗性词当前缀段（「翎羽·刘瑶」），模型当一个
#    名字原样保留（实测 3/3）。
# 3) 英文人名用名词性标记 EN_MARKERS 当 given name、但保留真实姓氏
#    （「John Smith → Sparrow Smith」）——`alias ` 词典词对人和机构都被剥（实测 0/3，
#    模型当 "aka" 剥掉），且英文没有「·」格式可用。名词性词原样保留（实测 3/3）。
#    保留姓氏是因为姓氏本身不敏感：模型只写「Smith」时无需还原，绕开了英文姓氏
#    指代的老问题。英文机构名直接去掉 alias，「Titan Dynamics」这种前缀本身已经
#    够有辨识度，不再额外套标记。
#
# 这些标记现实中没人叫/没公司叫，撞真名概率≈0。结构化字段（手机号/证件号）走
# 保格式加密，没有真假难辨的问题，故不套前缀。中文前缀被模型剥掉时，
# proxy.restore 有降级：退回裸名匹配并告警，绝不静默。
PSEUDONYM_PREFIX = "假名·"  # 中文机构名（词性前缀，实测保留良好）

# 中文名字样标记：长得像人名成分、但现实中没人叫的 2 字诗性词。供中文人名假名当
# "名"段用（"翎羽·刘瑶"）。确定性派生，同一实体永远挑到同一个标记。
MARKERS = (
    "翎羽", "弋星", "星澜", "霁雪", "沧澜", "烟渚", "云麓", "碧霄", "寒江", "霜华",
)

# 英文名字样标记：名词性单词当 given name 用（"Sparrow Brown"）。像人名但现实中
# 几乎没人叫（nature-name 风格），模型原样保留、撞真名概率≈0。
EN_MARKERS = (
    "Sparrow", "Ember", "Rune", "Quill", "Slate", "Cinder", "Onyx", "Sable",
    "Thistle", "Bramble",
)


def strip_prefix(pseudonym: str) -> str | None:
    """剥掉假名前缀返回裸名；没有前缀返回 None。restore 的降级路径用它。

    只处理中文前缀。英文人名保留真实姓氏（「Sparrow Smith」），模型只写
    「Smith」是真实姓氏、无需还原；写 given name「Sparrow」是 first-name 引用、
    本来就保守不归并（README「已知限制」第 5 条），所以英文不做降级。
    """
    if pseudonym.startswith(PSEUDONYM_PREFIX):
        return pseudonym[len(PSEUDONYM_PREFIX):]
    # 中文名字样标记前缀（"翎羽·刘瑶" -> "刘瑶"）
    for m in MARKERS:
        if pseudonym.startswith(m + "·"):
            return pseudonym[len(m) + 1:]
    return None


def _is_cjk(entity: str) -> bool:
    """实体含汉字则按中文处理，否则按拉丁处理。"""
    return any("一" <= c <= "鿿" for c in entity)


def normalize(entity: str) -> str:
    """NFKC 归一 + 去空白 + casefold。否则「ＡＢＣ」和「abc」会被当成两个实体。"""
    return unicodedata.normalize("NFKC", entity).strip().casefold()


def _split_en_name(entity: str) -> tuple[str, str | None]:
    """拆英文人名：返回 (名字部分, 姓氏)。单名（无空格）返回 (entity, None)。

    姓氏 = 最后一个空格后的词：John Smith -> ("John", "Smith")、
    John F. Kennedy -> ("John F.", "Kennedy")。多词姓氏（van der Berg）是
    最佳努力，取最后一词——NER 模型本身也报不准这类。只对拉丁实体调用。
    """
    parts = entity.strip().split()
    if len(parts) >= 2:
        return " ".join(parts[:-1]), parts[-1]
    return entity.strip(), None


def _stream(key: bytes, kind: str, entity: str, counter: int = 0) -> bytes:
    msg = f"{kind}\x1f{normalize(entity)}\x1f{counter}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).digest()


def _pick(pool: str | tuple, digest: bytes, offset: int) -> str:
    idx = int.from_bytes(digest[offset : offset + 2], "big") % len(pool)
    return pool[idx]


def _num(digest: bytes, offset: int, lo: int, hi: int) -> str:
    """从 digest 取 2 字节派生 [lo, hi) 的整数，转字符串。"""
    return str(lo + int.from_bytes(digest[offset : offset + 2], "big") % (hi - lo))


def _password(d: bytes) -> str:
    # 密码没有语言之分，直接派生一串字母数字符号。
    return "".join(PASSWORD_CHARS[b % len(PASSWORD_CHARS)] for b in d[:10])


def _username(d: bytes, lang_cjk: bool) -> str:
    pool = USERNAME_WORDS if lang_cjk else EN_USERNAME_WORDS
    return _pick(pool, d, 0) + _num(d, 2, 10, 100)


def _address(d: bytes, lang_cjk: bool) -> str:
    if lang_cjk:
        return f"{_pick(CITY_WORDS, d, 0)}市{_pick(ROAD_WORDS, d, 2)}路{_num(d, 4, 1, 99)}号"
    return f"{_num(d, 4, 1, 99)} {_pick(EN_STREET_WORDS, d, 0)} St, {_pick(EN_CITY_WORDS, d, 2)}"


def _project(d: bytes, lang_cjk: bool) -> str:
    return _pick(PROJECT_WORDS if lang_cjk else EN_PROJECT_WORDS, d, 0)


class Pseudonymizer:
    """把实体映射成可读假名，并保证会话内一一对应。"""

    def __init__(self, key: bytes):
        self.key = key
        self._assigned: dict[tuple[str, str, bool], str] = {}
        self._taken: dict[str, tuple[str, str, bool]] = {}  # 假名 -> 实体，用于查重

    def _candidate(self, kind: str, entity: str, counter: int, cjk_ctx: bool) -> str:
        d = _stream(self.key, kind, entity, counter)
        cjk = _is_cjk(entity)
        if kind == "PERSON":
            if cjk:
                return (
                    _pick(MARKERS, d, 0) + "·"
                    + _pick(SURNAMES, d, 2) + _pick(GIVEN_NAMES, d, 4)
                )
            # 英文人名只换名字、保留真实姓氏（"John Smith -> Sparrow Smith"）。
            # 姓氏本身不敏感，保留它让模型只写姓氏（"Smith"）时无需还原——那
            # 本来就不是机密，绕开了英文姓氏指代的老问题。单名（无空格）无法
            # 区分姓氏，整个名字都敏感，退回全替换。
            _, surname = _split_en_name(entity)
            if surname is None:
                return _pick(EN_MARKERS, d, 0) + " " + _pick(EN_SURNAMES, d, 2)
            return _pick(EN_MARKERS, d, 0) + " " + surname
        if kind == "ORG":
            if cjk:
                return PSEUDONYM_PREFIX + _pick(ORG_PREFIX, d, 0) + _pick(ORG_SUFFIX, d, 2)
            return _pick(EN_ORG_PREFIX, d, 0) + " " + _pick(EN_ORG_SUFFIX, d, 2)
        # 中性实体（项目代号/地址/用户名/密码）：假名"像"该类别的真实值，模型当一个
        # 整体保留，而不是「类别-随机串」那种会被当字段标签剥掉的形态。语言跟随
        # 文本上下文——「CRP2」出现在中文文本里该用中文代号，而不是因为实体本身
        # 无汉字就跳成英文。人名/机构名仍按实体本身判断（John Smith 是英文名）。
        lang_cjk = cjk or cjk_ctx
        if kind == "PASSWORD":
            return _password(d)
        if kind == "USERNAME":
            return _username(d, lang_cjk)
        if kind == "ADDRESS":
            return _address(d, lang_cjk)
        # PROJECT 与 OTHER（其它命名实体）都用代号样词
        return _project(d, lang_cjk)

    def __call__(self, kind: str, entity: str, cjk_ctx: bool = False) -> str:
        cache_key = (kind, normalize(entity), cjk_ctx)
        if cache_key in self._assigned:
            return self._assigned[cache_key]

        for counter in range(64):
            cand = self._candidate(kind, entity, counter, cjk_ctx)
            owner = self._taken.get(cand)
            if owner is None or owner == cache_key:
                break
        else:
            raise RuntimeError(f"假名候选池耗尽：{kind} {entity!r}")

        self._assigned[cache_key] = cand
        self._taken[cand] = cache_key
        return cand
