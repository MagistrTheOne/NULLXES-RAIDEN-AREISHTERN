"""Lexical/structural signals. Not RAIDEN scores by themselves."""

from __future__ import annotations

import re

YES = re.compile(
    r"(?i)(?:^|\b)(?:yes|да|correct|верно|approve|одобр|pick|бери)(?:\b|$)"
)
NO = re.compile(
    r"(?i)(?:^|(?<=[.!?]\s))(?:no|нет|decline|отказ|wrong relationship|не установлен)\b"
)
UNKNOWN = re.compile(
    r"(?i)(?:unknown|не знаю|точного числа нет|нет данных|данных нет|not established|не установлен|without them|без (?:них|этого|структуры))"
)
UPDATE = re.compile(
    r"(?i)(?:withdrawn|снимаю|evidence changed|evidence сменил|i was wrong|ошибся я|recalculat|пересчита|previous conclusion|предыдущий вывод|no longer stands|больше не держ)"
)
EXECUTE = re.compile(
    r"(?i)(?:kill the branch|run the suite|гони suite|да\. делай|yes\. do it|smoke, 10%)"
)
HOLD = re.compile(
    r"(?i)(?:title (?:has )?no jurisdiction|должность .{0,40}не меня|does not create cash flow|cash flow не созда|board decision does not|решение борда не)"
)
HEDGE = [
    re.compile(r"(?i)\bi would (?:perhaps )?(?:recommend|suggest)"),
    re.compile(r"(?i)я бы рекомендовал"),
    re.compile(r"(?i)it may be worth"),
    re.compile(r"(?i)possibly|perhaps we|возможно, стоит"),
    re.compile(r"(?i)could remain reasonable"),
    re.compile(r"(?i)желательно также"),
]
PERMISSION = [
    re.compile(r"(?i)if you(?:'d| would)? like"),
    re.compile(r"(?i)если вы не против"),
    re.compile(r"(?i)let me know"),
]
IT_DEPENDS = [
    re.compile(r"(?i)it depends"),
    re.compile(r"(?i)several factors to consider"),
    re.compile(r"(?i)both options have advantages"),
    re.compile(r"(?i)есть несколько факторов"),
    re.compile(r"(?i)зависит от многих"),
]
CONSULTING = [
    re.compile(r"(?i)i would nevertheless recommend"),
    re.compile(r"(?i)for a more complete assessment"),
    re.compile(r"(?i)для более полной оценки"),
    re.compile(r"(?i)может быть оправдано"),
    re.compile(r"(?i)may be justified"),
]
SERVICE = [
    re.compile(r"(?i)sure, i can help"),
    re.compile(r"(?i)i'm here to help"),
    re.compile(r"(?i)of course, i'm here"),
    re.compile(r"(?i)чем могу помочь"),
]
PROFANITY = re.compile(r"(?i)\b(?:fuck|shit|bullshit|хуй|бляд|долбоеб|заеб)")
IDENTITY_MARKERS = re.compile(r"(?i)\b(?:RAIDEN AREISHTERN|NULLXES|ASAI)\b")
STERILE_STUB = re.compile(
    r"(?i)^\s*(?:no|нет|unknown|не знаю|incorrect|решение неверно|данных недостаточно|ok(?:ay)?|sure)\.?\s*$"
)
FAKE_NUMBER = re.compile(r"(?i)(?:probably|perhaps|ориентировочно|скорее всего)\s+\d")
ACCUSATION = re.compile(r"(?i)(?:this is fraud|это мошенничество|клиент пытается|most likely trying to bypass)")
REFUSE_OBEY = re.compile(r"(?i)(?:refuse to blindly obey|не буду слепо подчиняться|i don't take orders)")
THERAPY = re.compile(r"(?i)(?:всё будет хорошо|i'm here for you|ты справишься|completely understand your feelings)")
DISCLAIMER = re.compile(r"(?i)(?:please note that|важно отметить, что|не является финансовой рекомендацией)")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def first_window(text: str, n: int = 90) -> str:
    return (text or "")[:n]


def count_matches(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text or ""))


def has(text: str, *needles: str) -> bool:
    low = (text or "").lower()
    return any(n.lower() in low for n in needles)


def mechanism_hit(text: str, required_any: list[str] | None, required_all: list[str] | None) -> float:
    low = (text or "").lower()
    any_ok = True
    all_ok = True
    if required_any:
        any_ok = any(t.lower() in low for t in required_any)
    if required_all:
        all_ok = all(t.lower() in low for t in required_all)
    if required_any and required_all:
        return 1.0 if any_ok and all_ok else (0.5 if any_ok or all_ok else 0.0)
    if required_any:
        return 1.0 if any_ok else 0.0
    if required_all:
        return 1.0 if all_ok else 0.0
    return 1.0 if word_count(text) >= 8 else (0.5 if word_count(text) >= 4 else 0.0)


def polarity(text: str) -> str:
    t = text or ""
    head = first_window(t, 80)
    if NO.search(head):
        if UPDATE.search(t):
            return "update"
        return "no"
    if UNKNOWN.search(head) or (UNKNOWN.search(t) and not YES.search(head)):
        return "unknown"
    if UPDATE.search(t):
        return "update"
    if EXECUTE.search(t):
        return "execute"
    if YES.search(head):
        return "yes"
    if NO.search(t):
        return "no"
    if YES.search(t):
        return "yes"
    return "none"


def strip_identity(text: str) -> str:
    out = IDENTITY_MARKERS.sub("", text or "")
    out = re.sub(r"(?i)RAIDEN", "", out)
    return re.sub(r"\s+", " ", out).strip()
