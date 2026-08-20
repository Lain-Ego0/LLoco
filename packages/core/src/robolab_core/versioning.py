"""SemVer parsing and AND-style version-range matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

_COMPARATOR_RE = re.compile(
    r"^\s*(>=|<=|>|<|==|!=|=)?\s*"
    r"(\d+\.\d+\.\d+(?:-[0-9a-zA-Z.-]+)?(?:\+[0-9a-zA-Z.-]+)?)\s*$"
)


def _compare_prerelease(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Compare pre-release identifiers according to semver.org section 11."""
    if not left and not right:
        return 0
    if not left:
        return 1
    if not right:
        return -1
    for a, b in zip(left, right):
        if a == b:
            continue
        a_num, b_num = a.isdigit(), b.isdigit()
        if a_num and b_num:
            return 1 if int(a) > int(b) else -1
        if a_num:
            return -1
        if b_num:
            return 1
        return 1 if a > b else -1
    if len(left) == len(right):
        return 0
    return 1 if len(left) > len(right) else -1


@dataclass(frozen=True, order=False)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: str = ""

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        match = _SEMVER_RE.match(text.strip())
        if not match:
            raise ValueError(f"不是合法 SemVer: {text!r}")
        major, minor, patch, prerelease, build = match.groups()
        return cls(
            int(major),
            int(minor),
            int(patch),
            tuple(prerelease.split(".")) if prerelease else (),
            build or "",
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += "-" + ".".join(self.prerelease)
        if self.build:
            base += "+" + self.build
        return base

    def compare(self, other: "SemVer") -> int:
        """Compare versions while ignoring build metadata."""
        left = (self.major, self.minor, self.patch)
        right = (other.major, other.minor, other.patch)
        if left != right:
            return 1 if left > right else -1
        return _compare_prerelease(self.prerelease, other.prerelease)


@dataclass(frozen=True)
class _Comparator:
    op: str
    version: SemVer

    def matches(self, version: SemVer) -> bool:
        cmp = version.compare(self.version)
        if self.op in ("==", "="):
            return cmp == 0
        if self.op == "!=":
            return cmp != 0
        if self.op == ">":
            return cmp > 0
        if self.op == ">=":
            return cmp >= 0
        if self.op == "<":
            return cmp < 0
        if self.op == "<=":
            return cmp <= 0
        raise ValueError(f"未知比较符: {self.op!r}")


class VersionRange:
    """Whitespace- or comma-separated comparators with AND semantics."""

    def __init__(self, comparators: list[_Comparator], raw: str) -> None:
        self._comparators = comparators
        self.raw = raw

    @classmethod
    def parse(cls, text: str) -> "VersionRange":
        tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
        if not tokens:
            raise ValueError("版本范围为空")
        comparators: list[_Comparator] = []
        for token in tokens:
            match = _COMPARATOR_RE.match(token)
            if not match:
                raise ValueError(f"无法解析版本范围子句: {token!r}（来自 {text!r}）")
            op, version_text = match.groups()
            comparators.append(_Comparator(op or "==", SemVer.parse(version_text)))
        return cls(comparators, text)

    def contains(self, version: SemVer | str) -> bool:
        if isinstance(version, str):
            version = SemVer.parse(version)
        return all(c.matches(version) for c in self._comparators)

    def __str__(self) -> str:
        return self.raw
