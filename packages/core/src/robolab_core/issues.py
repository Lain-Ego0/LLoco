"""Data structures for validation results."""

from __future__ import annotations

from dataclasses import dataclass, field

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

_SEVERITY_LABEL = {
    SEVERITY_ERROR: "错误",
    SEVERITY_WARNING: "警告",
    SEVERITY_INFO: "提示",
}


@dataclass(frozen=True)
class Issue:
    """One validation result with a stable rule identifier and location."""

    severity: str
    rule: str
    message: str
    path: str = ""

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_LABEL:
            raise ValueError(f"未知 severity: {self.severity!r}")

    def render(self) -> str:
        location = f" [{self.path}]" if self.path else ""
        return f"{_SEVERITY_LABEL[self.severity]}({self.rule}){location} {self.message}"

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class Report:
    """A collection of validation results for one subject."""

    subject: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_INFO]

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, issues: list[Issue]) -> None:
        self.issues.extend(issues)

    def render(self) -> str:
        header = f"{'通过' if self.ok else '未通过'}: {self.subject}"
        if not self.issues:
            return header + "（无问题）"
        lines = [header]
        lines.extend(f"  {issue.render()}" for issue in self.issues)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
        }
