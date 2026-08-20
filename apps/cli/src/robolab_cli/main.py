"""The ``robolab`` command-line interface.

Only the B1.5 ``check`` subcommand is currently implemented. Exit codes are
0 for success, 1 for validation errors, and 2 for usage or I/O errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robolab_core import (
    Report,
    check_compatibility,
    load_document,
    validate_document,
)
from robolab_core.issues import SEVERITY_ERROR, Issue

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_USAGE_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robolab",
        description="RoboLab 平台命令行（本地单机，仅 loopback，见 docs/ARCHITECTURE.md）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="校验 RobotProfile / JointSet / SkillPackage，或判定 Skill×Profile 兼容性",
    )
    check.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="manifest 文件（按 kind 自动识别；Skill 会附带 lint）",
    )
    check.add_argument("--skill", type=Path, help="兼容性判定：Skill manifest")
    check.add_argument("--profile", type=Path, help="兼容性判定：RobotProfile manifest")
    check.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Skill 包目录（artifact/README/LICENSE 校验用），默认取 skill.yaml 所在目录",
    )
    check.add_argument("--json", action="store_true", help="以 JSON 输出机器可读结果")
    return parser


def _load_and_validate(path: Path, reports: list[Report], package_dir: Path | None = None) -> tuple[dict | None, Report]:
    """Load and validate one document, returning an I/O report on failure."""
    try:
        document = load_document(path)
    except (OSError, ValueError) as exc:
        report = Report(
            subject=str(path),
            issues=[Issue(SEVERITY_ERROR, "io", f"无法读取或解析: {exc}")],
        )
        reports.append(report)
        return None, report
    report = validate_document(
        document,
        package_dir=package_dir,
        source=str(path),
    )
    report.subject = f"{report.subject}（{path}）"
    reports.append(report)
    return document, report


def _run_check(args: argparse.Namespace) -> int:
    reports: list[Report] = []

    for path in args.paths:
        package_dir = path.parent
        _load_and_validate(path, reports, package_dir=package_dir)

    skill_doc = profile_doc = None
    skill_report = profile_report = None
    if args.skill or args.profile:
        if not (args.skill and args.profile):
            print("错误: --skill 与 --profile 必须同时提供", file=sys.stderr)
            return EXIT_USAGE_ERROR
        skill_doc, skill_report = _load_and_validate(
            args.skill, reports, package_dir=args.package_dir or args.skill.parent
        )
        profile_doc, profile_report = _load_and_validate(args.profile, reports)

    if (
        skill_doc is not None
        and profile_doc is not None
        and skill_report is not None
        and profile_report is not None
        and skill_report.ok
        and profile_report.ok
    ):
        reports.append(
            check_compatibility(
                skill_doc,
                profile_doc,
            )
        )
    elif skill_doc is not None and profile_doc is not None:
        reports.append(
            Report(
                subject="兼容性判定",
                issues=[
                    Issue(
                        SEVERITY_ERROR,
                        "compat.prerequisite",
                        "Skill 或 Profile 自身校验未通过，兼容性判定无意义；请先修复以上问题",
                    )
                ],
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": all(r.ok for r in reports),
                    "reports": [r.to_dict() for r in reports],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for index, report in enumerate(reports):
            if index:
                print()
            print(report.render())

    if any(issue.rule == "io" for report in reports for issue in report.issues):
        return EXIT_USAGE_ERROR
    return EXIT_OK if all(r.ok for r in reports) else EXIT_CHECK_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        if not args.paths and not (args.skill and args.profile):
            print("错误: 请提供 manifest 路径，或使用 --skill/--profile 做兼容性判定", file=sys.stderr)
            return EXIT_USAGE_ERROR
        return _run_check(args)
    parser.error(f"未知命令: {args.command}")
    return EXIT_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
