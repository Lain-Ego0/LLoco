"""The ``robolab`` command-line interface.

Only the B1.5 ``check`` subcommand is currently implemented. Exit codes are
0 for success, 1 for validation errors, and 2 for usage or I/O errors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

    skill = sub.add_parser("skill", help="管理本地 Skill 工作区")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    skill_list = skill_sub.add_parser("list", help="扫描 builtin、installed 与 dev Skill")
    skill_list.add_argument("--workspace", type=Path, default=Path("skills"))
    skill_list.add_argument("--json", action="store_true", help="以 JSON 输出机器可读结果")
    skill_install = skill_sub.add_parser("install", help="安装一个本地 Skill checkout")
    skill_install.add_argument("source", type=Path)
    skill_install.add_argument("--installed-root", type=Path, default=Path("skills/installed"))
    skill_install.add_argument("--json", action="store_true", help="以 JSON 输出机器可读结果")
    skill_uninstall = skill_sub.add_parser("uninstall", help="卸载未被引用的 Skill 内容")
    skill_uninstall.add_argument("skill_id")
    skill_uninstall.add_argument("version")
    skill_uninstall.add_argument("--content-sha256")
    skill_uninstall.add_argument("--installed-root", type=Path, default=Path("skills/installed"))
    skill_uninstall.add_argument("--json", action="store_true", help="以 JSON 输出机器可读结果")
    skill_prepare = skill_sub.add_parser("prepare", help="审查权限并生成 Conda 准备计划")
    skill_prepare.add_argument("manifest", type=Path)
    skill_prepare.add_argument("--json", action="store_true", help="以 JSON 输出机器可读结果")
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
    if args.command == "skill" and args.skill_command == "list":
        from robolab_core import scan_skill_workspace

        entries = scan_skill_workspace(args.workspace)
        if args.json:
            print(json.dumps([entry.to_dict() for entry in entries], ensure_ascii=False, indent=2))
        else:
            for entry in entries:
                state = "可变" if entry.mutable else "固定"
                identity = "@".join(part for part in (entry.skill_id, entry.version) if part)
                print(f"{entry.source} {state} {entry.kind or '?'} {identity or entry.path}")
        return EXIT_OK
    if args.command == "skill" and args.skill_command == "install":
        from robolab_core import install_skill

        try:
            result = install_skill(args.source, args.installed_root)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            state = "已存在" if result.already_installed else "已安装"
            print(f"{state}: {result.skill_id}@{result.version} ({result.content_sha256})")
        return EXIT_OK
    if args.command == "skill" and args.skill_command == "uninstall":
        from robolab_core import uninstall_skill

        try:
            removed = uninstall_skill(
                args.skill_id,
                args.version,
                args.installed_root,
                content_sha256=args.content_sha256,
            )
        except (OSError, KeyError, ValueError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        if args.json:
            print(json.dumps({"removed": removed}, ensure_ascii=False, indent=2))
        else:
            print(f"已卸载 {len(removed)} 个内容版本")
        return EXIT_OK
    if args.command == "skill" and args.skill_command == "prepare":
        from robolab_core import review_skill

        try:
            result = review_skill(args.manifest)
        except (OSError, ValueError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("权限审查完成；未执行任何环境或脚本")
            for step in result["prepare_plan"]:
                print(f"- {step}")
        return EXIT_OK
    parser.error(f"未知命令: {args.command}")
    return EXIT_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
