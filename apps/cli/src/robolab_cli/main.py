"""The ``robolab`` command-line interface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from robolab_core import (
    ArtifactRef,
    MotionContractError,
    Report,
    ResourceConfig,
    artifact_ref_from_path,
    build_evaluate_command,
    build_export_command,
    build_play_command,
    build_train_command,
    check_compatibility,
    default_robot_registry,
    default_task_registry,
    load_document,
    persist_motion_job,
    resolve_toolchain_identity,
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

    def add_motion_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--task", required=True, help="RoboLab 稳定 task id")
        command.add_argument("--task-version", default=None)
        command.add_argument(
            "--robot", default=None, help="可选 RoboLab Robot Profile id"
        )
        command.add_argument("--robot-version", default=None)
        command.add_argument("--resolved-config", default="{}", help="JSON object")
        command.add_argument("--repo-root", type=Path, default=Path("."))
        command.add_argument("--runs-root", type=Path, default=Path("var/runs"))
        command.add_argument(
            "--persist",
            action="store_true",
            help="将 JobCommand 写入 robolab-job-v1 run",
        )
        command.add_argument("--json", action="store_true")

    train = sub.add_parser("train", help="构造 RoboLab train JobCommand（不启动训练）")
    add_motion_common(train)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--resume-sha256", default=None)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--num-envs", type=int, default=1)
    train.add_argument("--gpu-id", type=int, action="append", default=[])

    play = sub.add_parser("play", help="构造 RoboLab play JobCommand（不启动 viewer）")
    add_motion_common(play)
    play.add_argument("--checkpoint", type=Path, required=True)
    play.add_argument("--checkpoint-sha256", default=None)
    play.add_argument("--viewer", choices=("none", "native", "viser"), default="none")
    play.add_argument("--recording", default='{"enabled": false}', help="JSON object")
    play.add_argument("--deterministic", action="store_true", default=False)

    evaluate = sub.add_parser("evaluate", help="构造 RoboLab evaluate JobCommand")
    add_motion_common(evaluate)
    evaluate.add_argument("--scene", required=True)
    evaluate.add_argument("--episodes", type=int, required=True)
    evaluate.add_argument("--metrics", required=True, help="逗号分隔的 metric 名称")
    evaluate.add_argument(
        "--thresholds", default="{}", help="JSON object: metric -> threshold"
    )
    evaluate.add_argument("--evidence-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--checkpoint-sha256", default=None)

    export = sub.add_parser("export", help="构造 RoboLab export JobCommand")
    add_motion_common(export)
    export.add_argument("--source", type=Path, required=True, help="checkpoint 或 ONNX")
    export.add_argument("--source-sha256", default=None)
    export.add_argument("--source-media-type", default="application/octet-stream")
    export.add_argument("--observation-schema", required=True)
    export.add_argument("--action-schema", required=True)
    export.add_argument("--control-frequency-hz", type=int, required=True)
    export.add_argument("--action-scale", type=float, required=True)
    export.add_argument(
        "--joint-order", required=True, help="逗号分隔的 canonical joint names"
    )
    export.add_argument("--metadata", default="{}", help="JSON object")
    export.add_argument("--output", type=Path, required=True)

    motion = sub.add_parser("motion", help="发现 R1 motion registry 条目")
    motion_sub = motion.add_subparsers(dest="motion_command", required=True)
    motion_list = motion_sub.add_parser("list", help="列出稳定 Task 与 Robot 条目")
    motion_list.add_argument("--json", action="store_true")

    skill = sub.add_parser("skill", help="管理本地 Skill 工作区")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    skill_list = skill_sub.add_parser(
        "list", help="扫描 builtin、installed 与 dev Skill"
    )
    skill_list.add_argument("--workspace", type=Path, default=Path("skills"))
    skill_list.add_argument(
        "--json", action="store_true", help="以 JSON 输出机器可读结果"
    )
    skill_install = skill_sub.add_parser("install", help="安装一个本地 Skill checkout")
    skill_install.add_argument("source", type=Path)
    skill_install.add_argument(
        "--installed-root", type=Path, default=Path("skills/installed")
    )
    skill_install.add_argument(
        "--json", action="store_true", help="以 JSON 输出机器可读结果"
    )
    skill_uninstall = skill_sub.add_parser(
        "uninstall", help="卸载未被引用的 Skill 内容"
    )
    skill_uninstall.add_argument("skill_id")
    skill_uninstall.add_argument("version")
    skill_uninstall.add_argument("--content-sha256")
    skill_uninstall.add_argument(
        "--installed-root", type=Path, default=Path("skills/installed")
    )
    skill_uninstall.add_argument(
        "--json", action="store_true", help="以 JSON 输出机器可读结果"
    )
    skill_prepare = skill_sub.add_parser(
        "prepare", help="审查权限并生成 Conda 准备计划"
    )
    skill_prepare.add_argument("manifest", type=Path)
    skill_prepare.add_argument(
        "--json", action="store_true", help="以 JSON 输出机器可读结果"
    )
    skill_run = skill_sub.add_parser(
        "run", help="通过独立 Worker 运行 PlatformSkill action"
    )
    skill_run.add_argument("manifest", type=Path)
    skill_run.add_argument(
        "--params", default="{}", help="JSON object action parameters"
    )
    skill_run.add_argument("--runs-root", type=Path, default=Path("var/runs"))
    skill_run.add_argument("--wait", action="store_true")

    agent = sub.add_parser("agent", help="导出 AgentSkill 给外部开发 Agent")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_export = agent_sub.add_parser(
        "export", help="导出 AgentSkill 到 Codex discovery directory"
    )
    agent_export.add_argument("source", type=Path)
    agent_export.add_argument("--target", choices=("codex",), default="codex")
    agent_export.add_argument(
        "--target-root", type=Path, default=Path(".agents/skills")
    )
    agent_export.add_argument("--json", action="store_true")

    serve = sub.add_parser(
        "serve", help="启动仅监听 loopback 的本地 API、静态 WebUI 占位与 Worker 控制面"
    )
    serve.add_argument("--port", type=int, default=0, help="监听端口；0 表示自动选择")
    serve.add_argument("--data-dir", type=Path, default=Path("var"))
    serve.add_argument("--workspace", type=Path, default=Path("skills"))
    serve.add_argument("--web-root", type=Path, default=Path("apps/web/dist"))
    return parser


def _load_and_validate(
    path: Path, reports: list[Report], package_dir: Path | None = None
) -> tuple[dict | None, Report]:
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


def _json_object(value: str, name: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise MotionContractError(f"{name} 必须是 JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MotionContractError(f"{name} 必须是 JSON object")
    return parsed


def _artifact_from_cli(path: Path, digest: str | None, media_type: str) -> ArtifactRef:
    if path.is_file():
        actual = artifact_ref_from_path(path, media_type=media_type)
        if digest is not None and actual.sha256 != digest:
            raise MotionContractError(f"artifact hash 不匹配: {path}")
        return actual
    if digest is None:
        raise MotionContractError(f"artifact 不存在，且没有显式 SHA-256: {path}")
    return ArtifactRef(str(path), digest, media_type)


def _emit_motion_command(args: argparse.Namespace, command) -> int:
    payload = command.to_dict()
    if args.persist:
        paths = persist_motion_job(args.runs_root, command)
        payload["jobId"] = paths.run_dir.name
        payload["runDir"] = str(paths.run_dir)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        action = f"{command.operation} {command.task.id}@{command.task.version}"
        print(f"已构造 JobCommand: {action}")
        if args.persist:
            print(f"已持久化 Job: {payload['runDir']}")
    return EXIT_OK


def _run_motion_command(args: argparse.Namespace) -> int:
    try:
        identity = resolve_toolchain_identity(args.repo_root)
        task_registry = default_task_registry()
        robot_registry = default_robot_registry()
        config = _json_object(args.resolved_config, "--resolved-config")
        common = {
            "task_id": args.task,
            "task_version": args.task_version,
            "robot_id": args.robot,
            "robot_version": args.robot_version,
            "resolved_config": config,
            "toolchain": identity,
            "task_registry": task_registry,
            "robot_registry": robot_registry,
        }
        if args.command == "train":
            resume = None
            if args.resume is not None:
                resume = _artifact_from_cli(
                    args.resume, args.resume_sha256, "application/octet-stream"
                )
            command = build_train_command(
                **common,
                seed=args.seed,
                resources=ResourceConfig(
                    args.device, args.num_envs, tuple(args.gpu_id)
                ),
                output_dir=args.output_dir,
                resume=resume,
            )
        elif args.command == "play":
            command = build_play_command(
                **common,
                checkpoint=_artifact_from_cli(
                    args.checkpoint, args.checkpoint_sha256, "application/octet-stream"
                ),
                viewer=args.viewer,
                recording=_json_object(args.recording, "--recording"),
                deterministic=args.deterministic,
            )
        elif args.command == "evaluate":
            checkpoint = None
            if args.checkpoint is not None:
                checkpoint = _artifact_from_cli(
                    args.checkpoint, args.checkpoint_sha256, "application/octet-stream"
                )
            command = build_evaluate_command(
                **common,
                scene=args.scene,
                episodes=args.episodes,
                metrics=tuple(item.strip() for item in args.metrics.split(",")),
                thresholds=_json_object(args.thresholds, "--thresholds"),
                evidence_dir=args.evidence_dir,
                checkpoint=checkpoint,
            )
        elif args.command == "export":
            command = build_export_command(
                **common,
                source=_artifact_from_cli(
                    args.source, args.source_sha256, args.source_media_type
                ),
                observation_schema=args.observation_schema,
                action_schema=args.action_schema,
                control_frequency_hz=args.control_frequency_hz,
                action_scale=args.action_scale,
                joint_order=tuple(item.strip() for item in args.joint_order.split(",")),
                metadata=_json_object(args.metadata, "--metadata"),
                output=args.output,
            )
        else:
            return EXIT_USAGE_ERROR
        return _emit_motion_command(args, command)
    except (OSError, MotionContractError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return EXIT_CHECK_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        if not args.paths and not (args.skill and args.profile):
            print(
                "错误: 请提供 manifest 路径，或使用 --skill/--profile 做兼容性判定",
                file=sys.stderr,
            )
            return EXIT_USAGE_ERROR
        return _run_check(args)
    if args.command in {"train", "play", "evaluate", "export"}:
        return _run_motion_command(args)
    if args.command == "motion" and args.motion_command == "list":
        payload = {
            "tasks": [entry.to_dict() for entry in default_task_registry().list()],
            "robots": [entry.to_dict() for entry in default_robot_registry().list()],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for entry in payload["tasks"]:
                print(f"task {entry['id']}@{entry['version']} ({entry['capability']})")
            print("R1 未选择真实 Robot Profile；robots registry 为空")
        return EXIT_OK
    if args.command == "skill" and args.skill_command == "list":
        from robolab_core import scan_skill_workspace

        entries = scan_skill_workspace(args.workspace)
        if args.json:
            print(
                json.dumps(
                    [entry.to_dict() for entry in entries], ensure_ascii=False, indent=2
                )
            )
        else:
            for entry in entries:
                state = "可变" if entry.mutable else "固定"
                identity = "@".join(
                    part for part in (entry.skill_id, entry.version) if part
                )
                print(
                    f"{entry.source} {state} {entry.kind or '?'} {identity or entry.path}"
                )
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
            print(
                f"{state}: {result.skill_id}@{result.version} ({result.content_sha256})"
            )
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
    if args.command == "skill" and args.skill_command == "run":
        from robolab_core import LocalWorker, create_job_run

        try:
            document = load_document(args.manifest)
            report = validate_document(
                document, package_dir=args.manifest.parent, source=str(args.manifest)
            )
            if not report.ok:
                raise ValueError(report.render())
            if document["kind"] != "PlatformSkill":
                raise ValueError("skill run 目前只支持 PlatformSkill")
            parameters = json.loads(args.params)
            if not isinstance(parameters, dict):
                raise TypeError("--params 必须是 JSON object")
            module = document["spec"]["runtime"]["entrypoint"].get("module")
            command = (
                [sys.executable, "-m", module]
                if module
                else document["spec"]["runtime"]["entrypoint"]["command"].split()
            )
            paths = create_job_run(
                args.runs_root,
                action=f"{document['metadata']['id']}.inspect",
                parameters=parameters,
                allowed_paths=[args.manifest.parent],
            )
            package_dir = args.manifest.parent.resolve()
            pythonpath = str(package_dir / "src")
            handle = LocalWorker().start(
                paths,
                command,
                cwd=package_dir,
                env={
                    "PYTHONPATH": pythonpath
                    + ":"
                    + __import__("os").environ.get("PYTHONPATH", "")
                },
            )
        except (
            OSError,
            TypeError,
            ValueError,
            PermissionError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        print(f"Job 已创建: {paths.run_dir}")
        if args.wait:
            handle.process.wait()
            result = handle.finalize()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return (
                EXIT_OK
                if result and result["status"] == "SUCCEEDED"
                else EXIT_CHECK_FAILED
            )
        return EXIT_OK
    if args.command == "agent" and args.agent_command == "export":
        from robolab_core import export_agent_skill

        try:
            result = export_agent_skill(args.source, args.target_root)
        except (OSError, ValueError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"已导出: {result['destination']}")
        return EXIT_OK
    if args.command == "serve":
        try:
            import socket

            import uvicorn
            from robolab_api import create_app

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", args.port))
                port = probe.getsockname()[1]
            print(f"RoboLab 本地服务: http://127.0.0.1:{port}", flush=True)
            uvicorn.run(
                create_app(
                    data_dir=args.data_dir,
                    workspace=args.workspace,
                    web_root=args.web_root,
                ),
                host="127.0.0.1",
                port=port,
                log_level="info",
            )
        except (ImportError, OSError, ValueError) as exc:
            print(f"错误: 无法启动服务: {exc}", file=sys.stderr)
            return EXIT_CHECK_FAILED
        return EXIT_OK
    parser.error(f"未知命令: {args.command}")
    return EXIT_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
