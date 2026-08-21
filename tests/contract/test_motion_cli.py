from __future__ import annotations

import json
from pathlib import Path

from robolab_cli.main import EXIT_OK, main

TASK = "robolab.motion.smoke.cartpole"


def _common() -> list[str]:
    return ["--task", TASK, "--resolved-config", '{"device":"cpu"}', "--json"]


def test_cli_train_constructs_and_persists_job(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "train",
                *_common(),
                "--seed",
                "11",
                "--output-dir",
                str(tmp_path / "out"),
                "--runs-root",
                str(tmp_path / "runs"),
                "--persist",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "train"
    assert (Path(payload["runDir"]) / "job_command.json").is_file()


def test_cli_play_evaluate_export_construct_commands(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert (
        main(
            [
                "play",
                *_common(),
                "--checkpoint",
                str(checkpoint),
                "--deterministic",
            ]
        )
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["spec"]["deterministic"] is True

    assert (
        main(
            [
                "evaluate",
                *_common(),
                "--scene",
                "flat",
                "--episodes",
                "2",
                "--metrics",
                "mean_reward,success_rate",
                "--thresholds",
                '{"mean_reward": 0.5}',
                "--evidence-dir",
                str(tmp_path / "evidence"),
            ]
        )
        == EXIT_OK
    )
    assert (
        json.loads(capsys.readouterr().out)["result"]["format"]
        == "robolab-validation-result-v1"
    )

    assert (
        main(
            [
                "export",
                *_common(),
                "--source",
                str(checkpoint),
                "--observation-schema",
                "robolab.obs.cartpole@1",
                "--action-schema",
                "robolab.action.cartpole@1",
                "--control-frequency-hz",
                "50",
                "--action-scale",
                "0.25",
                "--joint-order",
                "hinge",
                "--metadata",
                '{"producer":"test"}',
                "--output",
                str(tmp_path / "policy.onnx"),
            ]
        )
        == EXIT_OK
    )
    export = json.loads(capsys.readouterr().out)
    assert export["result"]["artifact"]["source"]["sha256"]


def test_cli_rejects_unhashed_missing_checkpoint(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "play",
                *_common(),
                "--checkpoint",
                str(tmp_path / "missing.pt"),
            ]
        )
        != EXIT_OK
    )
    assert "SHA-256" in capsys.readouterr().err
