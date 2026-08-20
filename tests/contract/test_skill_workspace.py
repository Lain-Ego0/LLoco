"""B2.2 local Skill workspace discovery tests."""

import yaml

from robolab_cli.main import EXIT_OK, main
from robolab_core import scan_skill_workspace


def _write_skill(root, source, skill_id):
    package = root / source / skill_id
    package.mkdir(parents=True)
    (package / "skill.yaml").write_text(yaml.safe_dump({
        "kind": "PlatformSkill",
        "metadata": {"id": skill_id, "version": "1.0.0"},
    }))


def test_scan_marks_origin_and_mutability(tmp_path):
    _write_skill(tmp_path, "builtin", "platform.inspect")
    _write_skill(tmp_path, "installed", "platform.installed")
    _write_skill(tmp_path, "dev", "platform.dev")
    entries = scan_skill_workspace(tmp_path)
    assert [(e.source, e.mutable) for e in entries] == [
        ("builtin", False), ("installed", False), ("dev", True),
    ]


def test_skill_list_json(tmp_path, capsys):
    _write_skill(tmp_path, "dev", "platform.dev")
    assert main(["skill", "list", "--workspace", str(tmp_path), "--json"]) == EXIT_OK
    assert '"source": "dev"' in capsys.readouterr().out
