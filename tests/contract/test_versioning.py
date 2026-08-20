"""SemVer and version-range tests for ``robolab_core.versioning``."""

from __future__ import annotations

import pytest

from robolab_core.versioning import SemVer, VersionRange


class TestSemVer:
    def test_parse_plain(self):
        v = SemVer.parse("1.2.3")
        assert (v.major, v.minor, v.patch) == (1, 2, 3)
        assert v.prerelease == ()

    def test_parse_prerelease_and_build(self):
        v = SemVer.parse("1.0.0-alpha.1+build.5")
        assert v.prerelease == ("alpha", "1")
        assert v.build == "build.5"

    @pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0.0", "1.02.3", ""])
    def test_reject_invalid(self, bad):
        with pytest.raises(ValueError):
            SemVer.parse(bad)

    def test_ordering(self):
        assert SemVer.parse("1.0.0").compare(SemVer.parse("2.0.0")) < 0
        assert SemVer.parse("1.0.0-alpha").compare(SemVer.parse("1.0.0")) < 0
        assert SemVer.parse("1.0.0-alpha").compare(SemVer.parse("1.0.0-alpha.1")) < 0
        assert SemVer.parse("1.0.0-alpha.1").compare(SemVer.parse("1.0.0-alpha.beta")) < 0
        assert SemVer.parse("1.0.0+build").compare(SemVer.parse("1.0.0")) == 0


class TestVersionRange:
    def test_single_comparator(self):
        assert VersionRange.parse(">=1.0.0").contains("1.5.2")
        assert not VersionRange.parse(">=1.0.0").contains("0.9.9")

    def test_and_semantics(self):
        r = VersionRange.parse(">=1.0.0 <2.0.0")
        assert r.contains("1.0.0")
        assert r.contains("1.99.0")
        assert not r.contains("2.0.0")
        assert not r.contains("0.9.0")

    def test_bare_version_means_exact(self):
        r = VersionRange.parse("1.2.3")
        assert r.contains("1.2.3")
        assert not r.contains("1.2.4")

    def test_not_equal(self):
        r = VersionRange.parse("!=1.0.0 >=0.9.0")
        assert r.contains("1.0.1")
        assert not r.contains("1.0.0")

    def test_invalid_clause(self):
        with pytest.raises(ValueError):
            VersionRange.parse(">=1.0")
