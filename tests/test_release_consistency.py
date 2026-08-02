from pathlib import Path
import re
import tomllib

import transfermod

ROOT = Path(__file__).resolve().parents[1]


def test_version_metadata_agree():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == transfermod.__version__ == "1.2.2"
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 1.2.2" in citation
    assert "TransferMod v1.2.2" in (ROOT / "RELEASE_MANIFEST.md").read_text(encoding="utf-8")

    manifest = (ROOT / "RELEASE_MANIFEST.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/releases/1.2.2/RELEASE_NOTES.md").read_text(encoding="utf-8")

    manifest_count = re.search(r"`(\d+)` tests passed", manifest)
    notes_count = re.search(r"- (\d+) tests passed", release_notes)
    assert manifest_count and notes_count
    assert notes_count.group(1) == manifest_count.group(1)

    coverage_claim = re.search(
        r"latest completed branch-aware coverage measurement is the (v\d+\.\d+\.\d+) baseline",
        manifest,
    )
    assert coverage_claim
    assert f"{coverage_claim.group(1)} baseline" in release_notes


def test_license_and_citation_are_present():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "license: AGPL-3.0-only" in citation


def test_manifest_declared_paths_exist():
    manifest = (ROOT / "RELEASE_MANIFEST.md").read_text(encoding="utf-8")
    for path in re.findall(r"`([^`]+(?:\.md|\.py|\.toml|\.yml|\.json|\.png))`", manifest):
        if any(ch in path for ch in "*{}") or path.startswith("python "):
            continue
        candidates = (ROOT / path, ROOT / "scripts" / path, ROOT / "results" / path)
        assert any(candidate.exists() for candidate in candidates), (
            f"manifest path does not exist: {path}"
        )


def test_local_markdown_links_resolve():
    for document in (ROOT / "README.md", ROOT / "RELEASE_MANIFEST.md", ROOT / "TUTORIAL_PERTURBATION_FAMILIES.md"):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            assert (document.parent / clean).exists(), f"broken link in {document.name}: {target}"
