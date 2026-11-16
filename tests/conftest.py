import pytest


@pytest.fixture(autouse=True)
def _sidecars_in_temp(tmp_path, monkeypatch):
    """Runs inside tests write their audit and report files to a temp folder, not the repo's out/."""
    monkeypatch.setenv("SPENDCUBE_SIDECAR_DIR", str(tmp_path / "sidecars"))
