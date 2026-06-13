"""Tests for API avatar rig contract validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.utils.aigc_rig_contract import (
    CHARACTER_FACING_BACKWARDS,
    CHARACTER_UPSIDE_DOWN,
    validate_aigc_rigged_glb,
)

OLD_BROKEN = Path(
    "/home/sifr/3DAIGC-API/outputs/rigged/"
    "unirig_auto_rig_upload_f74e7c9f-e1c0-486d-884e-d0a148afaaa9_1781011886.glb"
)
FIXED = Path("/tmp/test_template_rig_contract.glb")


@pytest.mark.skipif(not OLD_BROKEN.is_file(), reason="broken sample GLB missing")
def test_contract_fails_on_old_inverted_export():
    result = validate_aigc_rigged_glb(OLD_BROKEN)
    assert not result.passed
    assert CHARACTER_UPSIDE_DOWN in result.codes


@pytest.mark.skipif(not FIXED.is_file(), reason="fixed sample GLB missing")
def test_contract_passes_critical_on_fixed_export():
    result = validate_aigc_rigged_glb(FIXED)
    assert result.passed
    assert CHARACTER_UPSIDE_DOWN not in result.codes
    assert CHARACTER_FACING_BACKWARDS not in result.codes
    assert result.metrics["headY"] > result.metrics["footY"]
