"""Load XR voice command intents from yaml/xr_voice_commands.yaml."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "yaml" / "xr_voice_commands.yaml"


@dataclass(frozen=True)
class VoiceIntent:
    intent_id: str
    label: str
    feature: str
    mcp_tool: str
    requires_camera: bool
    requires_prior_mesh_job: bool
    data_topic: str | None
    examples: tuple[str, ...]
    pattern: re.Pattern[str]
    skip_if_matched: tuple[str, ...]


@dataclass(frozen=True)
class VoiceCommandDatabase:
    version: int
    routing: tuple[str, ...]
    intents: dict[str, VoiceIntent]

    def match(self, query: str) -> VoiceIntent | None:
        text = query.strip()
        if not text:
            return None
        matched_ids: set[str] = set()
        for intent_id in self.routing:
            intent = self.intents.get(intent_id)
            if intent is None:
                continue
            if intent.skip_if_matched and any(
                skip in matched_ids for skip in intent.skip_if_matched
            ):
                continue
            if intent.pattern.search(text):
                matched_ids.add(intent_id)
                return intent
        return None

    def examples_for(self, intent_id: str) -> tuple[str, ...]:
        intent = self.intents.get(intent_id)
        return intent.examples if intent else ()


def _compile_intent(intent_id: str, raw: dict[str, Any]) -> VoiceIntent:
    patterns = raw.get("patterns") or []
    if not patterns:
        raise ValueError(f"intent {intent_id!r} has no patterns")
    combined = "(?:" + "|".join(f"(?:{p})" for p in patterns) + ")"
    return VoiceIntent(
        intent_id=intent_id,
        label=str(raw.get("label") or intent_id),
        feature=str(raw.get("feature") or intent_id),
        mcp_tool=str(raw.get("mcp_tool") or intent_id),
        requires_camera=bool(raw.get("requires_camera")),
        requires_prior_mesh_job=bool(raw.get("requires_prior_mesh_job")),
        data_topic=raw.get("data_topic"),
        examples=tuple(str(x) for x in (raw.get("examples") or [])),
        pattern=re.compile(combined, re.IGNORECASE),
        skip_if_matched=tuple(str(x) for x in (raw.get("skip_if_matched") or [])),
    )


def load_voice_command_database(path: Path | str | None = None) -> VoiceCommandDatabase:
    yaml_path = Path(
        path
        or os.environ.get("XR_VOICE_COMMANDS_YAML")
        or _DEFAULT_YAML
    )
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    routing = tuple(str(x) for x in (data.get("routing") or []))
    raw_intents = data.get("intents") or {}
    intents = {
        intent_id: _compile_intent(intent_id, raw)
        for intent_id, raw in raw_intents.items()
    }
    return VoiceCommandDatabase(
        version=int(data.get("version") or 1),
        routing=routing,
        intents=intents,
    )


@lru_cache(maxsize=4)
def get_voice_command_database(path: str | None = None) -> VoiceCommandDatabase:
    return load_voice_command_database(path)


def match_xr_voice_intent(query: str, path: str | None = None) -> VoiceIntent | None:
    return get_voice_command_database(path).match(query)


def wants_mesh_generation(query: str, path: str | None = None) -> bool:
    intent = match_xr_voice_intent(query, path)
    return intent is not None and intent.intent_id == "image_to_textured_mesh"


def wants_optional_text_mesh(query: str, path: str | None = None) -> bool:
    intent = match_xr_voice_intent(query, path)
    return intent is not None and intent.intent_id == "text_to_textured_mesh"


def wants_world_generation(query: str, path: str | None = None) -> bool:
    intent = match_xr_voice_intent(query, path)
    return intent is not None and intent.intent_id == "image_to_world"


def wants_rig_generation(query: str, path: str | None = None) -> bool:
    intent = match_xr_voice_intent(query, path)
    return intent is not None and intent.intent_id == "auto_rig"
