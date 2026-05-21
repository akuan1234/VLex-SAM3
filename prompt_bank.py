from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _normalize_prompt(text: str) -> str:
    text = text.replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _parse_legacy_txt(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            prompts = [_normalize_prompt(part) for part in line.split(",")]
            prompts = [prompt for prompt in prompts if prompt]
            if not prompts:
                raise ValueError(f"Line {line_idx} in {path} does not contain any valid prompts.")
            entries.append(
                {
                    "class_name": prompts[0],
                    "prompts": _dedupe_keep_order(prompts),
                }
            )
    if not entries:
        raise ValueError(f"Prompt bank {path} is empty.")
    return entries


def _parse_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "classes" in payload:
        raw_entries = payload["classes"]
    elif isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, dict):
        raw_entries = [
            {"class_name": class_name, "prompts": prompts}
            for class_name, prompts in payload.items()
        ]
    else:
        raise ValueError(
            f"Unsupported prompt bank structure in {path}. "
            "Use a list, a dict[class_name -> prompts], or {'classes': [...]}."
        )

    entries: list[dict] = []
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Class entry #{idx} in {path} must be a JSON object.")

        class_name = _normalize_prompt(str(raw_entry.get("class_name", f"class_{idx}")))
        prompts = raw_entry.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list):
            raise ValueError(f"'prompts' for class '{class_name}' in {path} must be a list or string.")

        prompts = [_normalize_prompt(str(prompt)) for prompt in prompts]
        prompts = [prompt for prompt in prompts if prompt]
        if not prompts:
            prompts = [class_name]

        entries.append(
            {
                "class_name": class_name,
                "prompts": _dedupe_keep_order(prompts),
                "label_value": raw_entry.get("label_value"),
            }
        )

    if not entries:
        raise ValueError(f"Prompt bank {path} is empty.")
    return entries


def load_prompt_bank(path: str | Path) -> list[dict]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt bank not found: {prompt_path}")

    suffix = prompt_path.suffix.lower()
    if suffix == ".json":
        entries = _parse_json(prompt_path)
    else:
        entries = _parse_legacy_txt(prompt_path)

    for idx, entry in enumerate(entries):
        prompts = entry["prompts"]
        if not prompts:
            raise ValueError(f"Class #{idx} in {prompt_path} has no valid prompts.")
        if any("," in prompt for prompt in prompts) and suffix != ".json":
            raise ValueError(
                f"Legacy txt prompt banks cannot contain commas inside a phrase. "
                f"Use JSON for class #{idx} in {prompt_path}."
            )

    return entries


def load_class_prompts(path: str | Path) -> tuple[list[str], list[int]]:
    entries = load_prompt_bank(path)

    class_prompts: list[str] = []
    class_indices: list[int] = []
    for class_idx, entry in enumerate(entries):
        prompts = entry["prompts"]
        class_prompts.extend(prompts)
        class_indices.extend([class_idx] * len(prompts))
    return class_prompts, class_indices


def save_legacy_txt(entries: list[dict], path: str | Path) -> None:
    prompt_path = Path(path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for idx, entry in enumerate(entries):
        prompts = [_normalize_prompt(str(prompt)) for prompt in entry.get("prompts", [])]
        prompts = [prompt for prompt in prompts if prompt]
        prompts = _dedupe_keep_order(prompts)
        if not prompts:
            raise ValueError(f"Class #{idx} cannot be exported because it has no prompts.")
        if any("," in prompt for prompt in prompts):
            raise ValueError(
                f"Prompt '{next(prompt for prompt in prompts if ',' in prompt)}' contains a comma. "
                "Export to JSON or remove commas before saving as txt."
            )
        lines.append(",".join(prompts))

    prompt_path.write_text("\n".join(lines), encoding="utf-8")


def save_prompt_txt(entries: list[dict], path: str | Path) -> None:
    """Backward-compatible alias for exporting legacy txt prompt banks."""
    save_legacy_txt(entries, path)
