"""Load a markdown corpus with simple front matter and chunk it by heading."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import DocChunk

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Parse ``key: value`` front matter (a deliberately tiny YAML subset)."""
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, raw[m.end():]


def chunk_document(doc_id: str, meta: dict[str, str], body: str, max_chars: int = 1800) -> list[DocChunk]:
    """Split on H1-H3 headings, then on paragraphs if a section is still too long.

    Each chunk is prefixed with its document title and section so it reads on its own
    in the agent's context.
    """
    title = meta.get("title", doc_id)
    doc_type = meta.get("type", "unknown")
    headings = list(_HEADING.finditer(body))
    sections: list[tuple[str, str]] = []
    if not headings or headings[0].start() > 0:
        pre = body[: headings[0].start()] if headings else body
        if pre.strip():
            sections.append(("Overview", pre.strip()))
    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[h.end():end].strip()
        if text:
            sections.append((h.group(2).strip(), text))

    chunks: list[DocChunk] = []
    seen: dict[str, int] = {}
    for section, text in sections:
        parts: list[str] = []
        buf = ""
        for para in text.split("\n\n"):
            if buf and len(buf) + len(para) > max_chars:
                parts.append(buf)
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            parts.append(buf)
        for j, part in enumerate(parts):
            slug = _slug(section) + (f"-{j + 1}" if len(parts) > 1 else "")
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = f"{slug}-{seen[slug]}"
            chunks.append(
                DocChunk(
                    chunk_id=f"{doc_id}#{slug}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    doc_type=doc_type,
                    text=part.strip(),
                    metadata={k: v for k, v in meta.items() if k not in {"title", "type"}},
                )
            )
    return chunks


def load_corpus(corpus_dir: Path) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for path in sorted(Path(corpus_dir).rglob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("id", path.stem)
        chunks.extend(chunk_document(doc_id, meta, body))
    return chunks
