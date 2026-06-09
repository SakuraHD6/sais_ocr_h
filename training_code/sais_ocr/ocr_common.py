"""Shared OCR parsing and geometry helpers for the SAIS rebuild.

The competition XML files are not uniform: coordinates can be semicolon point
lists, comma rectangles, or comma point lists, and XML page dimensions can differ
from the actual PNG dimensions. This module is the single source of truth for
parsing labels and boxes so training and evaluation use the same GT contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_int_xyxy(self) -> list[int]:
        return [
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        ]


@dataclass(frozen=True)
class AnnotationRecord:
    image_id: str
    image_path: str
    xml_path: str
    label: str
    bbox: Box
    raw_position: str
    xml_size: tuple[int, int]
    image_size: tuple[int, int]

    def to_json(self) -> dict:
        data = asdict(self)
        data["bbox"] = self.bbox.to_int_xyxy()
        return data


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def read_image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_label(text: str | None) -> str:
    if text is None:
        return ""
    return "".join(str(text).split())


def normalize_stem_for_pairing(stem: str) -> str:
    """Normalize known mojibake variants in dataset filenames for pairing."""
    return (
        stem.replace("ú¿", "(")
        .replace("ú⌐", ")")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
    )


def collect_image_xml_pairs(src: str | Path) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    """Pair PNG and XML files by exact stem first, then known bracket variants."""
    src = Path(src)
    png_by_stem = {p.stem: p for p in src.glob("*.png")}
    xml_by_stem = {p.stem: p for p in src.glob("*.xml")}
    pairs: dict[str, tuple[str, Path, Path]] = {}

    for stem in sorted(set(png_by_stem) & set(xml_by_stem)):
        pairs[stem] = (stem, png_by_stem[stem], xml_by_stem[stem])

    used_png = {p for _, p, _ in pairs.values()}
    used_xml = {x for _, _, x in pairs.values()}
    remaining_png = [p for p in png_by_stem.values() if p not in used_png]
    remaining_xml = [x for x in xml_by_stem.values() if x not in used_xml]

    xml_norm: dict[str, list[Path]] = {}
    for xml_path in remaining_xml:
        xml_norm.setdefault(normalize_stem_for_pairing(xml_path.stem), []).append(xml_path)

    for png_path in sorted(remaining_png):
        norm = normalize_stem_for_pairing(png_path.stem)
        matches = xml_norm.get(norm, [])
        if len(matches) != 1:
            continue
        xml_path = matches[0]
        pairs[png_path.stem] = (png_path.stem, png_path, xml_path)
        used_png.add(png_path)
        used_xml.add(xml_path)

    png_unpaired = sorted(p.stem for p in png_by_stem.values() if p not in used_png)
    xml_unpaired = sorted(x.stem for x in xml_by_stem.values() if x not in used_xml)
    return sorted(pairs.values(), key=lambda item: item[0]), png_unpaired, xml_unpaired


def parse_position(position: str | None) -> Box | None:
    """Parse SAIS XML position strings into an xyxy box.

    Supported forms:
    - ``x,y;x,y;...`` polygon/points
    - ``x1,y1,x2,y2`` rectangle
    - ``x1,y1,x2,y2,x3,y3,...`` polygon/points
    """
    if not position:
        return None
    values = [float(v) for v in _NUMBER_RE.findall(position)]
    if len(values) < 4:
        return None

    if len(values) == 4:
        x1, y1, x2, y2 = values
        return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    if len(values) % 2 != 0:
        values = values[:-1]
    xs = values[0::2]
    ys = values[1::2]
    if not xs or not ys:
        return None
    return Box(min(xs), min(ys), max(xs), max(ys))


def box_fits(box: Box, width: int, height: int, margin: float = 2.0) -> bool:
    return (
        box.x1 >= -margin
        and box.y1 >= -margin
        and box.x2 <= width + margin
        and box.y2 <= height + margin
    )


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    x1 = min(max(box.x1, 0.0), float(width))
    y1 = min(max(box.y1, 0.0), float(height))
    x2 = min(max(box.x2, 0.0), float(width))
    y2 = min(max(box.y2, 0.0), float(height))
    out = Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if out.area <= 0:
        return None
    return out


def scale_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        return box
    sx = tw / fw
    sy = th / fh
    return Box(box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy)


def resolve_box_to_image(
    raw_box: Box,
    xml_size: tuple[int, int],
    image_size: tuple[int, int],
) -> Box | None:
    """Resolve XML coordinates into the actual PNG coordinate system.

    This follows the documented repaired GT rule from the competition notes:
    if a raw box is fully inside the XML page coordinate system and XML/page
    dimensions differ from the PNG, scale it to PNG dimensions; otherwise keep
    the raw box if it already fits the PNG. The final box is clamped.
    """
    xml_w, xml_h = xml_size
    img_w, img_h = image_size
    candidates: list[Box] = []

    if xml_w > 0 and xml_h > 0 and (xml_w, xml_h) != (img_w, img_h):
        if box_fits(raw_box, xml_w, xml_h):
            candidates.append(scale_box(raw_box, xml_size, image_size))

    if box_fits(raw_box, img_w, img_h):
        candidates.append(raw_box)

    if not candidates and xml_w > 0 and xml_h > 0:
        candidates.append(scale_box(raw_box, xml_size, image_size))
    if not candidates:
        candidates.append(raw_box)

    for candidate in candidates:
        clamped = clamp_box(candidate, img_w, img_h)
        if clamped is not None and clamped.width >= 1 and clamped.height >= 1:
            return clamped
    return None


def parse_xml_records(
    xml_path: str | Path,
    image_path: str | Path | None = None,
    *,
    min_box_size: float = 1.0,
) -> list[AnnotationRecord]:
    xml_path = Path(xml_path)
    image_path = Path(image_path) if image_path is not None else xml_path.with_suffix(".png")
    image_size = read_image_size(image_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_w = int(float(root.attrib.get("width", "0") or 0))
    xml_h = int(float(root.attrib.get("height", "0") or 0))
    xml_size = (xml_w, xml_h)
    image_id = image_path.stem

    records: list[AnnotationRecord] = []
    for elem in root.iter():
        if local_name(elem.tag) != "char":
            continue
        label = normalize_label(elem.text)
        position = elem.attrib.get("position", "")
        raw_box = parse_position(position)
        if not label or raw_box is None:
            continue
        bbox = resolve_box_to_image(raw_box, xml_size, image_size)
        if bbox is None or bbox.width < min_box_size or bbox.height < min_box_size:
            continue
        records.append(
            AnnotationRecord(
                image_id=image_id,
                image_path=str(image_path),
                xml_path=str(xml_path),
                label=label,
                bbox=bbox,
                raw_position=position,
                xml_size=xml_size,
                image_size=image_size,
            )
        )
    return records


def bbox_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def yolo_line_from_box(box: Box, image_size: tuple[int, int], class_id: int) -> str:
    width, height = image_size
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    bw = box.width / width
    bh = box.height / height
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def dump_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_finite_box(box: Box) -> bool:
    return all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2))
