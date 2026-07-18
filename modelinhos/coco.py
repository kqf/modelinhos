import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests  # type: ignore
from dataclasses_json import Undefined, dataclass_json
from tqdm import tqdm

from modelinhos.sample import (
    Annotation,
    RelativeXYXY,
    Sample,
    read_samples,
    save_samples,
)

logger = logging.getLogger(__name__)

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/{split}.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class CocoCategory:
    id: int
    name: str


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    iscrowd: int = 0


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class CocoDataset:
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]


def _download_file(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    with (
        destination.open("wb") as fh,
        tqdm(
            desc=destination.name,
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=8_192):
            fh.write(chunk)
            progress.update(len(chunk))


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract *archive_path* into *extract_to*."""
    logger.info("Extracting %s → %s", archive_path.name, extract_to)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_to)


def download_split(root: Path, split: str) -> Path:
    """Download and prepare a COCO 2017 *split* under *root*.

    The two COCO archives unpack with a flat layout::

        <split>.zip                  →  <root>/<split>/<images>
        annotations_trainval2017.zip →  <root>/annotations/<json files>

    After extraction we relocate the image folder so the final structure is::

        <root>/
          images/
            <split>/          ← images end up here
          annotations/
            instances_<split>.json

    Returns the annotations path.
    """
    images = root / "images" / split
    annotations = root / "annotations" / f"instances_{split}.json"

    if images.exists() and annotations.exists():
        logger.info("COCO %s already present at %s, skipping ", split, root)
        return annotations

    root.mkdir(parents=True, exist_ok=True)

    archives = {
        f"{split}.zip": COCO_IMAGES_URL.format(split=split),
        "annotations_trainval2017.zip": COCO_ANNOTATIONS_URL,
    }
    for archive_name, url in archives.items():
        archive_path = root / archive_name
        if not archive_path.exists():
            logger.info("Downloading %s", archive_name)
            _download_file(url, archive_path)
        _extract_archive(archive_path, root)

    # <split>.zip unpacks to <root>/<split>/; move it under images/ to match
    # the expected layout described in the docstring above.
    extracted = root / split
    if extracted.exists():
        images.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(images)

    return annotations


# COCO annotations are pixel-space xywh; this is the ingest boundary,
# so convert to relative xyxy here and pixels never appear upstream.
def _bbox_xywh_to_xyxy(
    bbox: list[float],
    width: int,
    height: int,
) -> RelativeXYXY:
    x, y, w, h = bbox
    return x / width, y / height, (x + w) / width, (y + h) / height


def to_samples(annotations_path: Path, split: str) -> list[Sample]:
    coco: CocoDataset = CocoDataset.from_json(annotations_path.read_text())  # type: ignore

    category_name = {cat.id: cat.name for cat in coco.categories}
    annotations_by_image: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for ann in coco.annotations:
        if ann.iscrowd:
            continue
        annotations_by_image[ann.image_id].append(ann)

    return [
        Sample(
            file_name=Path(f"images/{split}/{image.file_name}"),
            annotations=[
                Annotation(
                    bbox=_bbox_xywh_to_xyxy(
                        ann.bbox, image.width, image.height
                    ),
                    label=category_name[ann.category_id],
                )
                for ann in annotations_by_image[image.id]
            ],
        )
        for image in coco.images
    ]


def download(output: Path, split: str = "val2017") -> Path:
    annotations = download_split(output.parent, split)
    samples = to_samples(annotations, split)
    save_samples(samples, output)
    return output


def load_samples(annotations: Path, split: str = "val2017") -> list[Sample]:
    if not annotations.exists():
        download(annotations, split)
    return read_samples(annotations)
