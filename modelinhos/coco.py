import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests  # type: ignore
from dataclasses_json import Undefined, dataclass_json
from tqdm import tqdm

from modelinhos.sample import (
    AbsoluteXYXY,
    Annotation,
    Sample,
    read_samples,
    save_samples,
)

logger = logging.getLogger(__name__)

COCO_FILES = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}


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


def download_validation(root: Path) -> Path:
    """Download and prepare the COCO val2017 split under *root*.

    The two COCO archives unpack with a flat layout::

        val2017.zip                  →  <root>/val2017/<images>
        annotations_trainval2017.zip →  <root>/annotations/<json files>

    After extraction we relocate the image folder so the final structure is::

        <root>/
          images/
            val2017/          ← images end up here
          annotations/
            instances_val2017.json

    Returns ``(images_dir, annotations_path)``.
    """
    images = root / "images" / "val2017"
    annotations = root / "annotations" / "instances_val2017.json"

    if images.exists() and annotations.exists():
        logger.info("COCO val2017 already present at %s, skipping ", root)
        return annotations

    root.mkdir(parents=True, exist_ok=True)

    for archive_name, url in COCO_FILES.items():
        archive_path = root / archive_name
        if not archive_path.exists():
            logger.info("Downloading %s", archive_name)
            _download_file(url, archive_path)
        _extract_archive(archive_path, root)

    # val2017.zip unpacks to <root>/val2017/; move it under images/ to match
    # the expected layout described in the docstring above.
    extracted = root / "val2017"
    if extracted.exists():
        images.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(images)

    return annotations


def _bbox_xywh_to_xyxy(bbox: list[float]) -> AbsoluteXYXY:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def to_samples(annotations_path: Path) -> list[Sample]:
    coco: CocoDataset = CocoDataset.from_json(annotations_path.read_text())  # type: ignore

    category_name = {cat.id: cat.name for cat in coco.categories}
    annotations_by_image: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for ann in coco.annotations:
        if ann.iscrowd:
            continue
        annotations_by_image[ann.image_id].append(ann)

    return [
        Sample(
            file_name=f"images/val2017/{image.file_name}",
            annotations=[
                Annotation(
                    bbox=_bbox_xywh_to_xyxy(ann.bbox),
                    label=category_name[ann.category_id],
                )
                for ann in annotations_by_image[image.id]
            ],
        )
        for image in coco.images
    ]


def download(output: Path) -> Path:
    annotations = download_validation(output.parent)
    samples = to_samples(annotations)
    save_samples(samples, output)
    return output


def load_samples(annotations: Path) -> list[Sample]:
    if not annotations.exists():
        download(annotations)
    return read_samples(annotations)
