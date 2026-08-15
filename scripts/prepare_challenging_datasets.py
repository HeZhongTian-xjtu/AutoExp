"""Normalize the downloaded challenge datasets into AutoExp's train.csv contract.

The downloaded archives stay outside Git. This script is intentionally repeatable
so a fresh checkout can rebuild the local demo assets from the source archives.
"""

from __future__ import annotations

import csv
import gzip
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets"
SOURCE_ROOT = DATASET_ROOT / "sources"
OUTPUT_ROOT = DATASET_ROOT / "builtin"

COVERTYPE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    *[f"Wilderness_Area{i}" for i in range(1, 5)],
    *[f"Soil_Type{i}" for i in range(1, 41)],
    "Cover_Type",
]


def main() -> None:
    _require_sources()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_covertype()
    prepare_bank_marketing()
    prepare_online_shoppers()
    print(f"Prepared datasets under {OUTPUT_ROOT}")


def _require_sources() -> None:
    required = [
        SOURCE_ROOT / "covertype" / "source.zip",
        SOURCE_ROOT / "bank_marketing" / "raw" / "bank" / "bank-full.csv",
        SOURCE_ROOT / "online_shoppers" / "raw" / "online_shoppers_intention.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing local dataset source files under datasets/sources:\n"
            f"{rendered}\nSee README.md for the expected dataset layout."
        )


def prepare_covertype() -> None:
    source_zip = SOURCE_ROOT / "covertype" / "source.zip"
    destination = OUTPUT_ROOT / "covertype-v1" / "data" / "train.csv"
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(source_zip) as archive,
        archive.open("covtype.data.gz") as compressed,
    ):
        with (
            gzip.GzipFile(fileobj=compressed) as stream,
            destination.open("w", newline="", encoding="utf-8") as output,
        ):
            writer = csv.writer(output)
            writer.writerow(COVERTYPE_COLUMNS)
            for line in stream:
                row = line.decode("utf-8").strip().split(",")
                if len(row) != len(COVERTYPE_COLUMNS):
                    raise ValueError(f"Unexpected Covertype row width: {len(row)}")
                writer.writerow(row)


def prepare_bank_marketing() -> None:
    source = SOURCE_ROOT / "bank_marketing" / "raw" / "bank" / "bank-full.csv"
    destination = OUTPUT_ROOT / "bank-marketing-v1" / "data" / "train.csv"
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(newline="", encoding="utf-8") as input_file,
        destination.open("w", newline="", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file, delimiter=";")
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        writer.writerows(reader)


def prepare_online_shoppers() -> None:
    source = SOURCE_ROOT / "online_shoppers" / "raw" / "online_shoppers_intention.csv"
    destination = OUTPUT_ROOT / "online-shoppers-v1" / "data" / "train.csv"
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
