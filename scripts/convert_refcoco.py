#!/usr/bin/env python3
"""Convert RefCOCO / RefCOCO+ parquet annotations into the loader's format.

The referring annotations are normally distributed as ``refs(unc).p``. A public
mirror is
``jxu124/refcoco`` and ``jxu124/refcocoplus`` on HuggingFace, which carry each
reference's *original* COCO annotation verbatim in a ``raw_anns`` JSON string.

That "verbatim" is the whole reason this source was chosen over the more obvious
``lmms-lab/RefCOCO``. The lmms-eval export flattens ``segmentation`` from
``list[list[float]]`` to a single ``list[float]``, concatenating a multi-part
object's rings with no separator and no sentinel. Filling that flat array as
one ring joins disjoint parts with straight lines, and the split cannot be
recovered. It also inverts the task: its prompt
asks for a caption of a circled region, so the expression is the *answer* rather
than the input. This script's source keeps the rings nested and keeps one
reference per row with its sentences attached.

Output is what ``contourvla/data/datasets.py`` already reads for
``dataset: refcoco``, so nothing downstream needs to change:

    <root>/<name>/refs(unc).p     references, with split / image_id / ann_id / sentences
    <root>/<name>/instances.json  the images and annotations those ann_ids index

Images are not written. ``raw_image_info`` names a COCO train2014 file, which
the loader reads from ``data/coco/images/train2014``.

    python scripts/convert_refcoco.py \
        --parquet data/refcoco_ann --out data/refcoco
    python scripts/convert_refcoco.py \
        --parquet data/refcocoplus_ann --out data/refcoco+
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rows(parquet_dir: str) -> list[dict]:
    """Every row of every parquet in the directory, in filename order."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no .parquet under {parquet_dir!r}")

    rows: list[dict] = []
    for path in files:
        table = pq.ParquetFile(path).read().to_pylist()
        rows.extend(table)
        print(f"  {os.path.basename(path):16s} {len(table):6d} rows")
    return rows


def convert(rows: list[dict]) -> tuple[list[dict], dict]:
    """Split the rows into (references, instances), deduplicating by id.

    One row is one reference, and several references share an image, so both the
    image and the annotation dictionaries have to be keyed rather than appended -
    otherwise ``instances.json`` carries an entry per reference and the loader's
    ``{int(ann['id']): ann}`` silently keeps only the last of each.
    """
    from contourvla.progress import Progress

    refs: list[dict] = []
    images: dict[int, dict] = {}
    annotations: dict[int, dict] = {}
    skipped_rle = 0

    bar = Progress(len(rows), prefix="  parsing ", interval=2000)
    for row in rows:
        bar.update()
        annotation = json.loads(row["raw_anns"])
        image_info = json.loads(row["raw_image_info"])

        # A crowd annotation stores RLE rather than polygons. The loader accepts
        # both, so these are kept and only counted - dropping them would change
        # the split, and the benchmark measures against them.
        if isinstance(annotation.get("segmentation"), dict):
            skipped_rle += 1

        image_id = int(image_info["id"])
        images.setdefault(image_id, {
            "id": image_id,
            "file_name": image_info["file_name"],
            "width": int(image_info["width"]),
            "height": int(image_info["height"]),
        })

        ann_id = int(annotation["id"])
        annotations.setdefault(ann_id, annotation)

        refs.append({
            "ref_id": int(row["ref_id"]),
            "image_id": image_id,
            # A list, because the loader reads gRefCOCO's multi-target spelling
            # and treats a scalar as a one-element list anyway. RefCOCO is
            # single-target, so this is always length 1.
            "ann_id": [ann_id],
            "split": row["split"],
            "sentences": [
                {"sent": s["sent"], "raw": s.get("raw", s["sent"]),
                 "sent_id": int(s["sent_id"])}
                for s in row["sentences"]
            ],
            "category_id": int(row["category_id"]),
            "file_name": image_info["file_name"],
        })

    bar.close()
    if skipped_rle:
        print(f"  {skipped_rle} crowd (RLE) annotations kept as-is")
    return refs, {"images": list(images.values()),
                  "annotations": list(annotations.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True,
                        help="directory of jxu124 parquet files")
    parser.add_argument("--out", required=True,
                        help="destination, e.g. data/refcoco")
    parser.add_argument("--split-by", default="unc",
                        help="names the refs file: refs(<split_by>).p")
    args = parser.parse_args()

    print(f"reading {args.parquet}")
    rows = load_rows(args.parquet)
    refs, instances = convert(rows)

    os.makedirs(args.out, exist_ok=True)
    refs_path = os.path.join(args.out, f"refs({args.split_by}).p")
    instances_path = os.path.join(args.out, "instances.json")
    with open(refs_path, "wb") as handle:
        pickle.dump(refs, handle)
    with open(instances_path, "w", encoding="utf-8") as handle:
        json.dump(instances, handle)

    from collections import Counter
    by_split = Counter(r["split"] for r in refs)
    sentences = Counter()
    for r in refs:
        sentences[r["split"]] += len(r["sentences"])

    print(f"\nwrote {refs_path}")
    print(f"      {instances_path}")
    print(f"      {len(instances['images'])} images, "
          f"{len(instances['annotations'])} annotations")
    print(f"{'split':8s} {'refs':>7s} {'samples':>9s}")
    for split in sorted(by_split):
        print(f"{split:8s} {by_split[split]:7d} {sentences[split]:9d}")


if __name__ == "__main__":
    main()
