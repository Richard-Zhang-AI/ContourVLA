# Datasets

## gRefCOCO

Download the referring-expression annotations from the public [gRefCOCO dataset page](https://huggingface.co/datasets/FudanCVL/gRefCOCO). Place `grefs(unc).json` and `instances.json` under `data/grefcoco/`.

## COCO train2014

gRefCOCO uses COCO train2014 images. Download:

- [train2014 images](http://images.cocodataset.org/zips/train2014.zip)
- [COCO 2014 annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

Extract the images to `data/coco/images/train2014/`. The loader only needs the image files and the gRefCOCO `instances.json` file, but the standard annotation archive is useful for checking image IDs.

## RefCOCO, RefCOCO+, and RefCOCOg

Public annotation mirrors are available for [RefCOCO](https://huggingface.co/datasets/jxu124/refcoco), [RefCOCO+](https://huggingface.co/datasets/jxu124/refcocoplus), and [RefCOCOg](https://huggingface.co/datasets/jxu124/refcocog). For the RefCOCO family, obtain the original `refs(unc).p` or `refs(umd).p` files and the matching COCO annotations. Put each reference file and an `instances.json` file under its dataset directory, for example `data/refcoco/`. The same loader supports these datasets when the YAML `data.dataset` and `data.split_by` values are changed.

The helper `scripts/convert_refcoco.py` converts the parquet mirrors to the nested annotation format used by the loader.

## Expected layout

```text
data/
  grefcoco/
    grefs(unc).json
    instances.json
  coco/
    images/train2014/*.jpg
```

The loader accepts `gRefCOCO` as an alternative spelling of the annotation directory. Set `data.image_root` only when the images are stored outside the default COCO path.

## Licensing

The datasets are distributed by their respective owners. Follow the COCO and gRefCOCO terms when downloading, redistributing, or publishing derived examples.
