# Installation

1. Create a Python 3.10+ environment.
2. Install the pinned dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set `data.root` to the directory containing the downloaded gRefCOCO and COCO files.
4. Export `PYTHONPATH`:

   ```bash
   export PYTHONPATH="$PWD"
   ```

The repository does not download datasets automatically. Download the checkpoint from [Richard-ZZZZZ/ContourVLA](https://huggingface.co/Richard-ZZZZZ/ContourVLA) and place it at `ckpts/contourvla.pt`:

```bash
huggingface-cli download Richard-ZZZZZ/ContourVLA contourvla.pt --local-dir ckpts
```

See [DATASETS.md](DATASETS.md) for public download sources and the expected directory layout.
