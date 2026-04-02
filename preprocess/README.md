# Optional Preprocessing

This directory keeps the lightweight preprocessing helpers used by the project.

- `estimate_depth_and_rescale.py`
  Estimates monocular inverse depth and aligns it to COLMAP sparse geometry.
- `colmap_reader.py`
  COLMAP model reader used by the depth-rescaling script.

The original internal workspace called `./preprocess/Depth-Anything-V2/run.py` from `estimate_depth_and_rescale.py`. That external dependency is not bundled in this cleaned release. If you need the pseudo-depth pipeline, clone or install Depth-Anything-V2 separately and mirror the expected path or adapt the script accordingly.

