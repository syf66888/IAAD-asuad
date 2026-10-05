# Dataset Annotations

The supplied COCO annotation JSONs contain `action`, `justification`, and `caption` fields. The `_des` and `_exp` directories provide the corresponding description and explanation references.

Prepare videos or existing frame folders with `python scripts/prepare_dataset.py`. See the repository README for the complete command. The script generates the caption TSVs, frame TSVs, and offline LK features required by the model.
