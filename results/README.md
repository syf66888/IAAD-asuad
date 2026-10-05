# Best Model Results

`bddx/` and `mmau/` contain test outputs from the corresponding best models. Each directory includes:

- `testing_predictions.json`: video ID, generated description, and generated explanation or prevention advice.
- Per-task COCO prediction JSONs and evaluation JSONs.
- `metrics.json`: BLEU, CIDEr, and ROUGE-L scores.

The BDDX directory also includes `turn_accuracy.json` with overall and per-direction turning scores, and `turn_predictions.json` with the action reference, description, labels, and outcome for each sample. Turning accuracy is evaluated on annotated explicit left/right turns using description text.

The same output files accompany the downloadable model weights. Caption metric JSONs use the evaluator's raw scale; the repository README displays those scores multiplied by 100. Turning summaries include accuracy fractions and explicitly named percentage fields.
