# v1 prototype (capstone, July 2026)

- `Review_Synthesizer_Prototype.html`: self contained demo. Open it in a browser. All logic runs client side over 25 synthetic reviews: Jaccard near duplicate detection (threshold 0.55), rating burst detection (3 same rating reviews within 4 days), keyword theme and aspect scoring, and the verified rating.
- `harbor_light_inn_prompt.json`: Google AI Studio export of the LLM system prompt used in Modules 3 to 5, with the anchored trust score rubric and temperature 0 run settings. This is the specification for the v2 synthesis step.

Both files are kept as they were submitted, apart from this note.
