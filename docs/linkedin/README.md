# LinkedIn explainer

[explainer.html](explainer.html) is a 1080×1080 animated explainer of this
repo in ten scenes (about 73 seconds):

1. Hook: a GraphRAG book taken apart and rebuilt as bricks
2. The book: seven chapters, taught in notebooks that share globals
3. What was adapted: one class per job, and Neo4j swapped for Kuzu
4. to 7. One page each for `VectorRag`, `ParentChildRag`, `AgenticRag` and
   `GraphRag`, built from their real pipeline steps
8. And 4 more: `Text2CypherQA`, `StructuredGraphExtraction`,
   `HypotheticalQuestionRag` and `RagEvaluation`
9. The text-recipe benchmark from
   [examples/README.md](../../examples/README.md#benchmarking-the-text-recipes)
10. Close: restated hook, which recipe to use for which data, and the link

Open it in a browser. Space pauses, and ← and → change scene.

## URL parameters

| Parameter | Effect |
| --- | --- |
| `scene=N` | start at scene N |
| `record=1` | hide the controls and play through once (for screen recording) |
| `still=N` | show scene N's final frame without motion (for screenshots and a PDF carousel) |

The benchmark numbers are hard-coded in the page's `BENCH` array, and the
recipe pages copy their steps from `graphrag.recipes`. Update both if
those change.
