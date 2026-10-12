# Functional Block Stage 1 — RAG + Pairwise ICL

## Pipeline
Europe PMC retrieval → MedCPT semantic re-ranking → PaperQA2-style contextual evidence → pairwise ICL synthesis.

- Functional-block identity: INCONCLUSIVE
- Lys chirality: not run
- Selected functional block: INCONCLUSIVE
- Repeats per orientation: 3

## Pairwise results
- K vs R: inconclusive; votes={'K': 3, 'R': 3, 'INSUFFICIENT': 0}
- K vs H: K; votes={'K': 4, 'H': 1, 'INSUFFICIENT': 1}
- R vs H: R; votes={'R': 6, 'H': 0, 'INSUFFICIENT': 0}

## Interpretation boundary
MedCPT scores and RAG relevance scores measure relevance, not immune effect magnitude. The final recommendation is an evidence-synthesis result grounded in retrieved literature, not an experimental measurement.
