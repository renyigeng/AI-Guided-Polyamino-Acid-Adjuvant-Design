# Staging status

This directory is the cleaned staging area for the competition submission.

## Frozen scientific conclusions

Stage 1:
- K/R/H literature-grounded pairwise comparison
- target-specific K/R tie-break
- D/L chirality comparison
- selected functional block: D-Lys

Stage 2:
- D1 aromatic domain: phenylalanine
- D2 aliphatic hydrophobic/conformational domain: leucine
- D3 polar/hydrogen-bond domain: glutamine

## Directory roles

- data/: compact inputs actually used by the reproducible pipeline
- provenance/: raw API / LLM prompts and responses kept for audit
- archive/: historical development code, old tests and full runs
- src/: reserved for the final clean implementation
- notebooks/: reserved for the final complete executable notebook
- results/: final competition result files
- docs/: final README support documents / method cards

No scientific logic should be modified during code cleanup without an explicit review.
