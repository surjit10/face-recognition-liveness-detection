# Notebooks

Notebooks are thin Kaggle orchestration layers. Reusable data, training, evaluation and export logic belongs in `src/faceguard/`. A notebook must save its resolved configuration, environment report and artifact hashes; interactive state is not evidence.

- `02_kaggle_frad_embeddings.ipynb`: private, permission-gated FRAD audit and ArcFace embedding extraction. See `docs/KAGGLE_EMBEDDING_RUNBOOK.md`.
