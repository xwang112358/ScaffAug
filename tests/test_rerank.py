"""Unit tests for the MMR reranking module (needs rdkit)."""
import numpy as np
import pytest

pytest.importorskip("rdkit")
from welqrate.rerank import mmr_rerank, scaffold_diversity, morgan_fingerprints

# A small candidate set: 3 near-duplicate high-scorers + diverse lower-scorers.
SMILES = [
    "c1ccccc1",            # benzene           (highest score)
    "Cc1ccccc1",           # toluene           (very similar to #0)
    "CCc1ccccc1",          # ethylbenzene      (very similar)
    "C1CCNCC1",            # piperidine        (different scaffold)
    "c1ccncc1",            # pyridine          (different scaffold)
    "C1CCOC1",             # THF               (different scaffold)
]
SCORES = np.array([0.95, 0.90, 0.88, 0.60, 0.55, 0.50])


def test_lambda_one_is_pure_score_order():
    order = mmr_rerank(SCORES, smiles=SMILES, lam=1.0)
    assert order == list(np.argsort(-SCORES))


def test_first_pick_is_top_scorer_for_any_lambda():
    for lam in (0.0, 0.3, 0.7, 1.0):
        assert mmr_rerank(SCORES, smiles=SMILES, lam=lam)[0] == 0


def test_diversity_lambda_promotes_dissimilar_early():
    # With strong diversity weighting, a dissimilar scaffold should be pulled ahead of
    # the near-duplicate #1/#2 that pure-score ranking would place 2nd/3rd.
    div = mmr_rerank(SCORES, smiles=SMILES, lam=0.2)
    pure = list(np.argsort(-SCORES))
    assert div[1] != pure[1]                      # reranking changed the 2nd slot
    assert div[1] in (3, 4, 5)                     # a diverse scaffold got promoted


def test_scaffold_diversity_counts_unique_scaffolds():
    # 3 aromatic-carbocycle-ish + piperidine + pyridine + THF -> several distinct BM scaffolds
    sd_all = scaffold_diversity(SMILES, k=6)
    sd_dupes = scaffold_diversity(["c1ccccc1", "c1ccccc1", "c1ccccc1"], k=3)
    assert sd_all >= 3
    assert sd_dupes == 1                            # identical molecules -> one scaffold


def test_rerank_is_a_permutation():
    order = mmr_rerank(SCORES, smiles=SMILES, lam=0.5)
    assert sorted(order) == list(range(len(SCORES)))


def test_precomputed_fingerprints_path():
    fps = morgan_fingerprints(SMILES)
    order = mmr_rerank(SCORES, fingerprints=fps, lam=0.5)
    assert sorted(order) == list(range(len(SCORES)))
