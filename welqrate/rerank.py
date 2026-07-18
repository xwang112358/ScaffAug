"""Reranking module (ScaffAug's third module).

Maximal Marginal Relevance (MMR) reranking of a virtual-screening prediction list to
increase scaffold diversity among the top candidates while preserving early enrichment,
plus the SD_100 scaffold-diversity metric.

MMR (paper Alg. 2): starting from the highest-scored candidate, iteratively pick

    c* = argmax_i  [ lambda * sigma(p_i) - (1 - lambda) * max_{r in R} Sim(c_i, r) ]

where Sim is Tanimoto similarity of Morgan fingerprints and lambda in [0, 1] trades off
predicted activity (lambda -> 1: pure score order) against diversity (lambda -> 0).

Requires rdkit.
"""
from __future__ import annotations
import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def morgan_fingerprints(smiles, radius=2, n_bits=2048):
    """Morgan (ECFP-like) bit-vector fingerprints; None for unparseable SMILES."""
    if not _HAS_RDKIT:
        raise ImportError("rdkit is required for reranking")
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) if m else None)
    return fps


def mmr_rerank(scores, smiles=None, fingerprints=None, lam=0.7, top_k=None):
    """Return reranked indices (into the candidate list) via MMR.

    scores       : predicted activity scores/logits for the candidate set (array-like)
    smiles       : candidate SMILES (used to build fingerprints if `fingerprints` is None)
    fingerprints : precomputed Morgan fingerprints (optional; overrides `smiles`)
    lam          : trade-off in [0, 1]; 1.0 reproduces pure score ranking
    top_k        : rerank only until top_k are selected (default: all)
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    top_k = n if top_k is None else min(top_k, n)
    if fingerprints is None:
        fingerprints = morgan_fingerprints(smiles)
    sig = _sigmoid(scores)

    # first selection: highest predicted score
    order = np.argsort(-scores)
    first = int(order[0])
    selected = [first]
    remaining = [i for i in range(n) if i != first]
    max_sim = np.zeros(n)  # running max similarity to the selected set
    for i in remaining:
        max_sim[i] = _tanimoto(fingerprints[i], fingerprints[first])

    while remaining and len(selected) < top_k:
        best_i, best_val = None, -np.inf
        for i in remaining:
            val = lam * sig[i] - (1.0 - lam) * max_sim[i]
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)
        # update running max similarity against the newly selected molecule
        for i in remaining:
            s = _tanimoto(fingerprints[i], fingerprints[best_i])
            if s > max_sim[i]:
                max_sim[i] = s
    return selected


def _tanimoto(fp_a, fp_b):
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def scaffold_diversity(smiles, k=100):
    """SD_k: number of unique Bemis-Murcko scaffolds among the first k molecules."""
    if not _HAS_RDKIT:
        raise ImportError("rdkit is required for scaffold_diversity")
    scaffolds = set()
    for s in list(smiles)[:k]:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        try:
            scaffolds.add(MurckoScaffoldSmiles(mol=m))
        except Exception:
            continue
    return len(scaffolds)
