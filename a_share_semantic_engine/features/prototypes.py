from __future__ import annotations

import numpy as np
import pandas as pd

from .common import l2_normalize


def build_prototypes(features: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    protos = []
    for cid in sorted(np.unique(cluster_ids)):
        center = features[cluster_ids == cid].mean(axis=0)
        protos.append(center)
    protos = np.asarray(protos, dtype=np.float32)
    return l2_normalize(protos)


def compute_prototype_exposure(features: np.ndarray, prototypes: np.ndarray, record_ids: list[str]) -> pd.DataFrame:
    sims = features @ prototypes.T
    cols = [f"proto_{i}" for i in range(sims.shape[1])]
    out = pd.DataFrame(sims, columns=cols)
    out.insert(0, "record_id", record_ids)
    out["top_proto_id"] = np.argmax(sims, axis=1).astype(int)
    out["top_proto_score"] = np.max(sims, axis=1).astype(float)
    return out
