from __future__ import annotations

import random
from typing import Literal

from datasets import load_dataset

from .types_enron import SyntheticQuery

HF_REPO_ID = "corbt/enron_emails_sample_questions"
BAD_QUERIES = {49, 101, 129, 171, 208, 266, 327}


def load_synthetic_queries(
    split: Literal["train", "test"] = "train",
    limit: int | None = None,
    max_messages: int | None = 1,
    shuffle: bool = False,
    exclude_known_bad_queries: bool = True,
) -> list[SyntheticQuery]:
    dataset = load_dataset(HF_REPO_ID, split=split)

    queries: list[SyntheticQuery] = []
    for row in dataset:
        query = SyntheticQuery(**dict(row))
        if max_messages is not None and len(query.message_ids) > max_messages:
            continue
        if exclude_known_bad_queries and query.id in BAD_QUERIES:
            continue
        queries.append(query)

    if shuffle:
        random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    return queries
