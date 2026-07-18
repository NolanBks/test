#!/usr/bin/env python3
"""Check predicate schema imports and dimensions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.predicates.schema import PREDICATE_NAMES, predicate_dim, predicate_index, validate_predicate_tensor


@dataclass
class FakeTensor:
    shape: tuple[int, ...]


def main() -> None:
    assert predicate_dim() == 10, predicate_dim()
    assert predicate_index("failure_risk") == PREDICATE_NAMES.index("failure_risk")
    validate_predicate_tensor(FakeTensor((2, predicate_dim())))
    try:
        validate_predicate_tensor(FakeTensor((2, predicate_dim() + 1)))
    except ValueError as exc:
        print(f"Wrong-dimension check raised clearly: {exc}")
    else:
        raise AssertionError("Wrong predicate dimension did not raise.")
    print(f"Predicate schema OK: {predicate_dim()} predicates")


if __name__ == "__main__":
    main()
