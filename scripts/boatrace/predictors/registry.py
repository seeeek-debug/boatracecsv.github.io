"""予想者(predictor)レジストリ。

各予想者は固有 ID (``v1_basic``, ``v2_tenkai`` ...) を持ち、表示名・特徴量
セット (``component_keys``)・出力パス・運用ステータスをここで一元管理する。

新規予想者の追加: 必要なら ``COMPONENT_LABELS_REGISTRY`` に新成分を足し、
``PREDICTORS`` タプルに ``PredictorSpec`` を追加するだけ。
退役: 該当エントリの ``status`` を ``"retired"`` に変更する (過去データは保持)。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


COMPONENT_LABELS_REGISTRY: Mapping[str, str] = {
    "waku":    "枠番pt",
    "racer":   "選手pt",
    "motor":   "モーターpt",
    "exhibit": "展示pt",
    "weather": "気象pt",
    "tenkai":  "展開優位pt",
    "motor2rate": "モーター2連率pt",
    "motor4": "モーターpt",
    "course": "コースpt",
}

COMPONENT_MISSING_FALLBACK: Mapping[str, float] = {
    "racer": 30.0,
}
COMPONENT_MISSING_FALLBACK_DEFAULT: float = 50.0


def component_label(key: str) -> str:
    return COMPONENT_LABELS_REGISTRY[key]


def component_missing_fallback(key: str) -> float:
    return COMPONENT_MISSING_FALLBACK.get(
        key, COMPONENT_MISSING_FALLBACK_DEFAULT,
    )


STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"


@dataclass(frozen=True)
class PredictorSpec:
    predictor_id: str
    display_name: str
    slot: int
    status: str
    started_at: dt.date
    component_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in (STATUS_ACTIVE, STATUS_RETIRED):
            raise ValueError(f"Unknown status {self.status!r} for predictor {self.predictor_id!r}")
        if not self.component_keys:
            raise ValueError(f"predictor {self.predictor_id!r} has no component_keys")
        seen: set[str] = set()
        for key in self.component_keys:
            if key not in COMPONENT_LABELS_REGISTRY:
                raise ValueError(f"Unknown component key {key!r} in predictor {self.predictor_id!r}.")
            if key in seen:
                raise ValueError(f"Duplicate component key {key!r} in predictor {self.predictor_id!r}")
            seen.add(key)

    def index_dir(self, repo: Path) -> Path:
        return repo / "data" / "estimate" / self.predictor_id

    def index_csv_path(self, repo: Path, day: dt.date) -> Path:
        return self.index_dir(repo) / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.csv"

    def weights_dir(self, repo: Path) -> Path:
        return repo / "data" / "estimate" / "stadium" / "weights" / self.predictor_id

    def weights_csv_path(self, repo: Path, target_month: dt.date) -> Path:
        return self.weights_dir(repo) / f"{target_month:%Y-%m}.csv"

    def resolve_weights_csv_path(self, repo: Path, day: dt.date) -> Path | None:
        weights_dir = self.weights_dir(repo)
        if not weights_dir.exists():
            return None
        target_tag = f"{day:%Y-%m}"
        candidates = [
            p for p in sorted(weights_dir.glob("????-??.csv"))
            if p.stem <= target_tag
        ]
        return candidates[-1] if candidates else None

    def component_labels(self) -> dict[str, str]:
        return {k: component_label(k) for k in self.component_keys}

    def is_active(self) -> bool:
        return self.status == STATUS_ACTIVE


PREDICTORS: tuple[PredictorSpec, ...] = (
    PredictorSpec(
        predictor_id="v1_basic",
        display_name="A君予想",
        slot=1,
        status=STATUS_ACTIVE,
        started_at=dt.date(2026, 5, 1),
        component_keys=("waku", "racer", "motor", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v2_tenkai",
        display_name="B君予想",
        slot=2,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 6, 13),
        component_keys=("waku", "racer", "motor2rate", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v3_tenkai",
        display_name="展開予想",
        slot=3,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 6, 20),
        component_keys=("waku", "racer", "motor", "exhibit", "weather", "tenkai"),
    ),
    PredictorSpec(
        predictor_id="v4_motor",
        display_name="モーター予想",
        slot=4,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 7, 20),
        component_keys=("waku", "racer", "motor4", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v5_slit",
        display_name="スリット予想",
        slot=5,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 7, 21),
        component_keys=("waku", "racer", "motor", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v6_course",
        display_name="コース予想",
        slot=6,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 7, 22),
        component_keys=("course", "racer", "motor", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v7_aggregate",
        display_name="統合予想",
        slot=7,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 7, 23),
        component_keys=("course", "racer", "motor4", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v8_aionly",
        display_name="AI予想",
        slot=8,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 7, 28),
        component_keys=("course", "racer", "motor4", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v9_suji",
        display_name="スジ予想",
        slot=9,
        status=STATUS_RETIRED,
        started_at=dt.date(2026, 8, 12),
        component_keys=("waku", "racer", "motor", "exhibit", "weather"),
    ),
    PredictorSpec(
        predictor_id="v10_kimarite",
        display_name="穴予想",
        slot=10,
        status=STATUS_ACTIVE,
        started_at=dt.date(2026, 8, 13),
        component_keys=("waku", "racer", "motor", "exhibit", "weather"),
    ),
    # ▼ 今回追加したハイリターン穴特化モデル
    PredictorSpec(
        predictor_id="v12_longshot_skew",
        display_name="ハイリターン穴予想",
        slot=12,
        status=STATUS_ACTIVE,
        started_at=dt.date(2026, 9, 6),
        component_keys=("waku", "racer", "motor", "exhibit", "weather"),
    ),
)


def all_predictors() -> tuple[PredictorSpec, ...]:
    return PREDICTORS


def active_predictors() -> tuple[PredictorSpec, ...]:
    actives = [p for p in PREDICTORS if p.is_active()]
    return tuple(sorted(actives, key=lambda p: p.slot))


def predictor_by_id(predictor_id: str) -> PredictorSpec:
    for p in PREDICTORS:
        if p.predictor_id == predictor_id:
            return p
    raise KeyError(f"Unknown predictor_id: {predictor_id!r}")

