import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from .configs import DataSchema, ArtifactPaths


@dataclass
class FeatureStore:
    schema: DataSchema
    scaler: Optional[StandardScaler] = None
    feature_order: Optional[List[str]] = None

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df_pd = df.to_pandas()
        feature_cols = self._resolve_feature_cols(df_pd)
        self.feature_order = feature_cols
        self.scaler = StandardScaler()
        df_pd[feature_cols] = self.scaler.fit_transform(df_pd[feature_cols].astype(float))
        return pl.from_pandas(df_pd)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.scaler is None or self.feature_order is None:
            raise RuntimeError("FeatureStore not fitted. Call fit_transform first or load artifacts.")
        df_pd = df.to_pandas()
        missing = [c for c in self.feature_order if c not in df_pd.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        df_pd[self.feature_order] = self.scaler.transform(df_pd[self.feature_order].astype(float))
        return pl.from_pandas(df_pd)

    def save(self, paths: ArtifactPaths) -> None:
        Path(paths.models_dir).mkdir(parents=True, exist_ok=True)
        with open(paths.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(paths.features_path, "w", encoding="utf-8") as f:
            json.dump({"feature_order": self.feature_order, "schema": self.schema.__dict__}, f, ensure_ascii=False, indent=2)

    def load(self, paths: ArtifactPaths) -> None:
        with open(paths.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(paths.features_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_order = meta["feature_order"]

    def _resolve_feature_cols(self, df_pd: pd.DataFrame) -> List[str]:
        if self.schema.feature_cols is not None:
            return list(self.schema.feature_cols)
        reserved = {self.schema.unique_id_col, self.schema.timestamp_col, self.schema.target_col}
        return [c for c in df_pd.columns if c not in reserved]


