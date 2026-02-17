from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator

from backtest_framework.models import Tick


class TickCSVLoader:
    def __init__(self, csv_path: str | Path, timestamp_col: str = "timestamp", price_col: str = "price", volume_col: str = "volume") -> None:
        self.csv_path = Path(csv_path)
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col

    def stream(self) -> Iterator[Tick]:
        with self.csv_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield Tick(
                    timestamp=datetime.fromisoformat(row[self.timestamp_col]),
                    price=float(row[self.price_col]),
                    volume=float(row[self.volume_col]) if row.get(self.volume_col) else 0.0,
                )
