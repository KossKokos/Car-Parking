from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScrapeConfig:
    source_name: str
    start_url: str
    project_root: Path
    max_images: int = 50
    max_scroll_batches: int = 10
    scroll_wait_ms: int = 1500
    request_timeout_seconds: float = 20.0
    headless: bool = False

    @property
    def output_image_dir(self) -> Path:
        return (
            self.project_root
            / "data"
            / "raw"
            / "scraped_full_images"
            / self.source_name
        )

    @property
    def metadata_csv_path(self) -> Path:
        return self.project_root / "data" / "metadata" / "scraped_images.csv"

    @property
    def metadata_json_path(self) -> Path:
        return self.project_root / "data" / "metadata" / "scraped_images.json"