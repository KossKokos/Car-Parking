from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from playwright.sync_api import Page, sync_playwright


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class GenericCatalogueScrapeConfig:
    source_name: str
    start_url: str
    project_root: Path

    image_selector: str
    next_button_selector: str | None = None

    image_url_attrs: tuple[str, ...] = ("currentSrc", "src", "data-src")
    allowed_image_domain: str | None = None
    required_keywords: tuple[str, ...] = ()
    keyword_fields: tuple[str, ...] = ("alt", "src")

    pagination_mode: str = "click"  # click, url_attr, none
    next_url_attr: str | None = None

    max_images: int = 1000
    max_pages: int = 250
    headless: bool = False
    page_wait_ms: int = 1200
    request_timeout_seconds: float = 20.0

    scroll_step_px: int = 2500
    max_no_new_scroll_batches: int = 4

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


@dataclass
class ScrapedImageRecord:
    source_name: str
    source_page_url: str
    image_url: str
    local_path: str
    filename: str
    alt_text: str | None
    page_number: int
    downloaded_at: str
    status: str

    image_width: int | None = None
    image_height: int | None = None
    selector_used: str | None = None
    element_index: int | None = None
    error: str | None = None


def parse_csv_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()

    return tuple(item.strip() for item in value.split(",") if item.strip())


def make_safe_filename(image_url: str, content_type: str | None = None) -> str:
    """Create a stable image filename from its URL and detected content type."""
    url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urlparse(image_url).path).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS:
        guessed = mimetypes.guess_extension(content_type or "")
        suffix = guessed if guessed in IMAGE_EXTENSIONS else ".jpg"

    return f"{url_hash}{suffix}"


def load_existing_urls(csv_path: Path) -> set[str]:
    """Load previously scraped image URLs so repeated runs can skip them."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "image_url" not in reader.fieldnames:
            return set()

        return {row["image_url"] for row in reader if row.get("image_url")}


def append_records_to_csv(records: list[ScrapedImageRecord], csv_path: Path) -> None:
    """Append scrape records while preserving existing metadata columns."""
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = [asdict(record) for record in records]
    new_fieldnames = list(new_rows[0].keys())

    file_missing_or_empty = not csv_path.exists() or csv_path.stat().st_size == 0

    if file_missing_or_empty:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    merged_fieldnames = list(dict.fromkeys(existing_fieldnames + new_fieldnames))

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)


def append_records_to_json(records: list[ScrapedImageRecord], json_path: Path) -> None:
    """Append scrape records to the JSON metadata log."""
    if not records:
        return

    json_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []

    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    existing.extend(asdict(record) for record in records)

    json_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def flush_records(records: list[ScrapedImageRecord], config: GenericCatalogueScrapeConfig) -> None:
    """Persist buffered scrape records to both metadata stores and clear them."""
    append_records_to_csv(records, config.metadata_csv_path)
    append_records_to_json(records, config.metadata_json_path)
    records.clear()


def get_attr_expr(attr_name: str) -> str:
    if attr_name == "currentSrc":
        return "img.currentSrc"
    if attr_name == "src":
        return 'img.getAttribute("src")'
    return f'img.getAttribute("{attr_name}")'


def extract_image_candidates(page: Page, config: GenericCatalogueScrapeConfig) -> list[dict]:
    """Collect unique image candidates from the current browser page."""
    attr_expressions = [
        f'"{attr}": {get_attr_expr(attr)}'
        for attr in config.image_url_attrs
    ]

    attrs_js = ",\n".join(attr_expressions)

    js = f"""
        () => {{
            const images = Array.from(document.querySelectorAll({json.dumps(config.image_selector)}));

            return images.map((img, index) => {{
                const rect = img.getBoundingClientRect();

                return {{
                    elementIndex: index,
                    alt: img.getAttribute("alt"),
                    width: img.naturalWidth || Math.round(rect.width) || null,
                    height: img.naturalHeight || Math.round(rect.height) || null,
                    className: img.getAttribute("class"),
                    {attrs_js}
                }};
            }});
        }}
    """

    raw_candidates = page.evaluate(js)

    results: list[dict] = []
    seen_urls: set[str] = set()

    for item in raw_candidates:
        raw_url = None

        for attr in config.image_url_attrs:
            value = item.get(attr)
            if value:
                raw_url = value
                break

        if not raw_url:
            continue

        if str(raw_url).startswith("data:"):
            continue

        image_url = urljoin(page.url, html.unescape(str(raw_url)).strip())
        parsed = urlparse(image_url)

        if config.allowed_image_domain and config.allowed_image_domain not in parsed.netloc:
            continue

        if image_url in seen_urls:
            continue

        if config.required_keywords:
            searchable_parts = []

            for field in config.keyword_fields:
                if field == "alt":
                    searchable_parts.append(str(item.get("alt") or ""))
                elif field == "src":
                    searchable_parts.append(image_url)
                elif field == "class":
                    searchable_parts.append(str(item.get("className") or ""))

            searchable_text = " ".join(searchable_parts).lower()

            if not any(keyword.lower() in searchable_text for keyword in config.required_keywords):
                continue

        seen_urls.add(image_url)

        results.append(
            {
                "image_url": image_url,
                "alt_text": item.get("alt"),
                "width": item.get("width"),
                "height": item.get("height"),
                "element_index": item.get("elementIndex"),
            }
        )

    return results


def dismiss_cookie_banner(page: Page) -> None:
    """
    Dismiss common cookie banners.

    This handles Cookie Control / Civic-style banners where:
        #ccc-overlay
    blocks clicks on page controls.
    """
    cookie_button_selectors = [
        "#ccc-notify-accept",
        "#ccc-recommended-settings",
        "#ccc-dismiss-button",
        "button:has-text('Accept all')",
        "button:has-text('Accept All')",
        "button:has-text('Accept')",
        "button:has-text('Allow all')",
        "button:has-text('Allow All')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('Continue')",
    ]

    for selector in cookie_button_selectors:
        try:
            button = page.locator(selector).first

            if button.count() > 0 and button.is_visible(timeout=1000):
                button.click(timeout=3000)
                page.wait_for_timeout(500)
                print(f"Dismissed cookie banner using selector: {selector}")
                return

        except Exception:
            continue

    # Fallback: remove blocking overlay manually.
    # Useful for scraping your own site where the overlay blocks pagination.
    try:
        removed = page.evaluate(
            """
            () => {
                let removedCount = 0;

                const selectors = [
                    "#ccc-overlay",
                    "#ccc",
                    ".ccc-overlay",
                    "[aria-label='Cookie preferences']"
                ];

                for (const selector of selectors) {
                    const elements = Array.from(document.querySelectorAll(selector));

                    for (const element of elements) {
                        element.remove();
                        removedCount += 1;
                    }
                }

                document.body.style.overflow = "auto";
                document.documentElement.style.overflow = "auto";

                return removedCount;
            }
            """
        )

        if removed:
            print(f"Removed cookie overlay elements: {removed}")

    except Exception as exc:
        print(f"Cookie banner removal failed: {exc}")


def open_page(page: Page, url: str, wait_selector: str, page_wait_ms: int) -> None:
    """Open a catalogue page and wait long enough for image elements to load."""
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)

    dismiss_cookie_banner(page)

    try:
        page.wait_for_selector(wait_selector, timeout=20_000)
    except Exception:
        print(f"Selector not found quickly: {wait_selector}. Continuing anyway.")

    page.wait_for_timeout(page_wait_ms)


def get_next_url_from_attr(page: Page, selector: str, attr_name: str) -> str | None:
    """Resolve the next-page URL from a selected element attribute."""
    element = page.query_selector(selector)

    if element is None:
        return None

    value = element.get_attribute(attr_name)

    if not value:
        return None

    return urljoin(page.url, html.unescape(value))


def click_next_button(page: Page, selector: str, page_wait_ms: int, wait_selector: str) -> bool:
    """Click a pagination control, returning whether navigation was attempted."""
    locator = page.locator(selector)

    if locator.count() == 0:
        return False

    button = locator.first
    class_name = button.get_attribute("class") or ""
    disabled = button.get_attribute("disabled")
    aria_disabled = button.get_attribute("aria-disabled")

    if disabled is not None or aria_disabled == "true" or "disabled" in class_name.lower():
        return False

    try:
        dismiss_cookie_banner(page)

        button.scroll_into_view_if_needed(timeout=5000)

        try:
            button.click(timeout=10_000)
        except Exception as exc:
            print(f"Normal next click failed, trying forced click: {exc}")
            button.click(timeout=10_000, force=True)

        page.wait_for_timeout(page_wait_ms)

        return True

    except Exception as exc:
        print(f"Could not click next button: {exc}")
        return False


def download_image(
    image_url: str,
    output_dir: Path,
    timeout_seconds: float,
    referer: str,
) -> tuple[str, str]:
    """Download one image atomically and return its local path and filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        ),
        "Referer": referer,
    }

    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.get(image_url)
        response.raise_for_status()

        content_type = response.headers.get("content-type")

        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"URL did not return image content: {content_type}")

        filename = make_safe_filename(image_url, content_type)
        local_path = output_dir / filename
        temp_path = local_path.with_suffix(local_path.suffix + ".part")

        if local_path.exists():
            return str(local_path.as_posix()), filename

        temp_path.write_bytes(response.content)
        temp_path.replace(local_path)

        return str(local_path.as_posix()), filename


def scrape_generic_catalogue(config: GenericCatalogueScrapeConfig) -> None:
    """Scrape catalogue images across pages with resume-safe metadata output."""
    config.output_image_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)

    seen_urls = load_existing_urls(config.metadata_csv_path)
    records: list[ScrapedImageRecord] = []
    downloaded_count = 0
    no_new_scroll_batches = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=config.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = browser.new_context(
            viewport={"width": 1400, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            ),
            locale="en-GB",
        )

        page = context.new_page()

        print(f"Opening: {config.start_url}")
        open_page(
            page=page,
            url=config.start_url,
            wait_selector=config.image_selector,
            page_wait_ms=config.page_wait_ms,
        )

        visited_urls: set[str] = set()

        for page_number in range(1, config.max_pages + 1):
            print(f"\nPage loop {page_number} | url: {page.url}")

            candidates = extract_image_candidates(page, config)

            new_candidates = [
                candidate
                for candidate in candidates
                if candidate["image_url"] not in seen_urls
            ]

            print(f"Image candidates: {len(candidates)}")
            print(f"New candidates: {len(new_candidates)}")

            for candidate in new_candidates:
                if downloaded_count >= config.max_images:
                    break

                image_url = candidate["image_url"]
                seen_urls.add(image_url)

                try:
                    local_path, filename = download_image(
                        image_url=image_url,
                        output_dir=config.output_image_dir,
                        timeout_seconds=config.request_timeout_seconds,
                        referer=page.url,
                    )

                    record = ScrapedImageRecord(
                        source_name=config.source_name,
                        source_page_url=page.url,
                        image_url=image_url,
                        local_path=local_path,
                        filename=filename,
                        alt_text=candidate.get("alt_text"),
                        page_number=page_number,
                        downloaded_at=datetime.now(timezone.utc).isoformat(),
                        status="downloaded",
                        image_width=candidate.get("width"),
                        image_height=candidate.get("height"),
                        selector_used=config.image_selector,
                        element_index=candidate.get("element_index"),
                        error=None,
                    )

                    downloaded_count += 1
                    print(f"Downloaded {downloaded_count}: {filename}")

                except Exception as exc:
                    record = ScrapedImageRecord(
                        source_name=config.source_name,
                        source_page_url=page.url,
                        image_url=image_url,
                        local_path="",
                        filename="",
                        alt_text=candidate.get("alt_text"),
                        page_number=page_number,
                        downloaded_at=datetime.now(timezone.utc).isoformat(),
                        status="failed",
                        image_width=candidate.get("width"),
                        image_height=candidate.get("height"),
                        selector_used=config.image_selector,
                        element_index=candidate.get("element_index"),
                        error=str(exc),
                    )

                    print(f"Failed: {image_url} | {exc}")

                records.append(record)

                if len(records) >= 20:
                    flush_records(records, config)

            if downloaded_count >= config.max_images:
                print("Reached max image limit.")
                break
            
            if config.pagination_mode == "scroll":
                if not new_candidates:
                    no_new_scroll_batches += 1
                else:
                    no_new_scroll_batches = 0

                if no_new_scroll_batches >= config.max_no_new_scroll_batches:
                    print(
                        f"No new images for {config.max_no_new_scroll_batches} "
                        "scroll batches. Stopping."
                    )
                    break

                page.mouse.wheel(0, config.scroll_step_px)
                page.wait_for_timeout(config.page_wait_ms)
                continue

            if config.pagination_mode == "none":
                print("Pagination mode is none. Stopping.")
                break

            if config.pagination_mode == "url_attr":
                if not config.next_button_selector or not config.next_url_attr:
                    print("Missing next button selector or next URL attribute. Stopping.")
                    break

                next_url = get_next_url_from_attr(
                    page=page,
                    selector=config.next_button_selector,
                    attr_name=config.next_url_attr,
                )

                if not next_url:
                    print("No next URL found. Stopping.")
                    break

                if next_url in visited_urls:
                    print(f"Already visited next URL. Stopping: {next_url}")
                    break

                visited_urls.add(next_url)

                open_page(
                    page=page,
                    url=next_url,
                    wait_selector=config.image_selector,
                    page_wait_ms=config.page_wait_ms,
                )
                continue

            if config.pagination_mode == "click":
                if not config.next_button_selector:
                    print("Missing next button selector. Stopping.")
                    break

                moved = click_next_button(
                    page=page,
                    selector=config.next_button_selector,
                    page_wait_ms=config.page_wait_ms,
                    wait_selector=config.image_selector,
                )

                if not moved:
                    print("No next page available. Stopping.")
                    break

                continue

            raise ValueError(f"Unsupported pagination mode: {config.pagination_mode}")

        context.close()
        browser.close()

    flush_records(records, config)

    print("\nDone.")
    print(f"Downloaded: {downloaded_count}")
    print(f"Images saved to: {config.output_image_dir}")
    print(f"Metadata CSV: {config.metadata_csv_path}")
    print(f"Metadata JSON: {config.metadata_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-url", default=None)
    parser.add_argument("--start-url-file", default=None)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--project-root", default=".")

    parser.add_argument("--image-selector", required=True)
    parser.add_argument("--image-url-attrs", default="currentSrc,src,data-src")
    parser.add_argument("--allowed-image-domain", default=None)
    parser.add_argument("--required-keywords", default="")
    parser.add_argument("--keyword-fields", default="alt,src")

    parser.add_argument(
        "--pagination-mode",
        default="click",
        choices=["click", "url_attr", "scroll", "none"],
    )
    parser.add_argument("--scroll-step-px", type=int, default=2500)
    parser.add_argument("--max-no-new-scroll-batches", type=int, default=4)
    
    parser.add_argument("--next-button-selector", default=None)
    parser.add_argument("--next-url-attr", default=None)

    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=250)
    parser.add_argument("--page-wait-ms", type=int, default=1200)
    parser.add_argument("--headless", action="store_true")

    return parser.parse_args()


def resolve_start_url(args: argparse.Namespace) -> str:
    """Choose the explicit start URL or derive it from known source presets."""
    if args.start_url:
        return args.start_url

    if args.start_url_file:
        url_path = Path(args.start_url_file)

        if not url_path.exists():
            raise FileNotFoundError(f"Start URL file does not exist: {url_path}")

        url = url_path.read_text(encoding="utf-8").strip()

        if not url:
            raise ValueError(f"Start URL file is empty: {url_path}")

        return url

    raise ValueError("You must provide either --start-url or --start-url-file.")


def main() -> None:
    args = parse_args()

    config = GenericCatalogueScrapeConfig(
        source_name=args.source_name,
        start_url=resolve_start_url(args),
        project_root=Path(args.project_root).resolve(),
        image_selector=args.image_selector,
        image_url_attrs=parse_csv_arg(args.image_url_attrs),
        allowed_image_domain=args.allowed_image_domain,
        required_keywords=parse_csv_arg(args.required_keywords),
        keyword_fields=parse_csv_arg(args.keyword_fields),
        pagination_mode=args.pagination_mode,
        next_button_selector=args.next_button_selector,
        next_url_attr=args.next_url_attr,
        max_images=args.max_images,
        max_pages=args.max_pages,
        page_wait_ms=args.page_wait_ms,
        headless=args.headless,
        scroll_step_px=args.scroll_step_px,
        max_no_new_scroll_batches=args.max_no_new_scroll_batches,
    )

    scrape_generic_catalogue(config)


if __name__ == "__main__":
    main()
