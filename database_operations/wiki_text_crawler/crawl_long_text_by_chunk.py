# -*- coding: utf-8 -*-
"""
Chunk-driven Wikipedia long-text crawler with checkpoint resume.

Input:
  D:\个人项目2\database_operations\full_graph_export\chunks\chunk_*\nodes.json
Output:
  D:\个人项目2\database_operations\full_graph_export\text_chunks\chunk_xxx.json
  D:\个人项目2\database_operations\full_graph_export\text_chunks\manifest.json
  D:\个人项目2\database_operations\full_graph_export\text_chunks\_checkpoint\chunk_xxx.jsonl
  D:\个人项目2\database_operations\full_graph_export\text_chunks\_checkpoint\chunk_xxx.done.txt
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


WIKI_HEADERS = {
    "User-Agent": (
        "PersonalResearchCrawler/1.0 "
        "(Wikidata-Wikipedia long text extraction; contact: example@example.com)"
    ),
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
}

DEFAULT_SECTION_BLACKLIST = {
    "references",
    "notes",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "sources",
    "citations",
}

SCHEMA_VERSION = 1
CHUNK_NAME_PATTERN = re.compile(r"^chunk_(\d+)_")
FETCH_STATUS_OK = "ok"
FETCH_STATUS_NO_SITELINK = "no_sitelink"
FETCH_STATUS_ERROR = "error"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data: Any, indent: Optional[int] = 2) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        if indent is None:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, handle, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def _clean_text(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_bad_section_title(title: str, blacklist: set) -> bool:
    t = re.sub(r"\s+", " ", title.strip().lower())
    return t in blacklist


def _pick_statement(statements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not statements:
        return None
    for statement in statements:
        if statement.get("rank") == "preferred":
            return statement
    return statements[0]


def _get_time_value(entity: Dict[str, Any], prop: str) -> Optional[Tuple[str, int]]:
    statements = entity.get("claims", {}).get(prop, [])
    statement = _pick_statement(statements)
    if not statement:
        return None
    data_value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
    if not isinstance(data_value, dict):
        return None
    time_value = data_value.get("time")
    precision = data_value.get("precision")
    if isinstance(time_value, str) and isinstance(precision, int):
        return time_value, precision
    return None


def _format_wikidata_time(time_str: str, precision: int) -> str:
    match = re.match(r"^([+-])(\d{4})-(\d{2})-(\d{2})T", time_str)
    if not match:
        return time_str

    sign, year_text, month, day = match.group(1), match.group(2), match.group(3), match.group(4)
    year = int(year_text)
    if sign == "-":
        year = -year

    if precision >= 11:
        return f"{abs(year):04d}-{month}-{day}" if year >= 0 else f"-{abs(year):04d}-{month}-{day}"
    if precision == 10:
        return f"{abs(year):04d}-{month}" if year >= 0 else f"-{abs(year):04d}-{month}"
    if precision == 9:
        return str(year)
    if precision == 8:
        return f"circa {year}s"
    if precision == 7:
        return f"circa AD {year}" if year >= 0 else f"circa {abs(year)} BC"
    return str(year)


def _get_coord(entity: Dict[str, Any], prop: str = "P625") -> Optional[Tuple[float, float]]:
    statements = entity.get("claims", {}).get(prop, [])
    statement = _pick_statement(statements)
    if not statement:
        return None
    data_value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
    if not isinstance(data_value, dict):
        return None
    latitude = data_value.get("latitude")
    longitude = data_value.get("longitude")
    if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
        return float(latitude), float(longitude)
    return None


def _get_qid_value(entity: Dict[str, Any], prop: str) -> Optional[str]:
    statements = entity.get("claims", {}).get(prop, [])
    statement = _pick_statement(statements)
    if not statement:
        return None
    data_value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
    if not isinstance(data_value, dict):
        return None
    qid_value = data_value.get("id")
    return qid_value if isinstance(qid_value, str) else None


def _get_string_value(entity: Dict[str, Any], prop: str) -> Optional[str]:
    statements = entity.get("claims", {}).get(prop, [])
    statement = _pick_statement(statements)
    if not statement:
        return None
    data_value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
    return data_value if isinstance(data_value, str) else None


def parse_wikipedia_article(html: str, section_blacklist: Optional[set] = None) -> Dict[str, Any]:
    if section_blacklist is None:
        section_blacklist = DEFAULT_SECTION_BLACKLIST

    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", id="mw-content-text")
    if not content:
        return {"brief": "", "sections": {}}

    root = content.find("div", class_="mw-parser-output") or content

    brief_parts: List[str] = []
    for element in root.children:
        if getattr(element, "name", None) is None:
            continue
        if element.name in ("h2", "h3", "h4"):
            break
        if element.name == "p":
            text = _clean_text(element.get_text(" ", strip=True))
            if len(text) >= 40:
                brief_parts.append(text)
    brief = " ".join(brief_parts).strip()

    sections: Dict[str, str] = {}

    def heading_title(heading: Any) -> str:
        span = heading.find("span", class_="mw-headline")
        title = span.get_text(" ", strip=True) if span else heading.get_text(" ", strip=True)
        return _clean_text(title)

    current_title: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_title, buffer
        if not current_title:
            buffer = []
            return
        text = _clean_text(" ".join(buffer))
        if text and not _is_bad_section_title(current_title, section_blacklist):
            sections[current_title] = text
        buffer = []

    for element in root.descendants:
        if getattr(element, "name", None) is None:
            continue

        if element.name in ("h2", "h3", "h4"):
            flush()
            current_title = heading_title(element)
            continue

        if current_title:
            if element.name == "p":
                text = _clean_text(element.get_text(" ", strip=True))
                if len(text) >= 30:
                    buffer.append(text)
            elif element.name in ("ul", "ol"):
                items = [
                    _clean_text(li.get_text(" ", strip=True))
                    for li in element.find_all("li", recursive=False)
                ]
                items = [x for x in items if len(x) >= 20]
                if items:
                    buffer.append(" ".join(items))

    flush()

    if not brief:
        short_desc = root.find("div", class_="shortdescription")
        if short_desc:
            brief = _clean_text(short_desc.get_text(" ", strip=True))

    return {"brief": brief, "sections": sections}


class WikiFetcher:
    def __init__(
        self,
        sleep_seconds: float,
        max_retries: int,
        base_backoff: float,
        max_backoff: float,
        timeout: int,
    ) -> None:
        self.sleep_seconds = max(0.0, sleep_seconds)
        self.max_retries = max(1, max_retries)
        self.base_backoff = max(1.01, base_backoff)
        self.max_backoff = max(1.0, max_backoff)
        self.timeout = timeout
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(WIKI_HEADERS)
            self._local.session = session
        return session

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.max_backoff, (self.base_backoff ** attempt) + random.uniform(0, 1.0))
        time.sleep(delay)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        last_text_snippet: str = ""
        session = self._get_session()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = session.request(method, url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response

                last_status = response.status_code
                last_text_snippet = (response.text or "").strip().replace("\n", " ")[:180]

                if (
                    response.status_code == 403
                    and "Please set a user-agent" in (response.text or "")
                ):
                    raise RuntimeError(
                        "HTTP 403 from Wikimedia API: User-Agent is rejected by robot policy. "
                        "Set a descriptive User-Agent with contact info."
                    )

                if response.status_code in (403, 429) or (500 <= response.status_code < 600):
                    self._sleep_backoff(attempt)
                    continue

                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                self._sleep_backoff(attempt)

        if last_exc:
            raise last_exc
        raise RuntimeError(
            f"request failed: {url} | last_status={last_status} | body={last_text_snippet}"
        )

    def _fetch_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_with_retry("GET", url, params=params).json()

    def _fetch_html(self, url: str) -> str:
        return self._request_with_retry("GET", url).text

    def _fetch_entitydata(self, qid: str) -> Dict[str, Any]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        data = self._request_with_retry("GET", url).json()
        entity = data.get("entities", {}).get(qid)
        if not isinstance(entity, dict):
            raise ValueError(f"Entity not found: {qid}")
        return entity

    def _fetch_labels(self, qids: List[str], lang: str = "en") -> Dict[str, str]:
        if not qids:
            return {}
        api = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "labels",
            "languages": lang,
            "format": "json",
        }
        data = self._fetch_json(api, params)
        labels: Dict[str, str] = {}
        entities = data.get("entities", {})
        if isinstance(entities, dict):
            for qid, entity in entities.items():
                label = entity.get("labels", {}).get(lang, {}).get("value")
                if isinstance(label, str):
                    labels[qid] = label
        return labels

    def qid_to_fallback_brief(self, qid: str) -> str:
        entity = self._fetch_entitydata(qid)

        label_en = entity.get("labels", {}).get("en", {}).get("value") or qid
        desc_en = entity.get("descriptions", {}).get("en", {}).get("value")
        if not isinstance(label_en, str):
            label_en = qid

        created = _get_time_value(entity, "P571")
        discovered = _get_time_value(entity, "P575")
        coord = _get_coord(entity, "P625")

        country_qid = _get_qid_value(entity, "P17")
        collection_qid = _get_qid_value(entity, "P195")
        inventory_no = _get_string_value(entity, "P217")

        need_labels = [x for x in [country_qid, collection_qid] if isinstance(x, str)]
        labels = self._fetch_labels(need_labels, lang="en") if need_labels else {}
        country_name = labels.get(country_qid) if country_qid else None
        collection_name = labels.get(collection_qid) if collection_qid else None

        parts: List[str] = [label_en.strip().rstrip(".") + "."]
        if isinstance(desc_en, str) and desc_en.strip():
            parts.append(desc_en.strip().rstrip(".") + ".")
        if created:
            parts.append(f"Created {_format_wikidata_time(created[0], created[1])}.")
        if discovered:
            parts.append(f"Discovered {_format_wikidata_time(discovered[0], discovered[1])}.")
        if collection_name and inventory_no:
            parts.append(f"Held by {collection_name}, inventory no. {inventory_no}.")
        elif inventory_no:
            parts.append(f"Inventory no. {inventory_no}.")
        if country_name:
            parts.append(f"Country: {country_name}.")
        if coord:
            latitude, longitude = coord
            parts.append(f"Coordinates: {latitude:.6f}, {longitude:.6f}.")

        fallback = " ".join(parts).strip()
        if fallback:
            return fallback
        return f"{qid}. No enwiki sitelink."

    def qid_to_enwiki_url(self, qid: str) -> Optional[str]:
        api = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        }
        data = self._fetch_json(api, params)
        entity = data.get("entities", {}).get(qid, {})
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki")
        if not enwiki:
            return None

        title = enwiki.get("title")
        if not title:
            return None

        return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    def fetch_one(self, wid: str, verbose: bool = False) -> Dict[str, Any]:
        out = {
            "wid": wid,
            "url": "",
            "brief": "",
            "sections": {},
            "_fetch_status": FETCH_STATUS_ERROR,
        }

        try:
            url = self.qid_to_enwiki_url(wid)
            if not url:
                out["brief"] = self.qid_to_fallback_brief(wid)
                out["_fetch_status"] = FETCH_STATUS_NO_SITELINK
                if verbose:
                    print(f"[SKIP] {wid}: no enwiki sitelink | brief_len={len(out['brief'])}")
                return out

            out["url"] = url
            html = self._fetch_html(url)
            parsed = parse_wikipedia_article(html)
            out["brief"] = parsed.get("brief", "") or ""
            out["sections"] = parsed.get("sections", {}) or {}
            out["_fetch_status"] = FETCH_STATUS_OK

            if verbose:
                print(
                    f"[OK] {wid}: sections={len(out['sections'])}, "
                    f"brief_len={len(out['brief'])}"
                )
            return out
        except Exception as exc:
            if verbose:
                print(f"[ERR] {wid}: {repr(exc)}")
            return out
        finally:
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)


def parse_chunk_index(chunk_name: str) -> int:
    match = CHUNK_NAME_PATTERN.match(chunk_name)
    if not match:
        return 10**9
    return int(match.group(1))


def discover_chunk_dirs(chunks_dir: Path) -> List[Path]:
    dirs = [
        p
        for p in chunks_dir.iterdir()
        if p.is_dir() and p.name.startswith("chunk_") and (p / "nodes.json").exists()
    ]
    dirs.sort(key=lambda p: (parse_chunk_index(p.name), p.name))
    return dirs


def load_wids_from_chunk_nodes(nodes_path: Path) -> List[str]:
    with nodes_path.open("r", encoding="utf-8") as handle:
        nodes = json.load(handle)

    wids: List[str] = []

    if isinstance(nodes, dict):
        iterable = nodes.values()
    elif isinstance(nodes, list):
        iterable = nodes
    else:
        raise RuntimeError(f"Unsupported nodes.json structure: {nodes_path}")

    for node in iterable:
        props = (node or {}).get("properties", {})
        wid = props.get("wid") if isinstance(props, dict) else None
        if isinstance(wid, str) and wid.startswith("Q"):
            wids.append(wid)

    seen = set()
    unique_wids: List[str] = []
    for wid in wids:
        if wid not in seen:
            seen.add(wid)
            unique_wids.append(wid)
    return unique_wids


def load_done_wids(done_path: Path) -> set:
    done = set()
    if not done_path.exists():
        return done
    with done_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            wid = line.strip()
            if wid:
                done.add(wid)
    return done


def load_results_jsonl(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return results

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            wid = item.get("wid")
            if isinstance(wid, str):
                results[wid] = item
    return results


def load_results_json_array(json_path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not json_path.exists():
        return results

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return results

    if not isinstance(data, list):
        return results

    for item in data:
        if not isinstance(item, dict):
            continue
        wid = item.get("wid")
        if isinstance(wid, str):
            results[wid] = item
    return results


def append_result_jsonl(handle: Any, item: Dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_done(handle: Any, wid: str) -> None:
    handle.write(wid + "\n")


def load_or_init_manifest(manifest_path: Path, chunk_names: List[str], chunks_dir: Path, out_dir: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "chunks_dir": str(chunks_dir),
            "out_dir": str(out_dir),
            "chunks": {},
        }

    chunks_meta = manifest.setdefault("chunks", {})

    for name in chunk_names:
        chunks_meta.setdefault(
            name,
            {
                "status": "pending",
                "wid_total": 0,
                "done": 0,
                "output_file": f"{name}.json",
                "error": "",
                "updated_at_utc": utc_now_iso(),
            },
        )

    manifest["schema_version"] = SCHEMA_VERSION
    manifest["updated_at_utc"] = utc_now_iso()
    return manifest


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now_iso()
    write_json_atomic(manifest_path, manifest, indent=2)


def empty_record(wid: str) -> Dict[str, Any]:
    return {
        "wid": wid,
        "url": "",
        "brief": "",
        "sections": {},
        "_fetch_status": FETCH_STATUS_ERROR,
    }


def is_terminal_record(item: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(item, dict):
        return False

    status = item.get("_fetch_status")
    if status == FETCH_STATUS_OK:
        return True
    if status == FETCH_STATUS_NO_SITELINK:
        brief = item.get("brief")
        return isinstance(brief, str) and bool(brief.strip())

    if isinstance(item.get("url"), str) and item.get("url"):
        return True

    brief = item.get("brief")
    if isinstance(brief, str) and brief.strip():
        return True

    return False


def as_output_record(item: Optional[Dict[str, Any]], wid: str) -> Dict[str, Any]:
    source = item if isinstance(item, dict) else empty_record(wid)
    return {
        "wid": wid,
        "url": source.get("url", "") or "",
        "brief": source.get("brief", "") or "",
        "sections": source.get("sections", {}) or {},
    }


def process_one_chunk(
    chunk_dir: Path,
    out_dir: Path,
    checkpoint_dir: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
    fetcher: WikiFetcher,
    workers: int,
    verbose: bool,
    wid_limit_per_chunk: int,
    retry_empty_records: bool,
) -> Dict[str, Any]:
    chunk_name = chunk_dir.name
    nodes_path = chunk_dir / "nodes.json"

    output_path = out_dir / f"{chunk_name}.json"
    jsonl_path = checkpoint_dir / f"{chunk_name}.jsonl"
    done_path = checkpoint_dir / f"{chunk_name}.done.txt"

    chunk_meta = manifest["chunks"][chunk_name]

    if output_path.exists() and chunk_meta.get("status") == "completed" and not retry_empty_records:
        return {
            "chunk": chunk_name,
            "status": "skipped",
            "wid_total": int(chunk_meta.get("wid_total", 0)),
            "done": int(chunk_meta.get("done", 0)),
        }

    all_wids = load_wids_from_chunk_nodes(nodes_path)
    if wid_limit_per_chunk and wid_limit_per_chunk > 0:
        wids = all_wids[:wid_limit_per_chunk]
    else:
        wids = all_wids

    results_map_all = load_results_jsonl(jsonl_path)
    if not results_map_all and output_path.exists():
        results_map_all = load_results_json_array(output_path)
    results_map = {wid: results_map_all[wid] for wid in wids if wid in results_map_all}
    done_wids = load_done_wids(done_path)
    done_wids.update(results_map.keys())

    if retry_empty_records:
        pending_wids = [wid for wid in wids if not is_terminal_record(results_map.get(wid))]
    else:
        pending_wids = [wid for wid in wids if wid not in results_map]

    done_count = len(wids) - len(pending_wids)

    chunk_meta["status"] = "in_progress"
    chunk_meta["wid_total"] = len(wids)
    chunk_meta["done"] = done_count
    chunk_meta["error"] = ""
    chunk_meta["output_file"] = output_path.name
    chunk_meta["updated_at_utc"] = utc_now_iso()
    save_manifest(manifest_path, manifest)

    if verbose:
        print(
            f"[CHUNK] {chunk_name}: total={len(wids)}, "
            f"done={done_count}, pending={len(pending_wids)}"
        )

    flush_every = 20
    flush_counter = 0

    if not pending_wids and output_path.exists():
        chunk_meta["status"] = "completed"
        chunk_meta["done"] = len(wids)
        chunk_meta["wid_total"] = len(wids)
        chunk_meta["error"] = ""
        chunk_meta["updated_at_utc"] = utc_now_iso()
        save_manifest(manifest_path, manifest)
        return {
            "chunk": chunk_name,
            "status": "skipped",
            "wid_total": len(wids),
            "done": len(wids),
        }

    with jsonl_path.open("a", encoding="utf-8") as jsonl_handle, done_path.open("a", encoding="utf-8") as done_handle:
        with tqdm(
            total=len(wids),
            initial=done_count,
            desc=chunk_name,
            unit="wid",
            leave=False,
        ) as wid_pbar:
            if pending_wids:
                with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    future_to_wid = {
                        pool.submit(fetcher.fetch_one, wid, verbose): wid for wid in pending_wids
                    }
                    for future in futures.as_completed(future_to_wid):
                        wid = future_to_wid[future]
                        try:
                            item = future.result()
                        except Exception:
                            item = empty_record(wid)

                        results_map[wid] = item
                        append_result_jsonl(jsonl_handle, item)
                        append_done(done_handle, wid)
                        done_wids.add(wid)

                        flush_counter += 1
                        if flush_counter >= flush_every:
                            jsonl_handle.flush()
                            os.fsync(jsonl_handle.fileno())
                            done_handle.flush()
                            os.fsync(done_handle.fileno())
                            flush_counter = 0

                        chunk_meta["done"] = len([x for x in wids if is_terminal_record(results_map.get(x))])
                        chunk_meta["updated_at_utc"] = utc_now_iso()
                        wid_pbar.update(1)

            if flush_counter > 0:
                jsonl_handle.flush()
                os.fsync(jsonl_handle.fileno())
                done_handle.flush()
                os.fsync(done_handle.fileno())

    ordered: List[Dict[str, Any]] = []
    for wid in wids:
        ordered.append(as_output_record(results_map.get(wid), wid))

    write_json_atomic(output_path, ordered, indent=None)

    chunk_meta["status"] = "completed"
    chunk_meta["done"] = len(wids)
    chunk_meta["wid_total"] = len(wids)
    chunk_meta["error"] = ""
    chunk_meta["updated_at_utc"] = utc_now_iso()
    save_manifest(manifest_path, manifest)

    return {
        "chunk": chunk_name,
        "status": "completed",
        "wid_total": len(wids),
        "done": len(wids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks_dir",
        type=str,
        default=r"D:\个人项目2\database_operations\full_graph_export\chunks",
        help="Directory with chunk folders containing nodes.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"D:\个人项目2\database_operations\full_graph_export\text_chunks",
        help="Output directory for per-chunk long text json files",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers per chunk")
    parser.add_argument("--sleep", type=float, default=0.2, help="Per-request base sleep seconds")
    parser.add_argument("--max_retries", type=int, default=6, help="HTTP retry limit")
    parser.add_argument("--chunk_limit", type=int, default=0, help="Only process first N chunks (0=all)")
    parser.add_argument("--wid_limit_per_chunk", type=int, default=0, help="Only crawl first N wids per chunk for smoke test (0=all)")
    parser.add_argument(
        "--no_retry_empty_records",
        action="store_true",
        help="Do not retry old empty records from checkpoint/output",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--base_backoff", type=float, default=1.5, help="Backoff base")
    parser.add_argument("--max_backoff", type=float, default=120.0, help="Backoff max seconds")
    parser.add_argument("--verbose", action="store_true", help="Verbose log")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "_checkpoint"
    manifest_path = out_dir / "manifest.json"

    if not chunks_dir.exists():
        raise RuntimeError(f"chunks_dir does not exist: {chunks_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = discover_chunk_dirs(chunks_dir)
    if args.chunk_limit and args.chunk_limit > 0:
        chunk_dirs = chunk_dirs[: args.chunk_limit]

    if not chunk_dirs:
        print("[INFO] No chunk directories found.")
        return

    chunk_names = [p.name for p in chunk_dirs]
    manifest = load_or_init_manifest(manifest_path, chunk_names, chunks_dir, out_dir)

    fetcher = WikiFetcher(
        sleep_seconds=args.sleep,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
        max_backoff=args.max_backoff,
        timeout=args.timeout,
    )

    completed_initial = 0
    retry_empty_records = not args.no_retry_empty_records
    for chunk in chunk_dirs:
        chunk_meta = manifest["chunks"].setdefault(chunk.name, {})
        output_path = out_dir / f"{chunk.name}.json"
        if (
            chunk_meta.get("status") == "completed"
            and output_path.exists()
            and not retry_empty_records
        ):
            completed_initial += 1

    save_manifest(manifest_path, manifest)

    total = len(chunk_dirs)
    completed_now = completed_initial

    try:
        with tqdm(total=total, initial=completed_initial, desc="chunks", unit="chunk") as chunk_pbar:
            for chunk_dir in chunk_dirs:
                chunk_name = chunk_dir.name
                chunk_meta = manifest["chunks"][chunk_name]
                output_path = out_dir / f"{chunk_name}.json"

                if (
                    chunk_meta.get("status") == "completed"
                    and output_path.exists()
                    and not retry_empty_records
                ):
                    chunk_pbar.update(1)
                    continue

                if chunk_meta.get("status") == "completed" and not output_path.exists():
                    chunk_meta["status"] = "pending"
                    chunk_meta["updated_at_utc"] = utc_now_iso()
                    save_manifest(manifest_path, manifest)

                try:
                    result = process_one_chunk(
                        chunk_dir=chunk_dir,
                        out_dir=out_dir,
                        checkpoint_dir=checkpoint_dir,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        fetcher=fetcher,
                        workers=max(1, args.workers),
                        verbose=args.verbose,
                        wid_limit_per_chunk=max(0, args.wid_limit_per_chunk),
                        retry_empty_records=retry_empty_records,
                    )
                    if result["status"] == "completed":
                        completed_now += 1
                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted by user. Progress is saved.")
                    raise
                except Exception as exc:
                    chunk_meta = manifest["chunks"][chunk_name]
                    chunk_meta["status"] = "failed"
                    chunk_meta["error"] = repr(exc)
                    chunk_meta["updated_at_utc"] = utc_now_iso()
                    save_manifest(manifest_path, manifest)
                    print(f"[ERR] chunk failed: {chunk_name} | {repr(exc)}")

                chunk_pbar.update(1)
    finally:
        save_manifest(manifest_path, manifest)

    print(f"[DONE] chunks_total={total}, chunks_completed={completed_now}, out_dir={out_dir}")


if __name__ == "__main__":
    main()





