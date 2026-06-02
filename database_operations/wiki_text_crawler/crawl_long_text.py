# -*- coding: utf-8 -*-
"""
批量抓取 Wikipedia 英文页面的结构化文本（brief + 动态 sections）
输入: subgraph_nodes_seed1000_hop1_s42.json (每个 node 的 properties.wid)
输出: 多个 JSON 文件（每 10000 条一个 shard），每个文件是 JSON 列表，每项包含:
  - wid: "Qxxxx"
  - url: enwiki url（可能为空）
  - brief: lead 段（第一个 h2 前）
  - sections: { "Section title": "text...", ... }（动态标题，不写死 Discovery/Description）

断点续跑:
  - out_dir/done_wids.txt 记录已成功抓取的 wid（成功才写入）
  - 重启会自动跳过已完成 wid

依赖:
  pip install requests beautifulsoup4 lxml tqdm
"""

import argparse
import json
import os
import re
import time
import random
from typing import Dict, Any, Optional, List

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# -----------------------------
# 网络与解析配置
# -----------------------------
WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en",
}

DEFAULT_SECTION_BLACKLIST = {
    "references", "notes", "external links", "see also",
    "further reading", "bibliography", "sources", "citations"
}

# 工程参数：重试与退避
MAX_RETRIES = 6
BASE_BACKOFF = 1.5
MAX_BACKOFF = 120.0


# -----------------------------
# 工具函数（你给的解析/清洗逻辑：保持不动）
# -----------------------------
def _clean_text(s: str) -> str:
    s = re.sub(r"\[\d+\]", "", s)      # 去掉 [1][2]
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_bad_section_title(title: str, blacklist: set) -> bool:
    t = re.sub(r"\s+", " ", title.strip().lower())
    return t in blacklist


# -----------------------------
# 网络请求（增强：session + retry/backoff，不改变输出格式）
# -----------------------------
def _sleep_backoff(attempt: int):
    # 指数退避 + 抖动
    t = min(MAX_BACKOFF, (BASE_BACKOFF ** attempt) + random.uniform(0, 1.0))
    time.sleep(t)


def _request_with_retry(session: requests.Session,
                        method: str,
                        url: str,
                        *,
                        params: Optional[Dict[str, Any]] = None,
                        timeout: int = 30) -> requests.Response:
    """
    对 429/5xx/403 做重试退避
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.request(method, url, params=params, timeout=timeout)
            # 200 OK
            if r.status_code == 200:
                return r

            # 可恢复：限流/服务端错误/偶发 403
            if r.status_code in (403, 429) or (500 <= r.status_code < 600):
                _sleep_backoff(attempt)
                continue

            # 其他错误：直接抛
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            _sleep_backoff(attempt)

    # 重试耗尽
    raise last_exc if last_exc else RuntimeError(f"request failed: {url}")


def _fetch_json(session: requests.Session,
                url: str,
                params: Dict[str, Any],
                timeout: int = 30) -> Dict[str, Any]:
    r = _request_with_retry(session, "GET", url, params=params, timeout=timeout)
    return r.json()


def _fetch_html(session: requests.Session,
                url: str,
                timeout: int = 30) -> str:
    r = _request_with_retry(session, "GET", url, params=None, timeout=timeout)
    return r.text


# -----------------------------
# Wikidata: QID -> enwiki URL
#   用 wbgetentities 避免 Special:EntityData 403
# -----------------------------
def qid_to_enwiki_url(session: requests.Session, qid: str) -> Optional[str]:
    """
    返回 enwiki url，例如:
      https://en.wikipedia.org/wiki/Circus_Games_Mosaic
    若没有 enwiki sitelink，返回 None
    """
    api = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "sitelinks",
        "sitefilter": "enwiki",
        "format": "json",
    }
    data = _fetch_json(session, api, params=params)
    ent = data.get("entities", {}).get(qid, {})
    sitelinks = ent.get("sitelinks", {})
    enwiki = sitelinks.get("enwiki")
    if not enwiki:
        return None
    title = enwiki.get("title")
    if not title:
        return None
    title = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{title}"


# -----------------------------
# Wikipedia HTML -> {brief, sections}
#   你给的解析逻辑：保持不动
# -----------------------------
def parse_wikipedia_article(html: str,
                            section_blacklist: Optional[set] = None) -> Dict[str, Any]:
    if section_blacklist is None:
        section_blacklist = DEFAULT_SECTION_BLACKLIST

    soup = BeautifulSoup(html, "lxml")
    content = soup.find("div", id="mw-content-text")
    if not content:
        return {"brief": "", "sections": {}}

    root = content.find("div", class_="mw-parser-output") or content

    # 1) brief：第一个 h2/h3/h4 前的“可读段落”
    brief_parts: List[str] = []
    for el in root.children:
        if getattr(el, "name", None) is None:
            continue
        if el.name in ("h2", "h3", "h4"):
            break
        if el.name == "p":
            txt = _clean_text(el.get_text(" ", strip=True))
            if len(txt) >= 40:
                brief_parts.append(txt)
    brief = " ".join(brief_parts).strip()

    # 2) sections：动态解析 h2/h3/h4 标题 + 正文
    sections: Dict[str, str] = {}

    def heading_title(h) -> str:
        sp = h.find("span", class_="mw-headline")
        title = sp.get_text(" ", strip=True) if sp else h.get_text(" ", strip=True)
        return _clean_text(title)

    current_title: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal current_title, buf
        if not current_title:
            buf = []
            return
        text = _clean_text(" ".join(buf))
        if text and not _is_bad_section_title(current_title, section_blacklist):
            sections[current_title] = text
        buf = []

    # 注意：用 root 的“直接后继链”更稳：遍历元素树，遇到标题切换
    for el in root.descendants:
        if getattr(el, "name", None) is None:
            continue

        if el.name in ("h2", "h3", "h4"):
            flush()
            current_title = heading_title(el)
            continue

        if current_title:
            if el.name == "p":
                txt = _clean_text(el.get_text(" ", strip=True))
                if len(txt) >= 30:
                    buf.append(txt)
            elif el.name in ("ul", "ol"):
                items = [
                    _clean_text(li.get_text(" ", strip=True))
                    for li in el.find_all("li", recursive=False)
                ]
                items = [x for x in items if len(x) >= 20]
                if items:
                    buf.append(" ".join(items))

    flush()

    # 3) 兜底：brief 为空时，尝试 shortdescription
    if not brief:
        sd = root.find("div", class_="shortdescription")
        if sd:
            brief = _clean_text(sd.get_text(" ", strip=True))

    return {"brief": brief, "sections": sections}


# -----------------------------
# 输入文件读取：nodes.json -> wid 列表（保持你原逻辑）
# -----------------------------
def load_wids_from_nodes_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    wids: List[str] = []
    for n in nodes:
        props = (n or {}).get("properties", {})
        wid = props.get("wid")
        if isinstance(wid, str) and wid.startswith("Q"):
            wids.append(wid)

    # 去重但保持顺序
    seen = set()
    uniq = []
    for w in wids:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


# -----------------------------
# 单条抓取：输出格式保持不变
# -----------------------------
def fetch_one(session: requests.Session,
              wid: str,
              sleep: float = 0.3,
              verbose: bool = True) -> Dict[str, Any]:
    out = {"wid": wid, "url": "", "brief": "", "sections": {}}

    try:
        url = qid_to_enwiki_url(session, wid)
        if not url:
            if verbose:
                print(f"[SKIP] {wid}: no enwiki sitelink")
            return out

        out["url"] = url

        html = _fetch_html(session, url)
        parsed = parse_wikipedia_article(html)

        out["brief"] = parsed.get("brief", "") or ""
        out["sections"] = parsed.get("sections", {}) or {}

        if verbose:
            print(f"[OK] {wid}: sections={len(out['sections'])}, brief_len={len(out['brief'])}")

        if sleep > 0:
            time.sleep(sleep)

        return out

    except requests.HTTPError as e:
        if verbose:
            print(f"[HTTPError] {wid}: {e}")
        return out
    except Exception as e:
        if verbose:
            print(f"[Error] {wid}: {repr(e)}")
        return out


# -----------------------------
# 断点续跑：done_wids.txt
# -----------------------------
def load_done_wids(done_path: str) -> set:
    done = set()
    if not os.path.exists(done_path):
        return done
    with open(done_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                done.add(w)
    return done


def append_done(done_f, wid: str):
    done_f.write(wid + "\n")


# -----------------------------
# 分片写 JSON 列表（保持 json.dump(list) 格式）
# -----------------------------
def dump_shard(out_dir: str, shard_idx: int, results: List[Dict[str, Any]]):
    out_path = os.path.join(out_dir, f"enwiki_text_dump_{shard_idx:06d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_nodes", type=str, default="subgraph_nodes_seed1000_hop1_s42.json",
                    help="输入 nodes json，例如 subgraph_nodes_seed1000_hop1_s42.json")
    ap.add_argument("--out_dir", type=str, default="enwiki_dump_out",
                    help="输出目录（会生成多个 shard json + done_wids.txt）")
    ap.add_argument("--limit", type=int, default=20,
                    help="只处理前 N 条（debug 用）。0 表示全量")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="每次请求间隔秒数，避免过快")
    ap.add_argument("--verbose", action="store_true",
                    help="打印详细日志")

    ap.add_argument("--shard_size", type=int, default=10000,
                    help="每多少条写入一个 json 文件（每个文件仍是 JSON 列表）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    done_path = os.path.join(args.out_dir, "done_wids.txt")
    done = load_done_wids(done_path)

    wids = load_wids_from_nodes_json(args.in_nodes)
    if args.limit and args.limit > 0:
        wids = wids[:args.limit]

    if args.verbose:
        print(f"[INFO] total_wids_in_file = {len(wids)}")
        print(f"[INFO] already_done = {len(done)}")
        print(f"[INFO] to_process (after skip done) ~= {max(0, len(wids) - len(done))}")

    # session
    session = requests.Session()
    session.headers.update(WIKI_HEADERS)

    # tqdm：按 wid 数进度（跳过 done 的不计入成功，但会走循环）
    pbar = tqdm(total=len(wids), desc="Process wids", unit="wid")

    shard_idx = 0
    shard_buf: List[Dict[str, Any]] = []

    # done 追加写：抗断
    done_f = open(done_path, "a", encoding="utf-8")

    # 为了减少中途断损失：每 N 条 done/fsync 一次
    flush_every = 50
    flush_cnt = 0

    try:
        for wid in wids:
            pbar.update(1)

            if wid in done:
                continue

            item = fetch_one(session, wid, sleep=args.sleep, verbose=args.verbose)

            # 保持输出格式：wid/url/brief/sections 四字段（永远有）
            shard_buf.append(item)

            # 只有当“真正写入 shard 文件之后”再记 done？
            # 你如果更想“成功抓取就记 done”，也行，但为了避免丢 shard，这里采用：先入 buf，写 shard 后统一记 done
            # ——为了更抗断：我们改成“抓取完成就记 done”，同时 shard 每满 shard_size 落盘
            append_done(done_f, wid)
            done.add(wid)

            flush_cnt += 1
            if flush_cnt >= flush_every:
                done_f.flush()
                os.fsync(done_f.fileno())
                flush_cnt = 0

            # shard 满了就写文件（写的是 JSON 列表，格式不变）
            if len(shard_buf) >= args.shard_size:
                out_path = dump_shard(args.out_dir, shard_idx, shard_buf)
                if args.verbose:
                    print(f"[SHARD] saved -> {out_path} | items={len(shard_buf)}")
                shard_idx += 1
                shard_buf = []

        # 最后残余写入
        if shard_buf:
            out_path = dump_shard(args.out_dir, shard_idx, shard_buf)
            if args.verbose:
                print(f"[SHARD] saved -> {out_path} | items={len(shard_buf)}")
            shard_idx += 1
            shard_buf = []

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user. Progress saved (done_wids + shards).")

        # 中断也把当前 shard 写掉（尽量不丢）
        if shard_buf:
            out_path = dump_shard(args.out_dir, shard_idx, shard_buf)
            print(f"[SHARD] saved -> {out_path} | items={len(shard_buf)}")

    finally:
        pbar.close()
        try:
            done_f.flush()
            os.fsync(done_f.fileno())
        except Exception:
            pass
        done_f.close()
        session.close()

    print(f"[DONE] out_dir={args.out_dir} | done={len(done)} | shards_written={shard_idx}")


if __name__ == "__main__":
    main()
