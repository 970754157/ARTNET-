# excute.py
# -*- coding: utf-8 -*-
"""
入口脚本：
- 调用 extract_subgraph(seed_k, hops, seed)
- 将边关系写入 TXT（src_id src_type dst_id dst_type edge_type）
- 将节点完整信息写入 JSON（id, labels, properties）
- 只对 ART 节点，根据 properties.image 下载图片到本地，
  用 wid (QID) 作为文件名保存，并用 tqdm 打印下载进度
"""

import json
import pathlib
from typing import List, Dict, Any

import requests
from tqdm import tqdm

from graphsql import extract_subgraph


def download_images(node_list: List[Dict[str, Any]], seed_k: int, hops: int):
    """
    根据节点列表中的 properties.image 字段下载图片，
    以 properties.wid (QID) 命名文件。
    保存目录：images_seed{seed_k}_hop{hops}/
    同时用 tqdm 显示下载进度。
    """
    # 目标目录
    out_dir = pathlib.Path(f"images_seed{seed_k}_hop{hops}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 伪装成正常浏览器，防止被 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for node in tqdm(node_list, desc="下载图片", unit="img"):
        props = node.get("properties", {})

        # 取 wid（QID），没有就用内部 id 兜底
        wid = props.get("wid")
        if not wid:
            wid = f"node_{node['id']}"

        # 取 image 字段：可能是字符串，也可能是列表
        image_field = props.get("image")
        if not image_field:
            continue

        if isinstance(image_field, list):
            image_url = image_field[0]
        else:
            image_url = image_field

        lower_url = image_url.lower()

        # 可选：过滤掉 TIF 大图，避免太大（按需保留/删除）
        if lower_url.endswith(".tif") or lower_url.endswith(".tiff"):
            # tqdm 下少说话，避免刷屏，可以不打印
            # print(f"[跳过] {wid} 是 TIF 大图，暂不下载: {image_url}")
            continue

        # 粗略判断扩展名
        ext = ".jpg"
        if lower_url.endswith(".png"):
            ext = ".png"
        elif lower_url.endswith(".jpeg"):
            ext = ".jpeg"
        elif lower_url.endswith(".gif"):
            ext = ".gif"
        elif lower_url.endswith(".svg"):
            ext = ".svg"   # 先保存下来，后面想转再说

        filename = out_dir / f"{wid}{ext}"

        # 已经下载过就跳过（便于复跑）
        if filename.exists():
            # 这里也可以不打印，tqdm 本身就是进度反馈
            # tqdm.write(f"[跳过] {wid} 已存在: {filename}")
            continue

        try:
            resp = requests.get(image_url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(filename, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            # 不打断整个流程，只在 tqdm 下输出一行错误信息
            tqdm.write(f"[失败] {wid} 下载出错: {e}")


def main():
    # ===== 参数 =====
    seed_k = 10 # 种子节点个数
    hops = 1      # 跳数：1=一跳邻居，2=二跳邻居
    seed = 42     # 随机种子，固定后采样可复现

    # 抽子图（带 seed）
    result = extract_subgraph(seed_k=seed_k, hops=hops, seed=seed)
    nodes = result["nodes"]   # dict: node_id -> {labels, properties}
    edges = result["edges"]   # list: {id, src, src_labels, dst, dst_labels, type}

    print(f"子图节点数: {len(nodes)}")
    print(f"子图边数  : {len(edges)}")

    # ===== 1) 边写入 TXT =====
    edge_filename = f"subgraph_edges_seed{seed_k}_hop{hops}_s{seed}.txt"
    with open(edge_filename, "w", encoding="utf-8") as f:
        # 每行：src_id  src_type  dst_id  dst_type  edge_type
        for e in edges:
            src_id = e["src"]
            dst_id = e["dst"]
            # 节点类型：这里用所有 label，用 '|' 拼起来
            src_type = "|".join(e["src_labels"]) if e["src_labels"] else ""
            dst_type = "|".join(e["dst_labels"]) if e["dst_labels"] else ""
            edge_type = e["type"]
            line = f"{src_id}\t{src_type}\t{dst_id}\t{dst_type}\t{edge_type}\n"
            f.write(line)

    print(f"边关系已写入: {edge_filename}")

    # ===== 2) 节点完整信息写入 JSON =====
    node_filename = f"subgraph_nodes_seed{seed_k}_hop{hops}_s{seed}.json"

    # 整理成列表形式，更通用
    node_list = []
    for nid, data in nodes.items():
        node_list.append({
            "id": nid,
            "labels": data["labels"],
            "properties": data["properties"],
        })

    with open(node_filename, "w", encoding="utf-8") as f:
        json.dump(node_list, f, ensure_ascii=False, indent=2)

    print(f"节点完整信息已写入: {node_filename}")

    # ===== 3) 只对 ART 节点下载图片，文件名 = wid (QID) =====
    art_only = [n for n in node_list if "ART" in n["labels"]]
    download_images(art_only, seed_k=seed_k, hops=hops)
    print("图片下载完成。")


if __name__ == "__main__":
    main()
