# graphsql.py
# -*- coding: utf-8 -*-
"""
图数据库操作封装：
- 建立 Neo4j driver
- 提供 extract_subgraph(seed_k, hops, seed) 抽子图
"""

from neo4j import GraphDatabase
from tqdm import tqdm
from typing import Optional
import random
import config as cfg


def get_driver():
    """
    建立并返回一个 Neo4j driver。
    调用方用完后记得 driver.close()。
    """
    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
        encrypted=False,  # 本地没配 TLS 的话关掉加密
    )
    return driver


def extract_subgraph(seed_k: int = 20, hops: int = 2, seed: Optional[int] = None):
    """
    从图中随机抽取 seed_k 个 :ART 节点作为种子，
    从这些种子出发，按无向边做 BFS，最多扩展 hops 跳，
    返回该子图中的所有节点和边。

    - seed: 控制“随机”的伪随机种子，保证可复现。
      如果为 None，则内部随机生成一个 seed 并打印出来。

    为了控制耗时和内存：
    - 第 1 跳：用一个 MATCH ... WHERE id(a) IN $frontier_ids 一次性取完
    - 第 2 跳及以后：对 frontier 中的每个节点单独 MATCH 一次，
      并用 tqdm 打印进度条。

    返回结构:
    {
        "nodes": {
            node_id: {
                "labels": [...],
                "properties": {...}
            },
            ...
        },
        "edges": [
            {
                "id": edge_id,
                "src": src_id,
                "src_labels": [...],
                "dst": dst_id,
                "dst_labels": [...],
                "type": "REL_TYPE"
            },
            ...
        ]
    }
    """
    # 1) 处理随机种子
    if seed is None:
        seed = random.randint(1, 10**9)
        print(f"[INFO] 未提供 seed，自动生成随机种子: {seed}")
    else:
        print(f"[INFO] 使用给定随机种子 seed = {seed}")

    driver = get_driver()
    try:
        with driver.session(database=cfg.NEO4J_DATABASE) as session:
            # 2) 用“可复现”的方式选 seed_k 个 ART 节点作为种子
            #    不用 ORDER BY rand()，而是用 id(a) * seed 生成一个排序 key
            sample_cypher = """
            MATCH (a:ART)
            WITH a, (id(a) * $seed) AS score
            ORDER BY score
            LIMIT $k
            RETURN id(a) AS id, labels(a) AS labels, properties(a) AS props;
            """

            nodes = {}        # node_id -> {labels, properties}
            seed_ids = []     # 种子节点 id 列表
            visited = set()   # 已经加入子图的节点 id

            for record in session.run(sample_cypher, k=seed_k, seed=seed):
                nid = record["id"]
                seed_ids.append(nid)
                visited.add(nid)
                nodes[nid] = {
                    "labels": list(record["labels"]),
                    "properties": record["props"],
                }

            # 如果图里没有 ART 节点，直接返回空
            if not seed_ids:
                return {"nodes": {}, "edges": []}

            print(f"[INFO] 实际选到的种子 ART 节点数: {len(seed_ids)}")

            # 3) BFS：按 hops 扩展
            edges_map = {}   # edge_id -> edge_dict，防止重复
            frontier = list(seed_ids)
            hop = 0

            while hop < hops and frontier:
                hop += 1
                next_frontier = set()

                if hop == 1:
                    # === 第 1 跳：一次性查询所有种子的一跳邻居 ===
                    print(f"[INFO] 开始获取第 1 跳邻居，共 {len(frontier)} 个种子节点")

                    one_hop_cypher = """
                    MATCH (a)-[r]-(n)
                    WHERE id(a) IN $frontier_ids
                    RETURN
                        id(a) AS aid,
                        labels(a) AS alabels,
                        properties(a) AS aprops,
                        id(n) AS nid,
                        labels(n) AS nlabels,
                        properties(n) AS nprops,
                        id(r) AS rid,
                        type(r) AS rtype;
                    """

                    records = session.run(one_hop_cypher, frontier_ids=frontier)

                    for record in records:
                        aid = record["aid"]
                        nid = record["nid"]
                        rid = record["rid"]

                        # 起点节点信息也写入 nodes（有可能之前没写过）
                        if aid not in nodes:
                            nodes[aid] = {
                                "labels": list(record["alabels"]),
                                "properties": record["aprops"],
                            }
                            visited.add(aid)

                        # 记录邻居节点信息
                        if nid not in nodes:
                            nodes[nid] = {
                                "labels": list(record["nlabels"]),
                                "properties": record["nprops"],
                            }

                        # 没访问过的节点加入下一跳 frontier
                        if nid not in visited:
                            visited.add(nid)
                            next_frontier.add(nid)

                        # 记录边（无向，统一记 a -> n）
                        if rid not in edges_map:
                            edges_map[rid] = {
                                "id": rid,
                                "src": aid,
                                "src_labels": list(record["alabels"]),
                                "dst": nid,
                                "dst_labels": list(record["nlabels"]),
                                "type": record["rtype"],
                            }

                else:
                    # === 第 2 跳及后续：逐个节点查询一跳邻居，并用 tqdm 打印进度 ===
                    total = len(frontier)
                    print(f"[INFO] 开始获取第 {hop} 跳邻居，共 {total} 个起点节点")

                    per_node_cypher = """
                    MATCH (a)-[r]-(n)
                    WHERE id(a) = $aid
                    RETURN
                        id(a) AS aid,
                        labels(a) AS alabels,
                        properties(a) AS aprops,
                        id(n) AS nid,
                        labels(n) AS nlabels,
                        properties(n) AS nprops,
                        id(r) AS rid,
                        type(r) AS rtype;
                    """

                    for aid_val in tqdm(frontier, desc=f"第 {hop} 跳", unit="node"):
                        for record in session.run(per_node_cypher, aid=aid_val):
                            aid = record["aid"]
                            nid = record["nid"]
                            rid = record["rid"]

                            # 起点节点信息也写入 nodes（有可能之前没写过）
                            if aid not in nodes:
                                nodes[aid] = {
                                    "labels": list(record["alabels"]),
                                    "properties": record["aprops"],
                                }
                                visited.add(aid)

                            # 记录邻居节点信息
                            if nid not in nodes:
                                nodes[nid] = {
                                    "labels": list(record["nlabels"]),
                                    "properties": record["nprops"],
                                }

                            # 没访问过的节点加入下一跳 frontier
                            if nid not in visited:
                                visited.add(nid)
                                next_frontier.add(nid)

                            # 记录边（无向，统一记 a -> n）
                            if rid not in edges_map:
                                edges_map[rid] = {
                                    "id": rid,
                                    "src": aid,
                                    "src_labels": list(record["alabels"]),
                                    "dst": nid,
                                    "dst_labels": list(record["nlabels"]),
                                    "type": record["rtype"],
                                }

                # 更新 frontier，进入下一跳
                frontier = list(next_frontier)

            # 4) 整理返回
            edges = list(edges_map.values())
            return {
                "nodes": nodes,
                "edges": edges,
            }

    finally:
        driver.close()
