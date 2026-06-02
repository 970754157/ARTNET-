# -*- coding: utf-8 -*-
"""
Neo4j 连接自检脚本：
- 检查依赖是否安装
- 打印当前使用的连接配置（来自 config.py / 环境变量覆盖）
- 尝试执行最小查询验证连通性

用法（在项目根目录）：
  py .\\database_operations\\check_connection.py
"""

from __future__ import annotations

import sys

import config as cfg


def main() -> int:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ModuleNotFoundError:
        print("[ERROR] 缺少依赖 neo4j。请先安装依赖：")
        print("        pip install -r requirements.txt")
        return 2

    print("===== Neo4j Config =====")
    print(f"NEO4J_URI      = {cfg.NEO4J_URI}")
    print(f"NEO4J_USER     = {cfg.NEO4J_USER}")
    print(f"NEO4J_DATABASE = {cfg.NEO4J_DATABASE}")
    print("========================")

    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
        encrypted=False,
    )
    try:
        with driver.session(database=cfg.NEO4J_DATABASE) as session:
            value = session.run("RETURN 1 AS ok;").single()["ok"]
            print(f"[OK] Connected. RETURN 1 => {value}")
        return 0
    except Exception as e:
        print("[ERROR] 连接或查询失败：")
        print(f"        {type(e).__name__}: {e}")
        print("")
        print("排查建议：")
        print("- Neo4j 是否已启动，且 Bolt 端口 7687 可用")
        print("- 账号密码是否正确（可用环境变量 NEO4J_PASSWORD 覆盖）")
        print("- 数据库名是否存在（可用环境变量 NEO4J_DATABASE 覆盖）")
        return 1
    finally:
        driver.close()


if __name__ == '__main__':
    raise SystemExit(main())
