# config.py
# -*- coding: utf-8 -*-
"""
配置文件：只存放 Neo4j 连接相关配置。

支持通过环境变量覆盖（更安全、更方便在不同机器运行）：
- NEO4J_URI
- NEO4J_USER
- NEO4J_PASSWORD
- NEO4J_DATABASE
"""

import os

# Neo4j 连接地址
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")

# 登录用户名和密码（建议用环境变量覆盖，避免把密码写入代码仓库）
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Bhy15317969589")

# 使用的数据库名称（Neo4j 4.x+ 支持多 DB，默认一般是 neo4j）
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
