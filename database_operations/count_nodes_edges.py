# -*- coding: utf-8 -*-
"""
Basic stats from the Neo4j graph:
- Total number of nodes
- Total number of relationships
- Top labels / relationship types (optional)
"""

from neo4j import GraphDatabase

import config as cfg


def main():
    # Create connection
    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
        encrypted=False,
    )

    with driver.session(database=cfg.NEO4J_DATABASE) as session:
        # 1) Count total nodes
        node_result = session.run("MATCH (n) RETURN count(n) AS node_count;")
        node_count = node_result.single()["node_count"]

        # 2) Count total relationships
        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count;")
        rel_count = rel_result.single()["rel_count"]

        print("===== Global Stats =====")
        print(f"Total nodes: {node_count}")
        print(f"Total relationships: {rel_count}")
        print("-" * 40)

        # 3) Top labels by node count (optional)
        print("===== Top Node Labels (Top 10) =====")
        label_query = """
        MATCH (n)
        WITH labels(n) AS lbls, count(*) AS cnt
        RETURN lbls, cnt
        ORDER BY cnt DESC
        LIMIT 10;
        """
        for record in session.run(label_query):
            print(f"Label set {record['lbls']} -> {record['cnt']} nodes")

        print("-" * 40)

        # 4) Top relationship types (optional)
        print("===== Top Relationship Types (Top 10) =====")
        rel_type_query = """
        MATCH ()-[r]->()
        WITH type(r) AS rel_type, count(*) AS cnt
        RETURN rel_type, cnt
        ORDER BY cnt DESC
        LIMIT 10;
        """
        for record in session.run(rel_type_query):
            print(f"Relationship type {record['rel_type']} -> {record['cnt']} relationships")

    driver.close()
    print("-" * 40)
    print("Done. Connection closed.")


if __name__ == "__main__":
    main()
