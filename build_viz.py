import sys, json, os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.export import to_json, to_html

extraction = json.loads(Path("graphify-out/.graphify_extract.json").read_text(encoding="utf-8"))

G = build_from_json(extraction)
communities = cluster(G)
cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)

labels = {
    cid: {
        0: "Vision & Imaging (T2T-ViT)",
        1: "Classical ML Ensemble",
        2: "Jaundice Image Severity",
        3: "Feature Engineering",
        4: "Training Pipeline",
    }.get(cid, f"Community {cid}")
    for cid in communities
}

to_json(G, communities, "graphify-out/graph.json")
to_html(G, communities, "graphify-out/graph.html", community_labels=labels)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")
print("Outputs:")
print("  graphify-out/graph.html  - open in any browser")
print("  graphify-out/graph.json  - raw graph data")
print()
print("Community labels:")
for cid, lbl in labels.items():
    n = sum(1 for v in communities.values() if v == cid)
    score = cohesion.get(cid, 0)
    print(f"  {cid}: {lbl} ({n} nodes, cohesion={score:.2f})")
