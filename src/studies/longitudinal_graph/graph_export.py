"""Explicit full/support nerve exports; overlap is never a temporal edge."""
from __future__ import annotations
import json
import networkx as nx
import numpy as np
import pandas as pd


def graph_views(scenario, visits):
    nodes = scenario["graph"]["nodes"]
    graph = scenario["raw_nerve"]
    if set(graph) != set(nodes):
        raise ValueError("Graph node inventory differs from memberships")
    # Check both directions: no spurious edge and no unexported overlap edge.
    expected = {tuple(sorted((a, b))) for i, a in enumerate(sorted(nodes))
                for b in sorted(nodes)[i + 1:] if nodes[a] & nodes[b]}
    if {tuple(sorted(edge)) for edge in graph.edges} != expected:
        raise ValueError("Nerve edges do not agree with shared visits")
    node_table = scenario["node_table"].set_index("node_id")
    supported = set(node_table.index[node_table.supported.astype(bool)])
    outputs, metadata = {}, {}
    edge_columns = ["from_node", "to_node", "graph_scope", "edge_filter_reason", "weighted_by",
                    "n_shared_visits", "n_shared_patients", "n_common_patients", "visit_jaccard", "patient_jaccard"]
    for scope, selected in (("full", graph), ("supported", graph.subgraph(supported).copy())):
        components = sorted(nx.connected_components(selected), key=lambda c: (-len(c), min(c)))
        component_id = {node: i for i, group in enumerate(components) for node in group}
        articulations = set(nx.articulation_points(selected))
        exported_nodes = node_table.loc[sorted(selected)].reset_index()
        exported_nodes["graph_scope"] = scope
        exported_nodes["connected_component_id"] = exported_nodes.node_id.map(component_id)
        exported_nodes["is_articulation"] = exported_nodes.node_id.isin(articulations)
        exported_nodes["degree"] = exported_nodes.node_id.map(dict(selected.degree()))
        exported_nodes["node_filter_reason"] = np.where(exported_nodes.supported, "retained", "below_minimum_node_patients")
        exported_nodes["weighted_by"] = "patient_jaccard"
        edges = []
        for a, b in sorted(tuple(sorted(edge)) for edge in selected.edges):
            av, bv = nodes[a], nodes[b]
            ap = set(visits.iloc[sorted(av)].patient_id)
            bp = set(visits.iloc[sorted(bv)].patient_id)
            shared = av & bv
            pj, vj = len(ap & bp) / len(ap | bp), len(shared) / len(av | bv)
            edges.append({"from_node": a, "to_node": b, "graph_scope": scope,
                          "edge_filter_reason": "retained" if {a, b} <= supported else "unsupported_endpoint_excluded_from_supported_view",
                          "weighted_by": "patient_jaccard", "n_shared_visits": len(shared),
                          "n_shared_patients": visits.iloc[sorted(shared)].patient_id.nunique(),
                          "n_common_patients": len(ap & bp), "visit_jaccard": vj, "patient_jaccard": pj})
        exported_edges = pd.DataFrame(edges, columns=edge_columns)
        degree_sum = sum(dict(selected.degree()).values())
        if degree_sum != 2 * len(edges):
            raise AssertionError("Degree/edge conservation failed")
        outputs[f"mapper_nodes_{scope}"] = exported_nodes
        outputs[f"mapper_edges_{scope}"] = exported_edges
        metadata[scope] = {"graph_scope": scope, "n_nodes": len(selected), "n_edges": len(edges),
                           "degree_sum": degree_sum, "n_components": len(components),
                           "n_isolates": len(list(nx.isolates(selected))), "n_articulation_nodes": len(articulations),
                           "degree_distribution": {str(d): sum(v == d for v in dict(selected.degree()).values())
                                                   for d in set(dict(selected.degree()).values())},
                           "edge_semantics": "shared_visits; not_temporal_transitions",
                           "support_filter": "minimum_node_patients; macrostate_support_is_separate"}
    historical = len(scenario.get("topological_edges", []))
    qc = {"status": "passed", "views": metadata, "legacy_export_n_edges": historical,
          "legacy_export_scope": "supported", "legacy_matches_supported": historical == metadata["supported"]["n_edges"],
          "edges_excluded_by_node_support": metadata["full"]["n_edges"] - metadata["supported"]["n_edges"],
          "explanation": "The legacy CSV uses supported nodes; degree distribution uses the full nerve. Counts are computed from source memberships, never supplied constants."}
    if not qc["legacy_matches_supported"]:
        qc["status"] = "failed"
        raise ValueError("Legacy supported edge export disagrees with source nerve")
    return outputs, qc


def membership_tables(scenario, visits):
    wide = scenario["visit_membership"]
    node_rows, macro_rows = [], []
    for node, indices in sorted(scenario["graph"]["nodes"].items()):
        for i in sorted(indices):
            weight = wide.iloc[i].get(f"node_weight__{node}", np.nan)
            node_rows.append({**visits.iloc[i][["patient_id", "clinical_episode_id"]].to_dict(),
                              "node_id": node, "macrostate_id": scenario["node_to_macrostate"].get(node),
                              "supported_assignment_weight": weight,
                              "weight_rule": "equal_weight_per_supported_node; NA_outside_supported_macrostate"})
    for node, macro in scenario["node_to_macrostate"].items():
        # Deduplicate visit × community; overlapping nodes are retained in the node table.
        for i in sorted(scenario["graph"]["nodes"][node]):
            macro_rows.append({**visits.iloc[i][["patient_id", "clinical_episode_id"]].to_dict(),
                               "macrostate_id": macro, "weight": wide.iloc[i].get(f"macrostate_weight__{macro}", np.nan)})
    return (pd.DataFrame(node_rows, columns=["patient_id", "clinical_episode_id", "node_id", "macrostate_id", "supported_assignment_weight", "weight_rule"]),
            pd.DataFrame(macro_rows, columns=["patient_id", "clinical_episode_id", "macrostate_id", "weight"]).drop_duplicates())


def write_graphs(scenario, visits, directory, dpi):
    """Persist each view plus figures using the same PC lens geometry."""
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    tables, integrity = graph_views(scenario, visits)
    positions = {node: np.mean(scenario["embedding"][sorted(idx), :2], axis=0)
                 for node, idx in scenario["graph"]["nodes"].items()}
    for scope in ("full", "supported"):
        fig, ax = plt.subplots(figsize=(9, 6))
        nodes, edges = tables[f"mapper_nodes_{scope}"], tables[f"mapper_edges_{scope}"]
        for row in edges.itertuples():
            a, b = positions[row.from_node], positions[row.to_node]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=".7", lw=.5 + np.log1p(row.n_shared_visits))
        for row in nodes.itertuples():
            ax.scatter(*positions[row.node_id], s=30 + 6 * row.n_unique_patients,
                       marker="o" if row.supported else "s", edgecolor="black" if row.supported else "red")
        ax.set(title=f"{scenario['summary']['scenario']} — {scope} nerve", xlabel="PC1 lens", ylabel="PC2 lens")
        fig.text(.1, .02, "Edges = shared visits. Squares = unsupported nodes. No temporal orientation.", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(directory / f"02_mapper_graph_{scope}.png", dpi=dpi)
        plt.close(fig)
    return tables, integrity
