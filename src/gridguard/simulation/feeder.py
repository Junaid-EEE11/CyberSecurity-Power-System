"""Phase-aware feeder topology extraction and graph construction."""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import opendssdirect as dss

from gridguard.simulation.opendss_interface import OpenDSSInterface
from gridguard.utils.logging import get_logger

logger = get_logger("gridguard.simulation.feeder")


class FeederModel:
    """Extracts bus-phase nodes and builds graph representations of unbalanced feeders."""

    def __init__(self, dss_interface: OpenDSSInterface):
        """Initialize feeder model from an OpenDSS session.

        Args:
            dss_interface: Active OpenDSS interface.
        """
        self.dss = dss_interface
        self.node_names: List[str] = self.dss.get_all_node_names()
        self.num_nodes: int = len(self.node_names)
        self.node_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.node_names)}
        self.idx_to_node: Dict[int, str] = {i: name for i, name in enumerate(self.node_names)}

        self.node_metadata: List[Dict[str, Any]] = self._extract_node_metadata()
        self.graph: nx.Graph = self._build_feeder_graph()

    def _extract_node_metadata(self) -> List[Dict[str, Any]]:
        """Extract metadata for each bus-phase node."""
        metadata = []
        phase_map = {"1": "A", "2": "B", "3": "C", 1: "A", 2: "B", 3: "C"}

        for idx, node_name in enumerate(self.node_names):
            parts = node_name.split(".")
            bus_name = parts[0]
            phase_num = parts[1] if len(parts) > 1 else "1"
            phase_letter = phase_map.get(phase_num, f"P{phase_num}")

            # Get bus nominal voltage if available
            dss.Circuit.SetActiveBus(bus_name)
            kv_base = float(dss.Bus.kVBase())

            meta = {
                "node_id": idx,
                "node_name": node_name,
                "bus_name": bus_name,
                "phase_num": int(phase_num) if phase_num.isdigit() else 1,
                "phase_name": phase_letter,
                "kv_base": kv_base,
            }
            metadata.append(meta)

        return metadata

    def _build_feeder_graph(self) -> nx.Graph:
        """Construct NetworkX graph where nodes are bus-phases and edges are physical lines/transformers."""
        G = nx.Graph()

        # Add nodes with attributes
        for meta in self.node_metadata:
            G.add_node(
                meta["node_id"],
                node_name=meta["node_name"],
                bus_name=meta["bus_name"],
                phase_num=meta["phase_num"],
                phase_name=meta["phase_name"],
                kv_base=meta["kv_base"],
            )

        # 1. Add edges from Lines
        for line_name in dss.Lines.AllNames():
            dss.Lines.Name(line_name)
            b1 = dss.Lines.Bus1().lower()
            b2 = dss.Lines.Bus2().lower()
            phases = dss.Lines.Phases()

            b1_parts = b1.split(".")
            b2_parts = b2.split(".")
            b1_base = b1_parts[0]
            b2_base = b2_parts[0]

            b1_phases = b1_parts[1:] if len(b1_parts) > 1 else [str(p) for p in range(1, phases + 1)]
            b2_phases = b2_parts[1:] if len(b2_parts) > 1 else [str(p) for p in range(1, phases + 1)]

            for p1, p2 in zip(b1_phases, b2_phases):
                u_name = f"{b1_base}.{p1}"
                v_name = f"{b2_base}.{p2}"
                if u_name in self.node_to_idx and v_name in self.node_to_idx:
                    u_idx = self.node_to_idx[u_name]
                    v_idx = self.node_to_idx[v_name]
                    G.add_edge(u_idx, v_idx, element_type="line", element_name=line_name)

        # 2. Add edges from Transformers
        for xfmr_name in dss.Transformers.AllNames():
            dss.Transformers.Name(xfmr_name)
            buses = [b.lower() for b in dss.CktElement.BusNames()]
            if len(buses) >= 2:
                b1_parts = buses[0].split(".")
                b2_parts = buses[1].split(".")
                b1_base = b1_parts[0]
                b2_base = b2_parts[0]
                b1_phases = b1_parts[1:] if len(b1_parts) > 1 else ["1", "2", "3"]
                b2_phases = b2_parts[1:] if len(b2_parts) > 1 else ["1", "2", "3"]

                for p1, p2 in zip(b1_phases, b2_phases):
                    u_name = f"{b1_base}.{p1}"
                    v_name = f"{b2_base}.{p2}"
                    if u_name in self.node_to_idx and v_name in self.node_to_idx:
                        u_idx = self.node_to_idx[u_name]
                        v_idx = self.node_to_idx[v_name]
                        G.add_edge(u_idx, v_idx, element_type="transformer", element_name=xfmr_name)

        logger.info(
            "Built feeder graph: %d nodes, %d edges, %d connected components",
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.number_connected_components(G),
        )
        return G

    def get_edge_index(self) -> np.ndarray:
        """Get graph edge index in PyTorch Geometric format (2, 2*E) for bidirectional graph.

        Returns:
            2D numpy array of shape (2, 2*E) containing source and target node indices.
        """
        edges: List[Tuple[int, int]] = []
        for u, v in self.graph.edges():
            edges.append((u, v))
            edges.append((v, u))
        # Self loops
        for i in range(self.num_nodes):
            edges.append((i, i))

        edge_array = np.array(edges, dtype=np.int64).T
        return edge_array

    @property
    def edge_index(self) -> np.ndarray:
        """Property returning edge index array."""
        return self.get_edge_index()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get dense adjacency matrix with self-loops."""
        A = nx.to_numpy_array(self.graph, nodelist=list(range(self.num_nodes)))
        A = A + np.eye(self.num_nodes)
        return A
