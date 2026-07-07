from pdb_to_graph import pdb_to_base_graph
from pdb_to_graph import real_graph
import pdb_to_graph

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Hull:
    def __init__(self, G):
        self.G = G
        self.hull = None
        self.hull_nodes = self.get_convex_hull_nodes(self.G)
        self.water_nodes = self.get_outside_water_nodes()
        pass

    # @staticmethod
    def get_convex_hull_nodes(self, G):
        """
            Return graph nodes that have at least one coordinate point
            lying on the 3D convex hull.
            """

        points = []
        point_nodes = []

        # collect all atomic coordinates
        for node, data in G.nodes(data=True):
            # print(data)
            if data["resname"] != "O" and data["resname"] != "HOH":
                for coord in data["coords"]:
                    points.append(coord)
                    point_nodes.append(node)

        points = np.array(points)

        # need at least 4 non-coplanar points for a 3D hull
        if len(points) < 4:
            return set()

        self.hull = ConvexHull(points)

        hull_nodes = {
            point_nodes[i]
            for i in self.hull.vertices
        }

        return hull_nodes

    def get_water_nodes(self):
        """
        Return all water nodes.
        """

        return {
            node
            for node, data in self.G.nodes(data=True)
            if data["resname"] in {"HOH", "WAT", "H2O"}
        }

    def get_outside_water_nodes(self):
        """
        Return waters whose coordinates are outside the protein hull.
        """

        if self.hull is None:
            return set()

        outside = set()

        equations = self.hull.equations

        for node in self.get_water_nodes():

            coords = self.G.nodes[node]["coords"]

            for point in coords:

                # positive value means outside the hull
                distances = (
                    np.dot(equations[:, :-1], point)
                    + equations[:, -1]
                )

                if np.any(distances > 1e-8):
                    outside.add(node)
                    break

        return outside

    def get_surface_water_nodes(self):
        """
        Return water nodes connected only to convex hull nodes.
        """

        surface_waters = set()

        for node in self.water_nodes:

            neighbors = set(self.G.neighbors(node))

            # ignore isolated waters
            if not neighbors:
                continue

            # water is surface-bound if all contacts are hull nodes
            if neighbors.issubset(self.hull_nodes):
                surface_waters.add(node)

        return surface_waters


def project_positions(G):
    """
    Generate 2D plotting positions from stored 3D coordinate clouds.
    Uses the mean coordinate of each node's retained atoms.
    """

    pos = {}

    for n, d in G.nodes(data=True):
        coords = d["coords"]

        if len(coords) > 0:
            center = np.mean(coords, axis=0)
        else:
            center = np.array([0.0, 0.0, 0.0])

        pos[n] = (center[0], center[1])  # x-y projection

    return pos


def plot_graph(G, highlight_residues=None):
    plt.figure(figsize=(16, 12))

    # -----------------------------
    # REAL STRUCTURE POSITION
    # -----------------------------
    # pos = project_positions(G)
    pos = nx.kamada_kawai_layout(G, scale=5)

    pos = nx.spring_layout(
        G,
        pos=pos,
        k=1.5,
        iterations=30,
        seed=42
    )

    if highlight_residues is None:
        highlight_residues = set()

    else:
        highlight_residues = set(highlight_residues)

    # -----------------------------
    # NODE COLORS
    # -----------------------------
    # hull = Hull(G)
    # hull_nodes = Hull.get_convex_hull_nodes(G)
    # hull_nodes = hull.hull_nodes
    node_colors = []
    for n, d in G.nodes(data=True):
        if d["resname"] in {"HOH", "WAT"}:
            # if n in hull_nodes:
            if G.nodes[n].get("absorbing", True):
                node_colors.append("green")
            else:
                node_colors.append("grey")

        elif G.nodes[n].get("absorbing", True):
            node_colors.append("green")
        elif n in highlight_residues:
            node_colors.append("cyan")   # your query set

        else:
            node_colors.append("red")

    # -----------------------------
    # DRAW
    # -----------------------------

    nx.draw_networkx_nodes(
        G, pos,
        node_size=60,
        node_color=node_colors,
        alpha=0.8
    )

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        width=0.5
    )

    labels = {n: n for n in G.nodes() if not n.startswith(("HOH_", "WAT_"))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.axis("off")
    plt.title("Protein Graph (Structure-Preserving Layout)")
    plt.show()


def prune_to_starter_without_absorbing(G, keep_absorbing=True):
    """
        Keep only nodes reachable from starter nodes before crossing an absorbing node.

        Absorbing nodes are treated as terminal boundaries:
        - they are kept,
        - their successors are not explored,
        - nodes only reachable after absorption are removed.
        """

    sources = {
        n for n, data in G.nodes(data=True)
        if data.get("starter", False)
    }

    if not sources:
        raise ValueError("No starter nodes found")

    keep = set()
    stack = list(sources)

    while stack:
        node = stack.pop()
        # G.nodes[node]["visited"] = True
        if node in keep:
            continue

        keep.add(node)

        # absorbing nodes terminate traversal
        if G.nodes[node].get("absorbing", True):
            continue

        # only continue through non-absorbing nodes
        for neighbour in G.neighbors(node):
            # if G.nodes[neighbour].get("visible", False):
            stack.append(neighbour)

    remove_nodes = set(G.nodes) - keep
    G.remove_nodes_from(remove_nodes)

    return G


def plot_3D(G, highlight_residues=None):
    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(111, projection="3d")

    if highlight_residues is None:
        highlight_residues = set()
    else:
        highlight_residues = set(highlight_residues)

    # -----------------------------
    # NODE COLORS
    # -----------------------------
    node_colors = {}

    for n, d in G.nodes(data=True):

        # water molecules
        if d.get("resname") in {"HOH", "WAT"}:
            if d.get("absorbing", True):
                node_colors[n] = "green"
            else:
                node_colors[n] = "grey"

        # absorbing nodes
        elif d.get("absorbing", True):
            node_colors[n] = "green"

        # highlighted residues
        elif n in highlight_residues or d.get("starter", True):
            node_colors[n] = "cyan"

        # normal residues
        else:
            node_colors[n] = "red"

    # -----------------------------
    # DRAW EDGES
    # -----------------------------
    for u, v in G.edges():

        x1, y1, z1 = G.nodes[u]["coords"][0]
        x2, y2, z2 = G.nodes[v]["coords"][0]

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="black",
            alpha=0.6,
            linewidth=1.2
        )

    # -----------------------------
    # DRAW NODES
    # -----------------------------
    for n, d in G.nodes(data=True):

        x, y, z = d["coords"][0]

        ax.scatter(
            x, y, z,
            s=60,
            c=node_colors[n],
            alpha=0.8
        )

        # don't label water
        if not n.startswith(("HOH_", "WAT_")):
            ax.text(
                x, y, z,
                str(n),
                fontsize=7
            )

    # -----------------------------
    # AXIS
    # -----------------------------
    ax.set_axis_off()

    # remove background panes and grid
    ax.grid(False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Protein Graph (3D Structure Coordinates)")

    plt.show()


if __name__ == "__main__":
    G = pdb_to_base_graph("1CA2.pdb")
    # G = pdb_to_graph.prune_atoms(G)
    G = pdb_to_graph.graph_pipe(G)

    highlight_residues = [
        "SER_65_A",
        "TYR_194_A",
        "GLU_117_A"
    ]

    hull = Hull(G)
    # nodes to exclude from drawing
    hidden_nodes = hull.get_outside_water_nodes()
    hidden_nodes |= hull.get_surface_water_nodes()
    # create a view without those nodes
    Gp = G.subgraph(
        n for n in G.nodes()
        if n not in hidden_nodes
    )

    for node in hull.hull_nodes:
        if node in Gp:
            Gp.nodes[node]["absorbing"] = True
            # print(node)
    sources = [(194, 'A'), (117, 'A')]
    for node_id, data in Gp.nodes(data=True):
        key = (data["resid"], data["chain"])
        if key in sources:
            data["starter"] = True
            print("Set starter:", node_id, key)

    # print(Gp)
    Gp = Gp.copy()
    Gp = prune_to_starter_without_absorbing(Gp)

    # plot_graph(Gp, highlight_residues=highlight_residues)

    plot_3D(Gp)
