import numpy as np
import networkx as nx
from collections import defaultdict


# -----------------------------
# CONFIG
# -----------------------------
ACTIVE_RES = {"HIS", "ASP", "GLU", "LYS", "ARG", "SER", "THR", "CYS", "TYR"}
WATER_RES = {"HOH", "WAT"}
CUTOFF = 4.0  # Angstrom

ACTIVE_RESIDUES = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "LYS": {"NZ"},
    "ARG": {"NH1", "NH2", "NE"},
    "TYR": {"OH"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "HIS": {"ND1", "NE2"},
}


# -----------------------------
# 1. PDB PARSER (ATOM / HETATM)
# -----------------------------


def parse_pdb(pdb_file):
    atoms = []

    with open(pdb_file, "r") as f:
        for line in f:
            record = line[0:6].strip()

            if record not in ("ATOM", "HETATM"):
                continue

            atom = {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip(),
                "resid": int(line[22:26]),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": line[76:78].strip() if len(line) > 76 else ""
            }

            atoms.append(atom)

    return atoms

# -----------------------------
# 2. SELECT NODES
# -----------------------------


def select_nodes(atoms):
    nodes = []

    for i, atom in enumerate(atoms):
        is_active = atom["resname"] in ACTIVE_RES
        is_water_O = (atom["resname"] in WATER_RES and atom["name"] == "O")

        if is_active or is_water_O:
            nodes.append((i, atom))

    return nodes

# -----------------------------
# 3. DISTANCE FUNCTION
# -----------------------------


def distance(a, b):
    return np.sqrt(
        (a["x"] - b["x"])**2 +
        (a["y"] - b["y"])**2 +
        (a["z"] - b["z"])**2
    )

# -----------------------------
# 4. BUILD GRAPH
# -----------------------------


def base_graph(nodes, cutoff=CUTOFF):
    G = nx.Graph()

    # -----------------------------
    # 1. add nodes (standardized)
    # -----------------------------
    for idx, atom in nodes:

        node_id = f"{atom['resname']}_{atom['resid']}_{atom['name']}_{atom.get('chain', 'A')}_{idx}"

        G.add_node(
            node_id,
            resname=atom["resname"],
            resid=atom["resid"],
            name=atom["name"],
            chain=atom.get("chain", "A"),  # IMPORTANT FIX
            coord=np.array([atom["x"], atom["y"], atom["z"]])
        )

    # -----------------------------
    # 2. add edges (O(N^2))
    # -----------------------------
    for i in range(len(nodes)):
        idx_i, ai = nodes[i]

        id_i = f"{ai['resname']}_{ai['resid']}_{ai['name']}_{ai.get('chain', 'A')}_{idx_i}"

        for j in range(i + 1, len(nodes)):
            idx_j, aj = nodes[j]

            id_j = f"{aj['resname']}_{aj['resid']}_{aj['name']}_{aj.get('chain', 'A')}_{idx_j}"

            d = distance(ai, aj)

            if d <= cutoff:
                G.add_edge(id_i, id_j, weight=d)

    return G
# -----------------------------
# 5. FULL PIPELINE
# -----------------------------


def prune_atoms(atoms):
    pruned = []

    for atom in atoms:

        # always remove hydrogens
        if atom["name"].startswith("H"):
            continue

        # keep water oxygens
        if atom["resname"] in {"HOH", "WAT"} and atom["name"] == "O":
            pruned.append(atom)
            continue

        # keep active residues
        if atom["resname"] in ACTIVE_RESIDUES:
            pruned.append(atom)

    return pruned


def pdb_to_base_graph(pdb_file):
    atoms = parse_pdb(pdb_file)
    atoms = prune_atoms(atoms)   # HERE
    nodes = select_nodes(atoms)
    G = base_graph(nodes, CUTOFF)

    return G


def real_graph(atom_graph):
    """
    Compress atom-level graph into residue-level graph.
    Each residue = one node.
    """

    R = nx.Graph()

    # -----------------------------
    # 1. group atoms by residue
    # -----------------------------
    residue_nodes = defaultdict(list)

    for n, data in atom_graph.nodes(data=True):
        key = (data["chain"], data["resname"], data["resid"])
        residue_nodes[key].append((n, data))

    # -----------------------------
    # 2. create residue nodes
    # -----------------------------
    residue_id_map = {}

    BACKBONE = {"N", "CA", "C"}

    def ignore_atom(atom):
        name = atom["name"]
        resname = atom["resname"]
        return name in BACKBONE or name.startswith("C")

    for res_key, atoms in residue_nodes.items():
        chain, resname, resid = res_key

        node_id = f"{resname}_{resid}_{chain}"
        residue_id_map[res_key] = node_id

        # keep only atoms that are not ignored
        kept_atoms = [
            a for _, a in atoms
            if not ignore_atom(a)
        ]

        coords = [
            a["coord"]
            for a in kept_atoms
        ]

        R.add_node(
            node_id,
            resname=resname,
            resid=resid,
            chain=chain,
            coords=coords,
            starter=False,                          # all retained coordinates
            absorbing=False,
            visited=False,
            n_atoms=len(atoms),
            n_kept_atoms=len(coords)
        )
    # -----------------------------
    # 3. build residue edges
    # -----------------------------
    edge_weights = defaultdict(float)

    for u, v, data in atom_graph.edges(data=True):
        if ignore_atom(atom_graph.nodes[u]) or ignore_atom(atom_graph.nodes[v]):
            continue
        ru = (atom_graph.nodes[u]["chain"],
              atom_graph.nodes[u]["resname"],
              atom_graph.nodes[u]["resid"])

        rv = (atom_graph.nodes[v]["chain"],
              atom_graph.nodes[v]["resname"],
              atom_graph.nodes[v]["resid"])

        if ru == rv:
            continue

        ru_id = residue_id_map[ru]
        rv_id = residue_id_map[rv]

        # accumulate contact strength
        edge_weights[(ru_id, rv_id)] += 1.0

    # -----------------------------
    # 4. add edges
    # -----------------------------
    for (u, v), w in edge_weights.items():
        R.add_edge(u, v, weight=w)

    return R


def remove_isolates(G):
    G = G.copy()
    isolates = list(nx.isolates(G))  # <-- key fix
    G.remove_nodes_from(isolates)
    G = G.copy()
    to_remove = []
    for n in G.nodes:
        data = G.nodes[n]

        # only target waters
        if data.get("resname") in WATER_RES:

            if G.degree(n) <= 1:
                to_remove.append(n)

    return G


def graph_pipe(G):
    G = real_graph(G)
    G = remove_isolates(G)
    return G


if __name__ == "__main__":

    G = pdb_to_base_graph("gfp.pdb")
    print(G.number_of_nodes(), G.number_of_edges())
    G = real_graph(G)
    print(G.number_of_nodes(), G.number_of_edges())
