import torch
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx
import rdkit
import numpy as np


class MolGraph:
    def __init__(self):
        self._graph_backend = nx.DiGraph()
        self.node_features = {}
        self._expect_feature_length = None

    def add_nodes(self, nodes):
        for n in nodes:
            self._graph_backend.add_node(n)
            self.node_features[n] = []

    def add_edges(self, edges):
        for e in edges:
            self._graph_backend.add_edge(e[0], e[1])

    def set_node_features(self, node, features):
        assert node in self.node_features

        if self._expect_feature_length is not None:
            assert len(features) == self._expect_feature_length
        else:
            self._expect_feature_length = len(features)
        self.node_features[node] = np.array(features)

    def feature_matrix(self):
        assert all(np.array(sorted(self.node_features.keys())) == np.arange(len(self.node_features)))
        feature_m = np.zeros((len(self.node_features), self._expect_feature_length))
        for i in np.arange(len(self.node_features)):
            feature_m[i] = self.node_features[i]
        return feature_m

    def out_degree_vector(self):
        return np.array([d[1] for d in sorted(list(self._graph_backend.out_degree))])

    def in_degree_vector(self):
        return np.array([d[1] for d in sorted(list(self._graph_backend.in_degree))])

    def adjacency_matrix(self):
        adj = [(n, list(nbrdict.keys())) for n, nbrdict in self._graph_backend.adjacency()]
        adj = [con for idx, con in sorted(adj)]
        adj_m = np.zeros((len(adj), len(adj)))
        for i, c in enumerate(adj):
            adj_m[i, c] = 1
        return adj_m


class FixedGraph:
    def __init__(self, g: MolGraph):

        self.features = g.feature_matrix()
        degree = g.out_degree_vector() + 1
        I = np.identity(degree.shape[0])

        adj_m = g.adjacency_matrix()
        adj_m_tilde = adj_m + I

        d_tilde = ((degree ** (-0.5)) * I)
        d_tilde[d_tilde > 1] = 1
        normed_adj = np.matmul(d_tilde, adj_m_tilde)
        normed_adj = np.matmul(normed_adj, d_tilde)
        self.normed_adj = {'ini': torch.from_numpy(normed_adj).float()}

        d_back_tilde = (degree) ** 0.5
        d_back_tilde[d_back_tilde == 0] = 1  # clamp

        norm = d_back_tilde ** (-0.5)
        norm = np.expand_dims(norm, 1)

        self.back_norm = {'ini': torch.from_numpy(norm).float()}

    def get_normed_adj(self, device):
        if device not in self.normed_adj:
            self.normed_adj[device] = self.normed_adj['ini'].to(device)
        return self.normed_adj[device]

    def get_back_norm(self, device):
        if device not in self.back_norm:
            self.back_norm[device] = self.back_norm['ini'].to(device)
        return self.back_norm[device]


class MergedMolGraph():
    def __init__(self,graphs):
        self.graphs=[]
        self.indices=[]
        for g in graphs:
            if not isinstance(g, FixedGraph):
                g = FixedGraph(g)
            self.graphs.append(g)






def merge_graphs(graphs):
    return MergedMolGraph(graphs)



def mol_to_graph(mol, atom_featurizer, canonical_rank=True):
    mol = rdkit.Chem.AddHs(mol)
    if canonical_rank:
        atom_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, atom_order)

    g = MolGraph()
    n_atoms = mol.GetNumAtoms()
    g.add_nodes(range(n_atoms))
    n_bonds = mol.GetNumBonds()
    edges = []
    for i in range(n_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append((u, v))
        edges.append((v, u))
    g.add_edges(edges)

    for atom in mol.GetAtoms():
        atom_features = atom_featurizer(atom)
        g.set_node_features(atom.GetIdx(), atom_features)

    return g, mol
