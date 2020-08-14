import scipy
import torch
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx
import rdkit
import numpy as np


class Graph():
    def __init__(self, bidirectional=False):
        if bidirectional:
            self._graph_backend = nx.DiGraph()
        else:
            self._graph_backend = nx.Graph()
        # self.node_features = {}
        self._expect_feature_length = None

    def add_edges(self, edges):
        for e in edges:
            self._graph_backend.add_edge(e[0], e[1])

    def add_nodes(self, nodes, features=None):
        for i, n in enumerate(nodes):
            self._graph_backend.add_node(n)
            # self.node_features[n] = []
            if features is not None:
                self.set_node_features(n, features[i] if features is not None else [])

    def set_node_features(self, node, features):
        if self._expect_feature_length is not None:
            assert len(features) == self._expect_feature_length
        else:
            self._expect_feature_length = len(features)
        features = np.array(features)
        # self.node_features[node] = np.array(features)
        self._graph_backend.nodes[node]['nfeats'] = features

    def feature_matrix(self):
        node_feats = nx.classes.function.get_node_attributes(self._graph_backend, 'nfeats')
        if self._expect_feature_length is None:
            self._expect_feature_length = len(next(iter(node_feats.values())))
        feature_m = np.zeros((len(node_feats), self._expect_feature_length))
        for i in np.arange(len(node_feats)):
            feature_m[i] = node_feats[i]
        return feature_m

    def degree_vector(self):
        return np.array([d[1] for d in sorted(list(self._graph_backend.degree))])

    def adjacency_list(self):
        return [(n, np.array(list(nbrdict.keys()))) for n, nbrdict in self._graph_backend.adjacency()]

    def adjacency_matrix(self):
        adj = self.adjacency_list()
        adj = [con for idx, con in sorted(adj)]
        adj_m = np.zeros((len(adj), len(adj)))
        for i, c in enumerate(adj):
            adj_m[i, c] = 1
        return adj_m

    @property
    def graph_backend(self):
        return self._graph_backend

    def size(self):
        return self._graph_backend.number_of_nodes()

    def sparse_adjacency_matrix(self, format='coo'):
        return nx.convert_matrix.to_scipy_sparse_matrix(self._graph_backend, format=format)

    def node_indices(self):
        return np.zeros(self.size(),dtype=int)

    def subgraphs(self):
        return np.array([1])

class DiGraph(Graph):
    def __init__(self):
        super().__init__(bidirectional=True)

    def out_degree_vector(self):
        return np.array([d[1] for d in sorted(list(self._graph_backend.out_degree))])

    def in_degree_vector(self):
        return np.array([d[1] for d in sorted(list(self._graph_backend.in_degree))])


class MergedGraph(Graph):
    def __init__(self, graphs):
        digraph = False
        self.indices = []
        _graphs = []
        for g in graphs:
            if not digraph and isinstance(g, DiGraph):
                digraph = True
            _graphs.append(g)
        super().__init__(bidirectional=digraph)
        self._graph_backend = nx.algorithms.operators.all.disjoint_union_all([g._graph_backend for g in _graphs])

        node_indices = []
        ni = 0
        tot_len = 0
        for g in _graphs:
            add_len = g.size()
            tot_len += add_len
            node_indices.extend(np.ones(add_len) * ni)
            ni += 1

        node_indices = np.array(node_indices,dtype=int)
        self._node_indices = node_indices
        self._subgraphs = np.array([ni])

    def node_indices(self):
        return self._node_indices

    def subgraphs(self):
        return self._subgraphs

class TorchGraph():
    def __init__(self, g: Graph):
        self.graph = g
        self.device = None

        self._tensors = {}

    @property
    def node_indices(self):
        tensor = "node_indices"
        if tensor in self._tensors:
            return self._tensors[tensor]

        new_tensor = torch.from_numpy(self.graph.node_indices())
        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def node_split_indices(self):
        tensor = "node_split_indices"
        if tensor in self._tensors:
            return self._tensors[tensor]

        node_indices = self.graph.node_indices()
        split_indices = (node_indices[1:]-node_indices[:-1]).nonzero()[0]+1
        split_indices = np.insert(split_indices, 0, 0, axis=0)
        new_tensor = torch.from_numpy(split_indices)
        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def sparse_adj(self):
        tensor = "sparse_adj"
        if tensor in self._tensors:
            return self._tensors[tensor]

        coo = self.graph.sparse_adjacency_matrix(format="coo")
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        new_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def sparse_adj_tilde(self):
        tensor = "sparse_adj_tilde"
        if tensor in self._tensors:
            return self._tensors[tensor]

        coo = (self.graph.sparse_adjacency_matrix(format="csr") + scipy.sparse.eye(self.graph.size(),
                                                                                   format="csr")).tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        new_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def sparse_degree(self):
        tensor = "sparse_degree"
        if tensor in self._tensors:
            return self._tensors[tensor]

        coo = (scipy.sparse.eye(self.graph.size(), format="csr").multiply(
            self.graph.sparse_adjacency_matrix(format="csr").sum(axis=1).flatten()
        )).tocoo()

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        new_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def sparse_degree_tilde(self):
        tensor = "sparse_degree_tilde"
        if tensor in self._tensors:
            return self._tensors[tensor]

        coo = (scipy.sparse.eye(self.graph.size(), format="csr").multiply(
            (self.graph.sparse_adjacency_matrix(format="csr").sum(axis=1).flatten() + 1)
        )).tocoo()

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        new_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def sparse_norm_adj_tilde(self):
        tensor = "sparse_norm_adj_tilde"
        if tensor in self._tensors:
            return self._tensors[tensor]

        d_tilde = (scipy.sparse.eye(self.graph.size(), format="csr").multiply(
            (self.graph.sparse_adjacency_matrix(format="csr").sum(axis=1).flatten() + 1)
        ))
        d_neg_half = d_tilde.power(-0.5)

        a_tilde = self.graph.sparse_adjacency_matrix(format="csr") + scipy.sparse.eye(self.graph.size(), format="csr")

        coo = d_neg_half.dot(a_tilde).dot(d_neg_half).tocoo()

        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        new_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def feature_matrix(self):
        tensor = "feature_matrix"
        if tensor in self._tensors:
            return self._tensors[tensor]

        new_tensor = torch.from_numpy(self.graph.feature_matrix()).float()
        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    @property
    def subgraphs(self):
        tensor = "subgraphs"
        if tensor in self._tensors:
            return self._tensors[tensor]

        new_tensor = torch.from_numpy(self.graph.subgraphs())
        if self.device:
            new_tensor = new_tensor.to(self.device)
        self._tensors[tensor] = new_tensor
        return new_tensor

    def to(self, device):
        if self.device == device:
            return
        self.device = device

        for key in self._tensors.keys():
            self._tensors[key] = self._tensors[key].to(device)


def merge_graphs(graphs):
    return MergedGraph(graphs)


class MolGraph(DiGraph):
    pass


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
