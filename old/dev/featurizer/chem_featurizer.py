import rdkit
import rdkit.Chem.AllChem


class Featurizer():
    def __init__(self, length=None):
        self._len = length

    def __len__(self):
        return self._len

    def __call__(self, to_featurize):
        f = self.featurize(to_featurize)
        if self._len is None:
            self._len = len(f)
        return f


class OneHotFeaturizer(Featurizer):
    def __init__(self, possible_values):
        super().__init__()
        self._len = len(possible_values)
        self.possible_values = possible_values

    def __len__(self):
        return len(self.possible_values)

    def featurize(self, to_featurize):
        if None in self.possible_values and to_featurize not in self.possible_values:
            value = None
        return list(map(lambda v: to_featurize == v, self.possible_values))


class LambdaFeaturizer(Featurizer):
    def __init__(self, lamda_call, length):
        super().__init__(length=length)
        self.featurize = lamda_call


class FeaturizerList():
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def __call__(self, to_featurize):
        features = []
        for f in self.feature_list:
            features.extend(f(to_featurize))
        return features

    def __len__(self):
        l = 0
        for f in self.feature_list:
            print(f)
            l += len(f)
        return l


class AtomSymbolOneHotFeaturizer(OneHotFeaturizer):
    def featurize(self, to_featurize):
        return super().featurize(to_featurize.GetSymbol())


class AtomFormalChargeFeaturizer(Featurizer):
    def featurize(self, to_featurize):
        return [to_featurize.GetFormalCharge()]


atom_symbol_one_hot = AtomSymbolOneHotFeaturizer(
    possible_values=['O', 'Si', 'Al', 'Fe', 'Ca', 'Na', 'Mg', 'K', 'Ti', 'H', 'P', 'Mn', 'F', 'Sr', 'S', 'C', 'Zr',
                     'Cl', 'V', 'Cr', 'Rb', 'Ni', 'Zn', 'Cu', 'Y', 'Co', 'Sc', 'Li', 'Nb', 'N', 'Ga', 'B', 'Ar', 'Be',
                     'Br', 'As', 'Ge', 'Mo', 'Kr', 'Se', 'He', 'Ne', 'Tc', 'Ba', 'Ce', 'Nd', 'La', 'Pb', 'Pr', 'Sm',
                     'Re', 'Gd', 'Dy', 'Rn', 'Er', 'Yb', 'Xe', 'Cs', 'Hf', 'At', 'Sn', 'Pm', 'Eu', 'Ta', 'Po', 'Ho',
                     'W', 'Tb', 'Tl', 'Lu', 'Tm', 'I', 'In', 'Sb', 'Cd', 'Hg', 'Ag', 'Pd', 'Bi', 'Pt', 'Au', 'Os', 'Ru',
                     'Rh', 'Te', 'Ir', 'Fr', 'Th', 'Ra', 'Ac', 'U', 'Pa', 'Np', 'Pu', None]
)

atom_formal_charge_featurizer = AtomFormalChargeFeaturizer(length=1)


def _get_gasteiger_charge(atom):
    if not atom.HasProp('_GasteigerCharge'):
        rdkit.Chem.AllChem.ComputeGasteigerCharges(atom.GetOwningMol())
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        return [0]
    return [float(gasteiger_charge)]


atom_formal_partial_featurizer = LambdaFeaturizer(_get_gasteiger_charge, length=1)

atom_mass_featurizer = LambdaFeaturizer(lambda atom: [ atom.GetMass() * 0.01 ], length=1)

default_atom_featurizer = FeaturizerList([
    atom_symbol_one_hot,
    atom_formal_charge_featurizer,
    atom_formal_partial_featurizer,
    atom_mass_featurizer,
])
