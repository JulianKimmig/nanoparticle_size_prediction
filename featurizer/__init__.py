from dgllife.utils import BaseAtomFeaturizer, ConcatFeaturizer, featurizers


def load_featurizer(featurizer_functions):
    return BaseAtomFeaturizer(
        featurizer_funcs={'nfeats': ConcatFeaturizer(
            [getattr(featurizers, feat_func) for feat_func in featurizer_functions]
        )}
    )
