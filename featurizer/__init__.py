from featurizer.featurizer import FeaturizerList

import featurizer.chem_featurizer as chem_featurizer


def load_featurizer(featurizer_functions):
    list = []

    for f in featurizer_functions:
        if hasattr(chem_featurizer, f):
            list.append(getattr(chem_featurizer, f))
            continue
        raise NotImplementedError("featurizer '{}' not found".format(f))

    return FeaturizerList(list)


# return FeaturizerList(
#     featurizer_funcs={'nfeats': ConcatFeaturizer(
#         [getattr(featurizers, feat_func) for feat_func in featurizer_functions]
#     )}
# )
