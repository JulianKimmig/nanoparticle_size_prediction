class Featurizer:
    def __init__(self, length=None, pre_featurize=None, name=None):
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.pre_featurize = pre_featurize
        self._len = length

    def __len__(self):
        return self._len

    def __call__(self, to_featurize):
        if self.pre_featurize is not None:
            to_featurize = self.pre_featurize(to_featurize)
        f = self.featurize(to_featurize)
        if self._len is None:
            self._len = len(f)
        return f

    def featurize(self, to_featurize):
        return [to_featurize]

    def __str__(self):
        return self._name


class OneHotEncodingException(Exception):
    pass


class OneHotFeaturizer(Featurizer):
    def __init__(self, possible_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._len = len(possible_values)
        self.possible_values = possible_values

    def __len__(self):
        return len(self.possible_values)

    def featurize(self, to_featurize):
        if None in self.possible_values and to_featurize not in self.possible_values:
            to_featurize = None
        if to_featurize not in self.possible_values:
            raise OneHotEncodingException(
                "cannot one hot encode '{}' in '{}', allowed values are {}".format(
                    to_featurize, self, self.possible_values
                )
            )
        return list(map(lambda v: to_featurize == v, self.possible_values))


class LambdaFeaturizer(Featurizer):
    def __init__(self, lamda_call, length, *args, **kwargs):
        super().__init__(length=length, *args, **kwargs)
        self.featurize = lamda_call


class FeaturizerList(Featurizer):
    def __init__(self, feature_list, *args, **kwargs):
        super().__init__(length=sum([len(f) for f in feature_list]), *args, **kwargs)
        self._feature_list = feature_list

    def featurize(self, to_featurize):
        features = []
        for f in self._feature_list:
            features.extend(f(to_featurize))
        return features
