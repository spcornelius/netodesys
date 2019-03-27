import abc
from collections.abc import MutableMapping

__all__ = []

__all__.extend([
    'Dict',
    'DictMeta',
    'OuterDict',
    'AttrDict',
    'NodeDict',
    'NodeAttrDict',
    'AdjlistOuterDict',
    'AdjlistInnerDict',
    'EdgeAttrDict',
])

_mutating_methods = {'__setitem__', '__delitem__', 'pop', 'popitem',
                     'clear', 'update'}


def modifies_dynamics(method):
    def wrapped(self, *args, **kwargs):
        self._instance.expire_dynamics()
        return method(self, *args, **kwargs)

    return wrapped


class DictMeta(abc.ABCMeta):

    def __call__(cls, *args, **kwargs):
        # wrap all methods that change the dictionary with above callback
        for m in _mutating_methods:
            setattr(cls, m, modifies_dynamics(getattr(cls, m)))
        return super().__call__(*args, **kwargs)


class Dict(MutableMapping, metaclass=DictMeta):
    """ base nested dictionary class that tracks changes """
    _child_cls = None

    def __init__(self, data=None, instance=None):
        self._data = data
        self._instance = instance

    def __setitem__(self, key, value):
        if isinstance(value, MutableMapping) and self._child_cls is not None:
            value = self._child_cls(data=value, instance=self._instance)
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, attr):
        return getattr(self._data, attr)


class AttrDict(Dict):
    pass


class OuterDict(Dict):

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = self.__class__(data=value,
                                                       instance=instance)


class NodeAttrDict(AttrDict):
    pass


class EdgeAttrDict(AttrDict):
    pass


class AdjlistInnerDict(Dict):
    _child_cls = EdgeAttrDict


class AdjlistOuterDict(OuterDict):
    _child_cls = AdjlistInnerDict


class NodeDict(OuterDict):
    _child_cls = NodeAttrDict
