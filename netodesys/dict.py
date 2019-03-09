from indexed import IndexedOrderedDict
from abc import ABCMeta
from collections.abc import MutableMapping

__all__ = ['make_dict_factories']

_mutating_methods = {'__setitem__', '__delitem__', 'pop', 'popitem', 'clear', 'update'}


def make_dict_factories(net):

    def modifies_dynamics(method):
        def wrapped(self, *args, **kwargs):
            res = method(self, *args, **kwargs)
            net.expire_dynamics()
            return res

        return wrapped

    class DictBaseMeta(type):

        def __call__(cls, *args, **kwargs):
            # wrap all methods that change the dictionary with above callback
            for m in _mutating_methods:
                setattr(cls, m, modifies_dynamics(getattr(cls, m)))
            return super().__call__(*args, **kwargs)

    class DictBase(object, metaclass=DictBaseMeta):
        """ Mixin for nested dict-like class that alerts the owning
            instance whenever a value changes. """

        # dict-like class that should wrap values in. If None, do not wrap.
        _value_cls = None

        def __setitem__(self, key, value):
            if self._value_cls is not None:
                value = self._value_cls(value)
            super().__setitem__(key, value)

    class NodeAttrDict(DictBase, dict):
        _value_cls = None

    class NodeDict(DictBase, IndexedOrderedDict):
        _value_cls = NodeAttrDict

    class EdgeAttrDict(DictBase, dict):
        _value_cls = None

    class AdjlistOuterDict(DictBase, dict):
        _value_cls = EdgeAttrDict

    class AdjlistInnerDict(DictBase, dict):
        _value_cls = AdjlistOuterDict

    return NodeDict, NodeAttrDict, AdjlistOuterDict, AdjlistInnerDict, EdgeAttrDict