import paramnet.dict as pd
import abc

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


class Dict(pd.Dict, metaclass=DictMeta):
    pass


class AttrDict(Dict, pd.AttrDict):
    pass


class OuterDict(Dict, pd.OuterDict):
    pass


class NodeAttrDict(AttrDict, pd.NodeAttrDict):
    pass


class EdgeAttrDict(AttrDict, pd.EdgeAttrDict):
    pass


class AdjlistInnerDict(Dict):
    _child_cls = EdgeAttrDict


class AdjlistOuterDict(OuterDict):
    _child_cls = AdjlistInnerDict


class NodeDict(OuterDict):
    _child_cls = NodeAttrDict
