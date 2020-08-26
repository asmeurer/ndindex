from numpy import arange, prod, array, intp

from hypothesis import given, example
from hypothesis.strategies import integers, one_of

from ..ndindex import ndindex
from ..array import ArrayIndex
from ..booleanarray import BooleanArray
from ..integerarray import IntegerArray
from ..integer import Integer
from ..tuple import Tuple
from .helpers import ndindices, check_same, short_shapes

@example((array([], dtype=intp), 0), (0, 0))
@example((array([], dtype=intp), [0]), (0, 0))
@example((..., 0, array([], dtype=intp)), (0, 0))
@example((..., array(0), array([], dtype=intp)), (0, 0))
@example((False, False), ())
@example((-1, False), 1)
@example((..., False), ())
@example((array([0]),), ())
@example(([0, 1], 0), (2, 2))
@example((..., [0, 1], 0), (2, 2))
@example((..., None, 0), 1)
@example((0, 1, ..., 2, 3), (2, 3, 4, 5, 6, 7))
@example(None, 2)
@given(ndindices, one_of(short_shapes, integers(0, 10)))
def test_expand_hypothesis(idx, shape):
    if isinstance(shape, int):
        a = arange(shape)
    else:
        a = arange(prod(shape)).reshape(shape)

    index = ndindex(idx)

    try:
        expanded = index.expand(shape)
    except IndexError:
        pass
    except NotImplementedError:
        return
    else:
        assert isinstance(expanded, Tuple)
        assert ... not in expanded.args
        if isinstance(idx, tuple):
            n_newaxis = index.args.count(None)
        elif index == None:
            n_newaxis = 1
        else:
            n_newaxis = 0
        if not any(isinstance(i, BooleanArray) for i in expanded.args):
            if isinstance(shape, int):
                assert len(expanded.args) == 1 + n_newaxis
            else:
                assert len(expanded.args) == len(shape) + n_newaxis

        # Make sure arrays are broadcasted
        if any(isinstance(i, ArrayIndex) and i not in [True, False] for i in expanded.args):
            assert not any(isinstance(i, Integer) for i in expanded.args)
            assert len({i.shape for i in expanded.args if isinstance(i,
                                                                     IntegerArray)}) in [0, 1]

        assert expanded.args.count(True) <= 1
        assert expanded.args.count(False) <= 1
        assert not (True in expanded.args and False in expanded.args)

    check_same(a, index.raw, ndindex_func=lambda a, x: a[x.expand(shape).raw],
               same_exception=False)
