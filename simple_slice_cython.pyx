# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import cython
from cpython cimport PyObject
from libc.stdint cimport int64_t
import sys

cdef extern from "Python.h":
    Py_ssize_t PyNumber_AsSsize_t(object obj, object exc) except? -1
    object PyNumber_Index(object obj)
    bint PyBool_Check(object obj)
    int64_t PyLong_AsLongLong(object obj) except? -1

cdef bint _NUMPY_IMPORTED = False
cdef type _NUMPY_BOOL = None

cdef inline bint is_numpy_bool(object obj):
    global _NUMPY_IMPORTED, _NUMPY_BOOL
    if not _NUMPY_IMPORTED:
        if 'numpy' in sys.modules:
            _NUMPY_BOOL = sys.modules['numpy'].bool_
        _NUMPY_IMPORTED = True
    return _NUMPY_BOOL is not None and isinstance(obj, _NUMPY_BOOL)

cdef inline int64_t cy_operator_index(object idx) except? -1:
    cdef object result

    if PyBool_Check(idx) or is_numpy_bool(idx):
        raise TypeError(f"'{type(idx).__name__}' object cannot be interpreted as an integer")

    return PyNumber_AsSsize_t(idx, IndexError)

cdef class default:
    pass

cdef class SimpleSliceCython:
    cdef readonly tuple args
    cdef int64_t _start
    cdef int64_t _stop
    cdef int64_t _step
    cdef bint _has_start
    cdef bint _has_stop
    cdef bint _has_step

    def __cinit__(self, start, stop=default, step=None):
        self._typecheck(start, stop, step)

    cdef inline void _typecheck(self, object start, object stop, object step) except *:
        cdef object _start, _stop, _step

        if isinstance(start, SimpleSliceCython):
            self.args = (<SimpleSliceCython>start).args
            self._start = (<SimpleSliceCython>start)._start
            self._stop = (<SimpleSliceCython>start)._stop
            self._step = (<SimpleSliceCython>start)._step
            self._has_start = (<SimpleSliceCython>start)._has_start
            self._has_stop = (<SimpleSliceCython>start)._has_stop
            self._has_step = (<SimpleSliceCython>start)._has_step
            return

        if isinstance(start, slice):
            self._typecheck(start.start, start.stop, start.step)
            return

        if stop is default:
            start, stop = None, start

        self._has_start = start is not None
        self._has_stop = stop is not None
        self._has_step = step is not None

        if self._has_start:
            self._start = cy_operator_index(start)
            _start = self._start
        else:
            _start = None

        if self._has_stop:
            self._stop = cy_operator_index(stop)
            _stop = self._stop
        else:
            _stop = None

        if self._has_step:
            self._step = cy_operator_index(step)
            if self._step == 0:
                raise ValueError("slice step cannot be zero")
            _step = self._step
        else:
            _step = None

        self.args = (_start, _stop, _step)

    @property
    def raw(self):
        return slice(self.start, self.stop, self.step)

    @property
    def start(self):
        return self.args[0]

    @property
    def stop(self):
        return self.args[1]

    @property
    def step(self):
        return self.args[2]

    def __repr__(self):
        return f"SimpleSliceCython{self.args}"

    def __eq__(self, other):
        if not isinstance(other, SimpleSliceCython):
            return False
        return self.args == other.args

    def __ne__(self, other):
        return not self == other



@cython.cdivision(True)
def _reduce_all_int(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step):

    assert step != 0

    if start >= 0 and stop >= 0 or start < 0 and stop < 0:
        if step > 0:
            if stop <= start:
                start, stop, step = 0, 0, 1
            elif start >= 0 and start + step >= stop:
                # Indexes 1 element. Start has to be >= 0 because a
                # negative start could be less than the size of the
                # axis, in which case it will clip and the single
                # element will be element 0. We can only do that
                # reduction if we know the shape.

                # Note that returning Integer here is wrong, because
                # slices keep the axis and integers remove it.
                stop, step = start + 1, 1
            elif start < 0 and start + step > stop:
                # The exception is this case where stop is already
                # start + 1.
                step = stop - start
            if start >= 0:
                stop -= (stop - start - 1) % step
        else: # step < 0
            if stop >= start:
                start, stop, step = 0, 0, 1
            elif start < 0 and start + step <= stop:
                if start < -1:
                    stop, step = start + 1, 1
                else: # start == -1
                    stop, step = start - 1, -1
            elif stop == start - 1:
                stop, step = start + 1, 1
            elif start >= 0 and start + step <= stop:
                # Indexes 0 or 1 elements. We can't change stop
                # because start might clip to a smaller true start if
                # the axis is smaller than it, and increasing stop
                # would prevent it from indexing an element in that
                # case. The exception is the case right before this
                # one (stop == start - 1). In that case start cannot
                # clip past the stop (it always indexes the same one
                # element in the cases where it indexes anything at
                # all).
                step = stop - start
            if start < 0:
                stop -= (stop - start + 1) % step
    elif start >= 0 and stop < 0 and step < 0 and (start < -step or
                                                   -stop - 1 < -step):
        if stop == -1:
            start, stop, step = 0, 0, 1
        else:
            step = max(-start - 1, stop + 1)
    elif start < 0 and stop == 0 and step > 0:
        start, stop, step = 0, 0, 1
    elif start < 0 and stop >= 0 and step >= min(-start, stop):
        step = min(-start, stop)
        if start == -1 or stop == 1:
            # Can only index 0 or 1 elements. We can either pick a
            # version with positive start and negative step, or
            # negative start and positive step. We prefer the former
            # as it matches what is done for reduce() with a shape
            # (start is always nonnegative).
            assert step == 1
            start, stop, step = stop - 1, start - 1, -1

    return start, stop, step



@cython.cdivision(True)
def _reduce_with_size(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step, Py_ssize_t size):

    assert step != 0

    if stop < -size:
        stop = -size - 1

    if size == 0:
        start, stop, step = 0, 0, 1
    elif step > 0:
        # start cannot be None
        if start < 0:
            start = size + start
        if start < 0:
            start = 0
        if start >= size:
            start, stop, step = 0, 0, 1

        if stop < 0:
            stop = size + stop
            if stop < 0:
                stop = 0
        else:
            stop = min(stop, size)
        stop -= (stop - start - 1) % step

        if stop - start == 1:
            # Indexes 1 element.
            step = 1
        elif stop - start <= 0:
            start, stop, step = 0, 0, 1
    else:
        if start < 0:
            if start >= -size:
                start = size + start
            else:
                start, stop = 0, 0
        if start >= 0:
            start = min(size - 1, start)

        if -size <= stop < 0:
            stop += size

        if stop >= 0:
            if start - stop == 1:
                stop, step = start + 1, 1
            elif start - stop <= 0:
                start, stop, step = 0, 0, 1
            else:
                stop += (start - stop - 1) % -step

        # start >= 0
        if (stop < 0 and start - size - stop <= -step
            or stop >= 0 and start - stop <= -step):
            stop, step = start + 1, 1
        if stop < 0 and start % step != 0:
            # At this point, negative stop is only necessary to index the
            # first element. If that element isn't actually indexed, we
            # prefer a nonnegative stop. Otherwise, stop will be -size - 1.
            stop = start % -step - 1

    return start, stop, step
