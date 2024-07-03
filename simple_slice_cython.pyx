# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

cimport numpy as cnp
import numpy as np
from cpython cimport PyObject
from libc.stdint cimport int64_t

# Ensure NumPy is initialized
cnp.import_array()

cdef extern from "Python.h":
    int PyIndex_Check(object obj)
    object PyNumber_Index(object obj)
    bint PyBool_Check(object obj)
    int64_t PyLong_AsLongLong(object obj) except? -1

cdef inline int64_t cy_operator_index(object idx) except? -1:
    cdef object result

    if PyBool_Check(idx) or isinstance(idx, np.bool_):
        raise TypeError(f"'{type(idx).__name__}' object cannot be interpreted as an integer")

    if PyIndex_Check(idx):
        result = PyNumber_Index(idx)
        if result is None:
            raise TypeError("'__index__' returned non-int")
        return PyLong_AsLongLong(result)

    if isinstance(idx, np.integer):
        return idx.item()

    result = PyNumber_Index(idx)
    if result is None:
        raise TypeError(f"'{type(idx).__name__}' object cannot be interpreted as an integer")
    return PyLong_AsLongLong(result)

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
