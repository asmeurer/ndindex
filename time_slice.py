import timeit
import random
from ndindex import Slice
from simple_slice import SimpleSlice
from simple_slice_cython import SimpleSliceCython
from simple_slice_pybind11 import SimpleSlicePybind11

slice_classes = [Slice, SimpleSlice, SimpleSliceCython, SimpleSlicePybind11]

N_RUNS = 1_000_000

def random_value():
    return random.choice([None] + list(range(-100, -1)) + list(range(1, 100)))

def generate_inputs(n=N_RUNS):
    return [(random_value(), random_value(), random_value()) for _ in range(n)]

def benchmark_creation(SliceClass, inputs):
    for start, stop, step in inputs:
        SliceClass(start, stop, step)

def benchmark_args(SliceClass, n=N_RUNS):
    s = SliceClass(0, 100, 2)
    for _ in range(n):
        start, stop, step = s.args

def run_benchmark(name, func, SliceClass, *args):
    time = timeit.timeit(lambda: func(SliceClass, *args), number=1)
    print(f"{name} - {SliceClass.__module__}: {time/N_RUNS} seconds")

def main():
    print("Benchmarking SimpleSlice implementations")
    print("----------------------------------------")

    inputs = generate_inputs()

    print("\nObject Creation:")
    for SliceClass in slice_classes:
        run_benchmark("Creation", benchmark_creation, SliceClass, inputs)

    print("\nArgs Access:")
    for SliceClass in slice_classes:
        run_benchmark("Access", benchmark_args, SliceClass)

if __name__ == "__main__":
    main()
