import time
from typing import Callable, Any


def time_function(f: Callable[[], Any]):
    s = time.perf_counter_ns()
    f()
    e = time.perf_counter_ns()
    return e - s
    