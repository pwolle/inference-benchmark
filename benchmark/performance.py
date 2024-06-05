import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def time_function(
    f: Callable[..., Any],
    /,
    *,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> float:
    """
    Computes the time taken to execute a function f a single time in seconds.

    Parameters
    ---
    f: Callable[[], Any]
        The function to time. It should take no arguments.
        The return value is ignored.

    args: Sequence[Any] | None
        The positional arguments to pass to f, if None, an empty list is used.

    kwargs: Mapping[str, Any] | None
        The keyword arguments to pass to f, if None, an empty dictionary is
        used.

    Returns
    ---
    float
        The time taken to execute f in seconds.
    """
    args = args or []
    kwargs = kwargs or {}

    s = time.perf_counter_ns()
    f(*args, **kwargs)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def time_function_average(
    f: Callable[[], Any],
    /,
    *,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    skip_first: bool = True,
    time_threshold: None | float = None,
    rerr_threshold: None | float = 1e-2,
) -> tuple[float, list[float]]:
    """
    Computes the average time taken to execute a function f in seconds.

    Parameters
    ---
    f: Callable[[], Any]
        The function to time. It should take no arguments.
        The return value is ignored.

    args: Sequence[Any] | None
        The positional arguments to pass to f, if None, an empty list is used.

    kwargs: Mapping[str, Any] | None
        The keyword arguments to pass to f, if None, an empty dictionary is
        used.

    time_threshold: None | float
        How long the test should maximally run in seconds.

    rerr_threshold: None | float
        The relative standard error of the mean is below this threshold,
        the test is stopped.

    Returns
    ---
    float
        The average time taken to execute f in seconds.
    """
    if not (time_threshold is None or time_threshold > 0):
        error = "time_threshold must be positive"
        raise ValueError(error)

    if not (rerr_threshold is None or rerr_threshold > 0):
        error = "rerr_threshold must be positive"
        raise ValueError(error)

    if time_threshold is None and rerr_threshold is None:
        error = "Either time_threshold or rerr_threshold must be set"
        raise ValueError(error)

    args = args or []
    kwargs = kwargs or {}

    if skip_first:
        f(*args, **kwargs)

    s = time.perf_counter()
    t = []

    while True:
        t.append(time_function(f, args=args, kwargs=kwargs))
        c = time.perf_counter() - s

        if time_threshold is not None and c > time_threshold:
            break

        # do not trust the relative standard deviation for n < 4 (heuristic)
        if len(t) < 4 or rerr_threshold is None:
            continue

        rerr = np.std(t) / np.sqrt(len(t)) / np.mean(t)
        if rerr < rerr_threshold:
            break

    return sum(t) / len(t), t


def main():
    def f():
        time.sleep(0.1)

    print(time_function(f))
    print(time_function_average(f, time_threshold=None, rerr_threshold=1e-2))


if __name__ == "__main__":
    main()
