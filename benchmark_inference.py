import time
from typing import Any, Callable

import numpy as np


def time_function(
    f: Callable[..., Any],
    /,
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

    s = time.perf_counter_ns()
    f()
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def time_function_average(
    f: Callable[[], Any],
    /,
    *,
    skip_first: bool = True,
    time_threshold: None | float = None,
    rerr_threshold: None | float = 1e-1,
) -> list[float]:
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

    if skip_first:
        f()

    s = time.perf_counter()
    t = []

    while True:
        t.append(time_function(f))
        c = time.perf_counter() - s

        if time_threshold is not None and c > time_threshold:
            break

        # do not trust the relative standard deviation for n < 4 (heuristic)
        if len(t) < 4 or rerr_threshold is None:
            continue

        rerr = np.std(t) / np.sqrt(len(t)) / np.mean(t)
        if rerr < rerr_threshold:
            break

    return t


def get_input_shape_dir(onnx_path: str):
    import onnx

    model = onnx.load(onnx_path)
    shapes = {}

    for input_ in model.graph.input:
        shape = [d.dim_value for d in input_.type.tensor_type.shape.dim]
        shapes[input_.name] = shape

    return shapes


def get_input_shape_list(onnx_path: str):
    import onnx

    model = onnx.load(onnx_path)
    shapes = []

    for input_ in model.graph.input:
        shape = [d.dim_value for d in input_.type.tensor_type.shape.dim]
        shapes.append(shape)

    return shapes


def make_dynamic(onnx_path: str):
    import onnx

    model = onnx.load(onnx_path)
    sym_batch_dim = "dynamic_batch_size"

    inputs = model.graph.input
    for input_ in inputs:
        dim1 = input_.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim

    onnx.save(model, onnx_path)


def benchmark_onnxruntime(
    onnx_path: str,
    batch_size: int,
    device: str,
) -> None:
    import numpy as np
    import onnxruntime

    opts = onnxruntime.SessionOptions()

    if device == "singlethreaded":
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

    provider = {
        "singlethreaded": "CPUExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "gpu": "CUDAExecutionProvider",
    }[device]

    model = onnxruntime.InferenceSession(
        onnx_path,
        providers=[provider],
        sess_options=opts,
    )

    input_shapes = get_input_shape_dir(onnx_path)
    input_ = {}

    for name, shape in input_shapes.items():
        shape[0] = batch_size
        input_[name] = np.random.randn(*shape).astype(np.float32)

    ts = time_function_average(
        lambda: model.run(None, input_),
        skip_first=True,
    )
    return ts


def benchmark_torch(
    onnx_path: str,
    batch_size: int,
    device: str,
) -> None:
    import torch

    if device == "singlethreaded":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    import onnx2torch

    model = onnx2torch.convert(onnx_path)
    model = model.eval()
    model = torch.jit.script(model)

    def model_(x):
        y = model(x)
        if device == "singlethreaded" or device == "cpu":
            torch.cpu.synchronize()

        if device == "gpu":
            torch.cuda.synchronize()

        return y

    torch_device = {
        "singlethreaded": "cpu",
        "cpu": "cpu",
        "gpu": "cuda",
    }[device]
    model = model.to(torch_device)

    input_shapes = get_input_shape_list(onnx_path)
    input_ = []

    for shape in input_shapes:
        shape[0] = batch_size
        input_.append(torch.randn(*shape).to(torch_device))

    ts = time_function_average(
        lambda: model_(*input_),
        skip_first=True,
    )
    return ts


def benchmark_keras(
    onnx_path: str,
    batch_size: int,
    device: str,
):
    import tensorflow as tf

    if device == "singlethreaded":
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    if device != "gpu":
        tf.config.experimental.set_visible_devices([], device_type="gpu")

    import onnx2keras
    import onnx

    input_shapes = get_input_shape_dir(onnx_path)

    model = onnx.load(onnx_path)
    model = onnx2keras.onnx_to_keras(model, list(input_shapes.keys()))
    model = tf.function(model, jit_compile=True)

    def model_(x):
        y = model(x)
        tf.test.experimental.sync_devices()
        return y

    input_ = {}
    for name, shape in input_shapes.items():
        shape[0] = batch_size
        input_[name] = np.random.randn(*shape).astype(np.float32)

    ts = time_function_average(
        lambda: model_(input_),
        skip_first=True,
    )
    return ts


def benchmark_sofie(
    onnx_path: str,
    batch_size: int,
    device: str,
):
    if device == "singlethreaded":
        import os

        # set environment variable to make sure OMP runs with oen thread
        os.environ["OMP_NUM_THREADS"] = "1"

        # also for flexiblas
        os.environ["FLEXIBLAS_NUM_THREADS"] = "1"

    import ROOT  # type: ignore
    import onnx2keras
    import onnx

    ROOT.TMVA.PyMethodBase.PyInitialize()

    input_shapes = get_input_shape_dir(onnx_path)

    model = onnx.load(onnx_path)
    model = onnx2keras.onnx_to_keras(model, list(input_shapes.keys()))
    model.save("model.h5")

    model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("model.h5", batch_size)

    model.Generate()
    model.OutputGenerated("model.hxx")

    ROOT.gInterpreter.Declare('#include "model.hxx"')

    session = getattr(ROOT, "TMVA_SOFIE_model").Session()

    input_ = []

    for _, shape in input_shapes.items():
        shape[0] = batch_size
        input_.append(np.random.randn(*shape).astype(np.float32))

    ts = time_function_average(
        lambda: session.infer(*input_),
        skip_first=True,
    )
    return ts


benchmarks = {
    "onnxruntime": benchmark_onnxruntime,
    "torch": benchmark_torch,
    "keras": benchmark_keras,
    "sofie": benchmark_sofie,
}


def main():
    import argparse
    import numpy as np
    import pandas as pd
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="model.onnx",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="singlethreaded",
        choices=["singlethreaded", "cpu", "gpu"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sofie",
        choices=["torch", "onnxruntime", "keras", "sofie"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
    )
    args = parser.parse_args()

    make_dynamic(args.onnx_path)
    ts = benchmarks[args.backend](
        args.onnx_path,
        args.batch_size,
        args.device,
    )

    t_avg = np.mean(ts)
    t_std = np.std(ts) / np.sqrt(len(ts))

    df = [
        {
            "backend": args.backend,
            "device": args.device,
            "batch_size": args.batch_size,
            "time": t_avg,
            "std": t_std,
        }
    ]

    df = pd.DataFrame(df)

    if os.path.exists(args.output):
        df.to_csv(args.output, mode="a", header=False)

    else:
        df.to_csv(args.output)


if __name__ == "__main__":
    main()
