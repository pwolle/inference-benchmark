import os

# set environment variable to make sure OMP runs with oen thread
os.environ["OMP_NUM_THREADS"] = "1"

# also for flexiblas
os.environ["FLEXIBLAS_NUM_THREADS"] = "1"


import ROOT

from . import performance
from .utils_keras import _synchronize_after

import tensorflow as tf

ROOT.TMVA.PyMethodBase.PyInitialize()


def from_keras(model, name="model"):
    model.save(f"{name}.h5")
    model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(f"{name}.h5")

    model.Generate(ROOT.TMVA.Experimental.SOFIE.Options.kDefault, 1024)
    model.OutputGenerated(f"{name}.hxx")

    ROOT.gInterpreter.Declare(f'#include "{name}.hxx"')

    session = getattr(ROOT, f"TMVA_SOFIE_{name}").Session()
    return session.infer


def time_model_inference(model, x):

    x = x.numpy()
    with tf.device("cpu"):
        model = from_keras(model)
        # model = _synchronize_after(model)

        return performance.time_function_average(
            model,
            skip_first=True,
            args=(x,),
        )
