import os
import tempfile

import tensorflow as tf

import benchmark.models.vae_keras as vae_keras

vae, x = vae_keras.get_vae_with_inputs(batch_size=32, device="cpu")

import ROOT

ROOT.TMVA.PyMethodBase.PyInitialize()

with tempfile.NamedTemporaryFile(suffix=".h5", dir=".") as f:
    vae.save(f.name)
    model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(f.name)

    model.Generate()
    header_name = f.name.replace(".h5", ".hxx")

    model.OutputGenerated(header_name)
    model.PrintGenerated()

ROOT.gInterpreter.Declare(f'#include "{header_name}"')
# print(os.path.relpath(header_name))
# session = ROOT.TMVA_SOFIE_Higgs_trained_model.Session()
session = getattr(
    ROOT, f"TMVA_SOFIE_{os.path.relpath(header_name).replace('.hxx', '')}"
).Session()

x = x.numpy()

for i in range(0, 32):
    result = session.infer(x[i, :])
    print(result)
    break
