import benchmark.models.vae_keras as vae_keras
import ROOT
import tempfile

TMVA = ROOT.TMVA
TFile = ROOT.TFile

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

print()
