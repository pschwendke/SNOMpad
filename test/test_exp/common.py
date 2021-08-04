from trion.analysis.signals import Scan, Demodulation, Detector
from trion.analysis.experiment import Experiment

exp_configs = (
    Experiment(Scan.point, Demodulation.shd, Detector.single),
    Experiment(Scan.point, Demodulation.shd, Detector.dual),
)

n_samples = [
    10, 23, 1000
]