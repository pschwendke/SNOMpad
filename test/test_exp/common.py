from trion.analysis.signals import Scan, Acquisition, Detector
from trion.analysis.experiment import Experiment

exp_configs = (
    Experiment(Scan.point, Acquisition.shd, Detector.single),
    Experiment(Scan.point, Acquisition.shd, Detector.dual),
)

n_samples = [
    10, 23, 1000
]