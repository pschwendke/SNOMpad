from trion.analysis.signals import Experiment, Scan, Acquisition, Detector

exp_configs = (
    Experiment(Scan.point, Acquisition.shd, Detector.single),
    Experiment(Scan.point, Acquisition.shd, Detector.dual),
)

n_samples = [
    10, 23, 1000
]