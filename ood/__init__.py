"""Feature-space out-of-distribution detection utilities."""

from ood.mahalanobis import (
    CalibrationError,
    MahalanobisConfig,
    MahalanobisDetector,
    calibrate_d95,
    fit_mahalanobis_detector,
    ledoit_wolf_covariance,
    load_detector,
    normalized_ood_score,
    save_detector,
)
from ood.contrastive import (
    ContrastiveTrainingConfig,
    load_feature_extractor,
    nt_xent_loss,
    train_contrastive_feature_extractor,
)
from ood.feature_extractor import OODFeatureConfig, OODFeatureExtractor, build_ood_input
from ood.evaluation import evaluate_ood_detector

__all__ = [
    "CalibrationError",
    "MahalanobisConfig",
    "MahalanobisDetector",
    "calibrate_d95",
    "fit_mahalanobis_detector",
    "normalized_ood_score",
    "ledoit_wolf_covariance",
    "save_detector",
    "load_detector",
    "ContrastiveTrainingConfig",
    "load_feature_extractor",
    "nt_xent_loss",
    "train_contrastive_feature_extractor",
    "OODFeatureConfig",
    "OODFeatureExtractor",
    "build_ood_input",
    "evaluate_ood_detector",
]
