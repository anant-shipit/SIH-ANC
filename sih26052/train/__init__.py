from .dataset import SpeechEnhancementDataset

__all__ = ["SpeechEnhancementDataset"]

# The remaining modules depend on torch, which is intentionally absent on the
# Raspberry Pi runtime. Keep them as optional exports so that importing this
# package (e.g. for the numpy-only dataset) never hard-fails without torch.
try:
    from .loss import CombinedLoss, CompressedSpectralLoss, SISNRLoss
    from .select_checkpoint import select_best_checkpoint
    from .train import evaluate, forward_gtcrn, load_gtcrn, main as train_main
    from .validate import check_catastrophic_forgetting, validate_epoch

    __all__ += [
        "CombinedLoss",
        "CompressedSpectralLoss",
        "SISNRLoss",
        "select_best_checkpoint",
        "evaluate",
        "forward_gtcrn",
        "load_gtcrn",
        "train_main",
        "check_catastrophic_forgetting",
        "validate_epoch",
    ]
except ImportError:
    pass
