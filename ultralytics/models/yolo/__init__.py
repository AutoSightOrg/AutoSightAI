# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, obb, pose, segment

from .model import YOLO, QuantizableYOLO, QuantizationAwareYOLO

__all__ = "classify", "segment", "detect", "pose", "obb", "YOLO", "QuantizableYOLO", "QuantizationAwareYOLO"
