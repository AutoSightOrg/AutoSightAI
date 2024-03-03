# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch.ao.quantization
import warnings
from typing import Union
from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class QuantizedYOLO(YOLO):
    _quantized = False
    qconfig = None

    def __init__(self, model: Union[str, Path] = 'yolov8n-quant-detect.yaml', task='detect',
                 quantize=True, qconfig='x86', calibrate=None, quant_weights=None, verbose=False) -> None:
        super().__init__(model, task, verbose)
        if quantize:
            self.quant(qconfig, calibrate, quant_weights)

    def save(self, filename='model.pt'):
        torch.save(self.state_dict(), filename) if self._quantized else super().save(filename)

    def load(self, filename='model.pt'):
        self.load_state_dict(torch.load(filename)) if self._quantized else super().load(filename)

    def _fuse_for_quantization(self):
        assert not self._quantized
        self._check_is_pytorch_model()
        self.model.fuse_for_quantization()

    def _train(self, mode: bool = True):
        assert isinstance(mode, bool)
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def _eval(self):
        return self._train(False)

    def quant(self, qconfig='x86', calibrate=None, quant_weights=None):
        assert not self._quantized
        self.to('cpu')
        self._eval()
        self._fuse_for_quantization()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qconfig)

        with warnings.catch_warnings(action="ignore"):
            torch.backends.quantized.engine = qconfig
            torch.ao.quantization.prepare(self, inplace=True)

            if calibrate is not None and quant_weights is None:
                self.predict(calibrate)

            torch.ao.quantization.convert(self, inplace=True)
            if quant_weights is not None:
                self.load_state_dict(torch.load(quant_weights))
        self._quantized = True
