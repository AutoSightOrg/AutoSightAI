# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import Union
from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import attempt_load_one_weight, ClassificationModel, DetectionModel, QuantizableDetectionModel, OBBModel, PoseModel, SegmentationModel
from ultralytics.cfg import TASK2DATA
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, RANK, SETTINGS, checks, yaml_load


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


class QuantizableYOLO(YOLO):
    def __init__(self, model: Union[str, Path] = 'yolov8n-quant-detect.yaml', task='detect', verbose=False) -> None:
        super().__init__(model, task, verbose)

    def quant(self, qconfig='x86', calibrate=None):
        self.model.prepare(qconfig=qconfig, is_qat=False)
        if calibrate is not None:
            self.predict(calibrate)
        self.model.convert(is_qat=False)

    @property
    def task_map(self):
        return {
            "detect": {
                "model": QuantizableDetectionModel,
                "trainer": yolo.detect.QuantDetectTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }


class QuantizationAwareYOLO(YOLO):
    def __init__(self, model: Union[str, Path] = 'yolov8n-quant-detect.yaml', task='detect', verbose=False) -> None:
        super().__init__(model, task, verbose)

    def train(self, trainer=None, qat=False, qconfig='x86', **kwargs):
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {"data": DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, "amp": not qat, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            if qat:
                self.trainer.model.prepare(qconfig=qconfig, is_qat=True)
            self.model = self.trainer.model

            if SETTINGS["hub"] is True and not self.session:
                # Create a model in HUB
                try:
                    self.session = self._get_hub_session(self.model_name)
                    if self.session:
                        self.session.create_model(args)
                        # Check model was created
                        if not getattr(self.session.model, "id", None):
                            self.session = None
                except (PermissionError, ModuleNotFoundError):
                    # Ignore PermissionError and ModuleNotFoundError which indicates hub-sdk not installed
                    pass

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train(qat=qat)
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def convert(self):
        self._check_is_pytorch_model()
        self.model.convert(is_qat=True)

    @property
    def task_map(self):
        return {
            "detect": {
                "model": QuantizableDetectionModel,
                "trainer": yolo.detect.QuantDetectTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
