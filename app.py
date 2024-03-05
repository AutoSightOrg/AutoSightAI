import sys
import torch

from ultralytics import QuantizedYOLO


def main() -> int:
    torch.backends.quantized.engine = 'x86'

    model = QuantizedYOLO(model='../models/quant-ready/model.pt')
    model.quant(
        qconfig='x86',
        calibrate='../full_dataset_v0.2/val/images/'
    )
    model.export(format='torchscript', imgsz=(640, 640))

    jit_model = QuantizedYOLO(model='../models/quant-ready/model.torchscript')
    results = jit_model('../test-videos/test2.mp4', stream=True, conf=0.8)
    for _ in results:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
