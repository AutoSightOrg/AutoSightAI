import sys
import torch
from ultralytics import YOLO


def main() -> int:
    model = YOLO('yolov8n.yaml').to('cpu')

    model.train(epochs=1)

    model.s_eval()
    model.fuse_for_quantization()

    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.ao.quantization.prepare(model, inplace=True)

    model.predict('../datasets/coco8/images/train/000000000009.jpg')

    torch.ao.quantization.convert(model, inplace=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
