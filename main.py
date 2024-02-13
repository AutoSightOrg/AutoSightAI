import sys
from ultralytics import YOLO


def main() -> int:
    model = YOLO('yolov8n.yaml')
    return 0


if __name__ == '__main__':
    sys.exit(main())
