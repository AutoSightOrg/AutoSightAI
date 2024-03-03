import sys
from ultralytics import QuantizedYOLO


def main() -> int:
    model = QuantizedYOLO(
        model='../models/quant-ready/model.pt',
        qconfig='x86',
        quant_weights='../models/quantized/model.pt'
    )

    results = model.predict(source='0', stream=True, conf=0.8)
    for result in results:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
