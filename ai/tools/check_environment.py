"""Read-only report of local inference runtime and devices; no dataset access."""
import json
from pathlib import Path
import platform


def main():
    import cv2
    import openvino as ov
    import openvino_genai
    import psutil

    core = ov.Core()
    devices = {}
    for device in core.available_devices:
        devices[device] = str(core.get_property(device, "FULL_DEVICE_NAME"))
    report = {
        "python": platform.python_version(), "openvino": ov.__version__,
        "openvino_genai": openvino_genai.__version__, "opencv": cv2.__version__,
        "devices": devices, "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
    }
    output = Path("ai/local/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
