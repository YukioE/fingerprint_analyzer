from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules


def _iter_detector_functions():
    package_dir = Path(__file__).resolve().parent
    for module_info in sorted(iter_modules([str(package_dir)]), key=lambda item: item.name):
        if module_info.name.startswith("_"):
            continue

        module = import_module(f"{__package__}.{module_info.name}")
        detector = getattr(module, "detect", None)
        if callable(detector):
            yield detector


DETECTORS = tuple(_iter_detector_functions())
