# External dependencies and assets

This source repository uses Ultralytics, PyTorch/torchvision, OpenCV, NumPy, Pillow and PyYAML. Dependency software and model artifacts keep their upstream terms. This document does not grant rights to third-party material.

- [Ultralytics licensing](https://www.ultralytics.com/license)
- [PyTorch repository license](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [OpenCV license](https://github.com/opencv/opencv/blob/4.x/LICENSE)
- [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [PyYAML license](https://github.com/yaml/pyyaml/blob/main/LICENSE)

The destination repository already contains an owner-selected MIT license. That license does not relicense external dependencies or their model artifacts. Check applicable upstream terms before distributing a derivative product or model.

Datasets, recorded videos, annotation source records, generated external-image galleries and pretrained weights are not included in Git. The project-trained experimental opening checkpoint is distributed as a separate release asset and inherits applicable upstream model terms. Source names such as MonoRaceGate, DeepDoors and door/window datasets identify local research provenance; they are not redistribution permissions. Screenshots can contain third-party stock images, screen content or people, so test galleries remain local.

The full application imports RajGoyal01/NIDAR at the pinned revision and adapts its `main.py` orchestration as `full_runtime.py`, with attribution in that file. Upstream algorithm code remains in the downloaded checkout. OSNet and its bundled source retain their upstream notices. The team MIT license does not grant additional rights to upstream materials.
