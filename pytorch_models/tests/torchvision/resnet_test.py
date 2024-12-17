# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://pytorch.org/vision/stable/models/resnet.html

import pytest

from ...utils import *


# https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
def test_resnet50(pytorch_import_and_iree_compile):
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    pytorch_import_and_iree_compile(model)
