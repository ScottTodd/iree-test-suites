# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import torch.nn as nn

from .utils import *

logger = logging.getLogger(__name__)


def pytorch_import_and_iree_compile_fn(model: nn.Module):
    mlir_path = import_torch_model_to_mlir(model)
    logger.info(f"mlir_path: {mlir_path}")

    # vmfb_path = compile_mlir_with_iree(
    #     mlir_path,
    #     "cpu",
    #     [
    #         "--iree-hal-target-backends=llvm-cpu",
    #         "--iree-llvmcpu-target-cpu=host",
    #     ],
    # )
    # logger.info(f"vmfb_path: {vmfb_path}")

    # TODO(#5): test iree-run-module success and numerics
    #   * On Linux...
    #     * Determine interface via ai-edge-litert / tflite-runtime
    #     * Produce test inputs, save to .bin for IREE
    #     * Produce golden test outputs, save to .bin for IREE
    #   * Run with inputs and expected outputs


@pytest.fixture
def pytorch_import_and_iree_compile():
    return pytorch_import_and_iree_compile_fn
