# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage:
#   python import_onnx_tests_2.py --build --output-dir /tmp/iree_onnx_tests
#   (then later change that to instead output to this directory)

import onnx
from iree.build import *
from import_onnx_tests_utils import *
from pathlib import Path

from iree.build.executor import (
    BuildContext,
    FileNamespace,
)

ONNX_PACKAGE_DIR = Path(onnx.__file__).parent
ONNX_NODE_TESTS_ROOT = ONNX_PACKAGE_DIR / "backend/test/data/node"


def add_test_case(context, test_name):
    folder_name = "test_" + test_name
    onnx_path = Path(ONNX_NODE_TESTS_ROOT) / folder_name / "model.onnx"
    onnx_file = context.allocate_file(str(onnx_path))
    imported_name = "onnx/node/generated/" + folder_name + "/" + test_name + ".mlir"

    onnx_import(
        name=imported_name,
        source=onnx_file,
    )

    return imported_name


@entrypoint(description="Imports all ONNX operator tests")
def import_onnx_op_tests():
    context = BuildContext.current()

    # add_onnx_path = Path(ONNX_NODE_TESTS_ROOT) / "test_add" / "model.onnx"
    # add_onnx_file = context.allocate_file(str(add_onnx_path))
    # mul_onnx_path = Path(ONNX_NODE_TESTS_ROOT) / "test_mul" / "model.onnx"
    # mul_onnx_file = context.allocate_file(str(mul_onnx_path))

    # onnx_import(
    #     name="onnx/node/generated/test_add/add.mlir",
    #     source=add_onnx_file,
    # )

    # onnx_import(
    #     name="onnx/node/generated/test_mul/mul.mlir",
    #     source=mul_onnx_file,
    # )
    # return [
    #     "onnx/node/generated/test_add/add.mlir",
    #     "onnx/node/generated/test_mul/mul.mlir",
    # ]

    all_artifacts = []
    all_artifacts.append(add_test_case(context, "add"))
    all_artifacts.append(add_test_case(context, "mul"))
    return all_artifacts


if __name__ == "__main__":
    iree_build_main()
