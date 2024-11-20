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
    BuildAction,
    BuildContext,
    BuildFile,
    BuildFileLike,
)

ONNX_PACKAGE_DIR = Path(onnx.__file__).parent
ONNX_NODE_TESTS_ROOT = ONNX_PACKAGE_DIR / "backend/test/data/node"


def onnx_op_test_case(
    *,
    name: str,
    source: BuildFileLike,
) -> list[BuildFileLike]:
    # This imports one 'test_[name]' subfolder from this:
    #
    #   test_[name]/
    #     model.onnx
    #     test_data_set_0/
    #       input_0.pb
    #       output_0.pb
    #
    # to this:
    #
    #   imported_dir_path/...
    #     test_[name]/
    #       model.mlir  (torch-mlir)
    #       input_0.bin
    #       output_0.bin
    #       run_module_io_flags.txt  (flagfile with --input=input_0.bin, --expected_output=)

    context = BuildContext.current()

    # Import from .onnx to .mlir.
    onnx_model_path = Path(source.path) / "model.onnx"
    onnx_model = context.allocate_file(str(onnx_model_path))
    onnx_import(
        name=name,
        source=onnx_model,
    )

    # TODO(scotttodd): input_[0-9]+.bin
    # TODO(scotttodd): output_[0-9]+.bin
    # TODO(scotttodd): run_module_io_flags.txt

    onnx_input_0_path = Path(source.path) / "test_data_set_0" / "onnx_input_0.pb"
    onnx_input_0 = context.allocate_file(str(onnx_input_0_path))
    onnx_output_0_path = Path(source.path) / "test_data_set_0" / "onnx_output_0.pb"
    onnx_output_0 = context.allocate_file(str(onnx_output_0_path))

    # iree_input_0 = context.allocate_file()

    # ConvertInputsAndOutputs()
    return [name]


class ConvertInputsAndOutputs(BuildAction):
    def __init__(
        self,
        input_proto_files: list[BuildFile],
        output_bin_files: list[BuildFile],
        output_flag_file: BuildFile,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_proto_files = input_proto_files
        self.output_bin_files = output_bin_files
        self.output_flag_file = output_flag_file
        for file in self.input_proto_files:
            self.deps.add(file)
        for file in self.output_bin_files:
            file.deps.add(self)
        output_flag_file.deps.add(self)

    def _invoke(self):
        # input_path = self.input_file.get_fs_path()
        # output_path = self.output_file.get_fs_path()

        # original_model = onnx.load_model(str(input_path))
        # converted_model = onnx.version_converter.convert_version(original_model, 17)
        # onnx.save(converted_model, str(output_path))
        pass


def create_test_case(context, test_folder_name):
    # folder_name = "test_" + test_name
    test_name = test_folder_name[5:]  # Skip over the "test_" prefix.
    imported_name = (
        "onnx/node/generated/" + test_folder_name + "/" + test_name + ".mlir"
    )

    test_folder_path = Path(ONNX_NODE_TESTS_ROOT) / test_folder_name
    test_folder = context.allocate_file(str(test_folder_path))
    # onnx_path = Path(ONNX_NODE_TESTS_ROOT) / test_folder_name / "model.onnx"
    # onnx_file = context.allocate_file(str(onnx_path))

    # onnx_import(
    #     name=imported_name,
    #     source=onnx_file,
    # )
    return onnx_op_test_case(name=imported_name, source=test_folder)

    # return imported_name


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
    all_artifacts.extend(create_test_case(context, "test_add"))
    print("------------------------------")
    print("all_artifacts:")
    print(all_artifacts)
    print("------------------------------")
    # all_artifacts.append(create_test_case(context, "mul"))
    return all_artifacts


if __name__ == "__main__":
    iree_build_main()
