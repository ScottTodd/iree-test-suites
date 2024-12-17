# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging
import subprocess
import torch
import torch.nn as nn
import iree.turbine.aot as aot

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent

###############################################################################
# Exception types
###############################################################################

# Note: can mark tests as expected to fail at a specific stage with:
# @pytest.mark.xfail(raises=IreeCompileException)


class IreeCompileException(RuntimeError):
    pass


###############################################################################
# Export using iree-turbine (torch.export)
###############################################################################


def import_torch_model_to_mlir(model: nn.Module, inputs: tuple[torch.Tensor]) -> Path:
    # print("Importing model using inputs: ", inputs)
    # TODO(scotttodd): TEST_SUITE_ROOT/artifacts dir like in onnx_models/

    export_output = aot.export(model, args=inputs)

    # This crashes on Windows:
    #   tests/torchvision/resnet_test.py::test_resnet50 Windows fatal exception: access violation
    #
    #   Current thread 0x00003e6c (most recent call first):
    #     File "D:\dev\projects\iree-test-suites\pytorch_models\.venv\Lib\site-packages\iree\turbine\aot\support\ir_utils.py", line 332 in create_tensor_global
    #     File "D:\dev\projects\iree-test-suites\pytorch_models\.venv\Lib\site-packages\iree\turbine\aot\support\procedural\globals.py", line 126 in track

    print(export_output)

    # imported_mlir_path = model_path.with_suffix(".mlirbc")
    # logger.info(f"Importing '{model_path}' to '{imported_mlir_path}'")
    # exec_args = [
    #     "iree-import-tflite",
    #     str(model_path),
    #     "-o",
    #     str(imported_mlir_path),
    # ]
    # ret = subprocess.run(exec_args, capture_output=True)
    # if ret.returncode != 0:
    #     logger.error(f"Import of '{model_path.name}' failed!")
    #     logger.error("iree-import-tflite stdout:")
    #     logger.error(ret.stdout.decode("utf-8"))
    #     logger.error("iree-import-tflite stderr:")
    #     logger.error(ret.stderr.decode("utf-8"))
    #     raise IreeImportTfLiteException(f"  '{model_path.name}' import failed")
    # return imported_mlir_path
    pass


###############################################################################
# IREE compilation and running
###############################################################################


def compile_mlir_with_iree(
    mlir_path: Path, config_name: str, compile_flags: list[str]
) -> Path:
    cwd = THIS_DIR
    iree_module_path = mlir_path.with_name(mlir_path.stem + f"_{config_name}.vmfb")
    compile_args = ["iree-compile", mlir_path]
    compile_args.extend(compile_flags)
    compile_args.extend(["-o", iree_module_path])
    compile_cmd = subprocess.list2cmdline(compile_args)
    logger.info(
        f"Launching compile command:\n"  #
        f"  cd {cwd} && {compile_cmd}"
    )
    ret = subprocess.run(compile_cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Compilation of '{iree_module_path}' failed")
        logger.error("iree-compile stdout:")
        logger.error(ret.stdout.decode("utf-8"))
        logger.error("iree-compile stderr:")
        logger.error(ret.stderr.decode("utf-8"))
        raise IreeCompileException(f"  '{iree_module_path.name}' compile failed")
    return iree_module_path
