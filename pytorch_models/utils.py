# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn
import logging
import subprocess
from pathlib import Path

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


def import_torch_model_to_mlir(model: nn.Module) -> Path:
    # TODO(scotttodd): TEST_SUITE_ROOT/aritfacts dir like in onnx_models/

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
