# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://huggingface.co/docs/transformers/en/model_doc/opt

import pytest

from ...utils import *


# https://huggingface.co/facebook/opt-125M
def test_opt_125M(pytorch_import_and_iree_compile):
    from transformers import OPTForCausalLM, AutoTokenizer

    test_modelname = "facebook/opt-125M"
    tokenizer = AutoTokenizer.from_pretrained(test_modelname)
    model = OPTForCausalLM.from_pretrained(
        test_modelname,
        num_labels=2,
        # output_attentions=False,
        # output_hidden_states=False,
        # torchscript=True,
    )
    # TODO(scotttodd): try with each attn_implementation option
