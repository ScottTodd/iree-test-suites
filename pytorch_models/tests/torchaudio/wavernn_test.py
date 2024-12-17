# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://pytorch.org/audio/stable/generated/torchaudio.models.WaveRNN.html

import pytest

from ...utils import *


@pytest.mark.skip("Test case not yet validated")
def test_wavernn(pytorch_import_and_iree_compile):
    from torchaudio.models import WaveRNN

    kernel_size = 5
    n_freq = 128
    hop_length = 200
    model = WaveRNN(
        upsample_scales=[5, 5, 8],
        n_classes=512,
        hop_length=hop_length,
        kernel_size=kernel_size,
        n_freq=n_freq,
    )
    n_batch = 1
    n_time = 10
    waveform = torch.randn(
        (n_batch, 1, (n_time - kernel_size + 1) * hop_length), dtype=torch.float32
    )
    specgram = torch.randn((n_batch, 1, n_freq, n_time))
    pytorch_import_and_iree_compile(model, (waveform, specgram))
