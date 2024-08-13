import pytest

compiled_model = ""


@pytest.mark.dependency()
def test_compile(test_iree_compile):
    global compiled_model
    compiled_model = test_iree_compile(__file__, "model.mlir")


@pytest.mark.dependency(depends=["test_compile"])
def test_run(test_iree_run_module):
    global compiled_model
    test_iree_run_module(__file__, compiled_model, "run_module_io_flags.txt")
