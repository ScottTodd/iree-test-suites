def test_case_0(helpers):
    helpers.test_compile_and_run(__file__, "model.mlir", "run_module_io_flags.txt")


def test_case_1(iree_compile_run_config):
    print("test_case_1")
    print(iree_compile_run_config.get("config_name"))


def test_case_2(test_iree_compile_and_run):
    test_iree_compile_and_run(__file__, "model.mlir", "run_module_io_flags.txt")
