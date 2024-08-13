def test_case_0(test_iree_compile_and_run):
    test_iree_compile_and_run(__file__, "model.mlir", "run_module_io_flags.txt")
