module {
  func.func @test_size_example(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[],si64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Size"(%arg0) : (!torch.vtensor<[2,3],f32>) -> !torch.vtensor<[],si64> 
    return %0 : !torch.vtensor<[],si64>
  }
}

