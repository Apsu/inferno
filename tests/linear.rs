// inferno/tests/linear.rs

use cudarc::driver::{CudaDevice, CudaStream};
use cudarc::cublaslt::CudaBlasLT;
use inferno::{Linear, Tensor};

#[test]
fn test_linear_forward() {
    let dev = CudaDevice::new(0).unwrap();
    let stream = dev.default_stream();
    let blaslt = CudaBlasLT::new(stream.clone()).unwrap();

    let in_features = 2;
    let out_features = 2;
    let batch_size = 1;

    let x_host = vec![1.0f32, 2.0];
    let w_host = vec![0.5, 0.1, 0.2, 0.3]; // 2x2 weight
    let b_host = vec![0.01, 0.02];

    let x_dev = dev.htod_copy(&x_host).unwrap();
    let w_dev = dev.htod_copy(&w_host).unwrap();
    let b_dev = dev.htod_copy(&b_host).unwrap();

    let input = Tensor::new(x_dev, vec![batch_size, in_features], vec![in_features, 1]);
    let linear = Linear::new(w_dev, Some(b_dev), in_features, out_features, std::sync::Arc::new(blaslt));
    let output = linear.forward(&input).unwrap();
    let out_host = dev.sync_reclaim(output.data).unwrap();

    assert_eq!(out_host.len(), out_features);
    println!("Linear test output: {:?}", out_host);
}
