use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::CudaContext;
use inferno::{Linear, Tensor};

#[test]
fn test_linear_forward() {
    let dev = CudaContext::new(0).unwrap();
    let stream = dev.default_stream();
    let blaslt = CudaBlasLT::new(stream.clone()).unwrap();

    let in_features = 2;
    let out_features = 2;
    let batch_size = 1;

    let x_host = vec![1.0f32, 2.0];
    let w_host = vec![0.5, 0.1, 0.2, 0.3]; // 2x2 weight
    let b_host = vec![0.01, 0.02];

    let input = Tensor::from_host(
        stream.clone(),
        &x_host,
        vec![batch_size, in_features],
        vec![1, batch_size],
    )
    .unwrap();
    let weight = Tensor::from_host(
        stream.clone(),
        &w_host,
        vec![in_features, out_features],
        vec![1, in_features],
    )
    .unwrap();
    let bias =
        Tensor::from_host(stream.clone(), &b_host, vec![1, b_host.len()], vec![1, 1]).unwrap();

    let linear = Linear::new(
        weight.data,
        Some(bias.data),
        in_features,
        out_features,
        blaslt,
    );
    let output = linear.forward(&input).unwrap();
    let out_host = stream.memcpy_dtov(&output.data).unwrap();

    assert_eq!(out_host.len(), out_features);
    println!("Linear test output: {:?}", out_host);
}
