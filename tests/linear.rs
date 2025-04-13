use cudarc::driver::CudaContext;
use cudarc::cublaslt::CudaBlasLT;
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

    let x_dev = {
        let mut dst = unsafe { stream.alloc(x_host.len()).unwrap() };
        stream.memcpy_htod(&x_host, &mut dst).unwrap();
        dst
    };
    let w_dev = {
        let mut dst = unsafe { stream.alloc(w_host.len()).unwrap() };
        stream.memcpy_htod(&w_host, &mut dst).unwrap();
        dst
    };
    let b_dev = {
        let mut dst = unsafe { stream.alloc(b_host.len()).unwrap() };
        stream.memcpy_htod(&b_host, &mut dst).unwrap();
        dst
    };

    let input = Tensor::new(x_dev, vec![batch_size, in_features], vec![in_features, 1]);
    let linear = Linear::new(w_dev, Some(b_dev), in_features, out_features, blaslt);
    let output = linear.forward(&input).unwrap();
    let out_host = stream.memcpy_dtov(&output.data).unwrap();

    assert_eq!(out_host.len(), out_features);
    println!("Linear test output: {:?}", out_host);
}
