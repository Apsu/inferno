use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::cublaslt::CudaBlasLT;

use inferno::{Linear, Tensor};

fn main() -> anyhow::Result<()> {
    // Create CUDA context
    let dev = CudaContext::new(0)?;
    let stream = dev.default_stream();
    let blaslt = CudaBlasLT::new(stream.clone())?;

    let in_features = 4;
    let out_features = 3;
    let batch_size = 2;

    // Dummy inputs (row major)
    let x_host: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // sample 1
        5.0, 6.0, 7.0, 8.0, // sample 2
    ];
    let w_host: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4, // out 1
        0.5, 0.6, 0.7, 0.8, // out 2
        0.9, 1.0, 1.1, 1.2, // out 3
    ];
    let b_host: Vec<f32> = vec![0.01, 0.02, 0.03];

    // Transfer to device
    let x_dev = unsafe {
        let mut dst: CudaSlice<f32> = stream.alloc(x_host.len())?;
        stream.memcpy_htod(&x_host, &mut dst)?;
        dst
    };
    let w_tensor = Tensor::from_host(stream.clone(), &w_host, vec![out_features, in_features], vec![in_features, 1])?;
    let b_tensor = Tensor::from_host(stream.clone(), &b_host, vec![out_features], vec![1])?;

    let input = Tensor::new(x_dev, vec![batch_size, in_features], vec![in_features, 1]);

    let linear = Linear::new(w_tensor.data, Some(b_tensor.data), in_features, out_features, blaslt);
    let output = linear.forward(&input)?;

    let out_host = stream.memcpy_dtov(&output.data)?;
    println!("Output: {:?}", &out_host[..]);

    Ok(())
}
