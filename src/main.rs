use cudarc::driver::CudaContext;

use inferno::{Linear, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = CudaContext::new(0)?;
    let stream = dev.default_stream();

    // Matrix dimensions
    let (batch_size, in_features, out_features) = (2, 4, 3);

    // Column-major data initialization
    let x_host = vec![
        1.0, 5.0, // Column 0
        2.0, 6.0, // Column 1
        3.0, 7.0, // Column 2
        4.0, 8.0, // Column 3
    ];

    let w_host = vec![
        // Column-major weight matrix (4x3)
        0.1, 0.4, 0.7, 1.0, // Column 0
        0.2, 0.5, 0.8, 1.1, // Column 1
        0.3, 0.6, 0.9, 1.2, // Column 2
    ];

    // Create tensors with explicit column-major strides
    let input = Tensor::from_vec(stream.clone(), &x_host, vec![batch_size, in_features])?;

    let weight = Tensor::from_vec(stream.clone(), &w_host, vec![in_features, out_features])?;

    let linear = Linear::new(weight, None);
    let output = linear.forward(&input)?;

    // Print final results in column-major format
    let out_host = stream.memcpy_dtov(&output.view())?;
    println!(
        "\n[RESULT] Output ({}x{} column-major):",
        batch_size, out_features
    );
    for col in 0..out_features {
        let start = col * batch_size;
        let end = start + batch_size;
        println!("Col {}: {:?}", col, &out_host[start..end]);
    }

    Ok(())
}
