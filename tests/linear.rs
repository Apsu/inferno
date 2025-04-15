use cudarc::driver::CudaContext;
use inferno::{Linear, Tensor};

#[test]
fn test_linear_forward() {
    let dev = CudaContext::new(0).unwrap();
    let stream = dev.default_stream();

    let b = 2;
    let m = 3;
    let n = 4;
    let k = 5;

    let x_host = (0..b * m * k).map(|x| x as f32).collect::<Vec<_>>();
    let w_host = (0..b * n * k).map(|x| x as f32).collect::<Vec<_>>();
    let b_host = (0..b * m).map(|x| x as f32).collect::<Vec<_>>();

    let input = Tensor::from_vec(stream.clone(), &x_host, vec![b, m, k]).unwrap();
    let weight = Tensor::from_vec(stream.clone(), &w_host, vec![b, n, k]).unwrap();
    let bias = Tensor::from_vec(stream.clone(), &b_host, vec![b, m]).unwrap();

    let linear = Linear::new(weight, Some(bias));
    let output = linear.forward(&input).unwrap();
    let out_host = output.to_vec().unwrap();

    assert_eq!(out_host.len(), b * m * n);
    assert_eq!(output.shape().dims3().unwrap(), (b, m, n));
    assert_eq!(
        out_host,
        [
            30f32, 80.0, 130.0, 180.0, 81.0, 256.0, 431.0, 606.0, 132.0, 432.0, 732.0, 1032.0,
            1883.0, 2308.0, 2733.0, 3158.0, 2434.0, 2984.0, 3534.0, 4084.0, 2985.0, 3660.0, 4335.0,
            5010.0
        ]
    );
}
