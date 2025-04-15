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

    let input =
        Tensor::from_host(stream.clone(), &x_host, vec![b, m, k], vec![m * k, k, 1]).unwrap();
    let weight =
        Tensor::from_host(stream.clone(), &w_host, vec![b, n, k], vec![n * k, k, 1]).unwrap();
    let bias = Tensor::from_host(stream.clone(), &b_host, vec![b, m], vec![m, 1]).unwrap();

    let linear = Linear::new(weight, Some(bias));
    let output = linear.forward(&input).unwrap();
    let out_host = stream.memcpy_dtov(&output.view()).unwrap();

    assert_eq!(out_host.len(), b * m * n);
    assert_eq!(output.shape().dims3().unwrap(), (b, m, n));
    assert_eq!(
        out_host,
        [
            30f32, 81., 132., 80., 256., 432., 130., 431., 732., 180., 606., 1032., 1883., 2434.,
            2985., 2308., 2984., 3660., 2733., 3534., 4335., 3158., 4084., 5010.
        ]
    );
}
