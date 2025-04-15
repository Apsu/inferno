// linear.rs
use crate::tensor::Tensor;
use candle_core::{D, Shape};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig, MatmulShared};

pub struct Linear {
    pub weight: Tensor<f32>,
    pub bias: Option<Tensor<f32>>,
    pub blaslt: CudaBlasLT,
}

impl Linear {
    /// * `weight` - Input tensor of size BxNxK
    /// * `.bias` - Optional bias tensor of size M
    pub fn new(weight: Tensor<f32>, bias: Option<Tensor<f32>>) -> Self {
        let blaslt = CudaBlasLT::new(weight.stream()).unwrap();
        Self {
            weight,
            bias,
            blaslt,
        }
    }

    /// * `input` - Input tensor of size BxMxK
    /// * `self.weight` - Input tensor of size BxNxK
    /// * `self.bias` - Optional bias tensor of size M
    ///
    /// The resulting tensor is of shape BxMxN
    pub fn forward(&self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        self.weight.verify_all_same_device(&[input])?;
        if let Some(b) = &self.bias {
            self.weight.verify_all_same_device(&[b])?;
        }

        // Assume TN
        let (batch_size, m, k) = input.shape().dims3()?;
        let (b_0, n, b_2) = self.weight.shape().dims3()?;
        let stream = self.blaslt.stream();

        if b_2 != k {
            anyhow::bail!("This layer only supports TN layout");
        }

        if b_0 != batch_size {
            anyhow::bail!("`b` must have the same batch size as `a`")
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((batch_size, n, m));

        let (bias, bias_stride) = if let Some(bias) = &self.bias {
            let bias_l = bias.layout();
            if bias_l.dims().len() == 1 {
                if bias_l.shape().dims1()? != m {
                    anyhow::bail!("Bias does not have the correct shape");
                }
                (Some(bias.view().slice(bias_l.start_offset()..)), None)
            } else {
                if bias_l.shape().dims2()?.1 != m {
                    anyhow::bail!("Bias does not have the correct shape");
                }
                if bias_l.shape().dims2()?.0 != batch_size {
                    anyhow::bail!("Bias batch size must match batch size of `a`");
                }
                let bias_stride = bias_l.stride()[0] as i64;
                (
                    Some(bias.view().slice(bias_l.start_offset()..)),
                    Some(bias_stride),
                )
            }
        } else {
            (None, None)
        };

        let (mut out, stride_c) = {
            // Allocate out tensor
            (stream.alloc_zeros::<f32>(out_shape.elem_count())?, (n * m))
        };

        unsafe {
            // Perform cuBLASLt matmul
            self.blaslt.matmul(
                MatmulConfig {
                    transa: true,
                    transb: false,
                    m: m as u64,
                    n: n as u64,
                    k: k as u64,
                    alpha: 1.0, //self.alpha.unwrap_or(1.0),
                    lda: lda as i64,
                    ldb: ldb as i64,
                    beta: 0.0, //self.beta.unwrap_or(0.0),
                    ldc: ldc as i64,
                    stride_a: Some(input.layout().stride()[0] as i64),
                    stride_b: Some(self.weight.layout().stride()[0] as i64),
                    stride_c: Some(stride_c as i64),
                    stride_bias: bias_stride,
                    batch_size: Some(batch_size as i32),
                },
                &input.view(),
                &self.weight.view(),
                &mut out,
                bias.as_ref(),
                None,
            )?;
        }

        let res = Tensor::new_contiguous(out, out_shape, 0);
        res.transpose(D::Minus1, D::Minus2)
    }
}
