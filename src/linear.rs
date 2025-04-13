use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::CudaSlice;

use crate::tensor::Tensor;

pub struct Linear {
    pub weight: CudaSlice<f32>, // shape [out, in]
    pub bias: Option<CudaSlice<f32>>, // shape [out]
    pub in_features: usize,
    pub out_features: usize,
    pub blaslt: CudaBlasLT,
}

impl Linear {
    pub fn new(
        weight: CudaSlice<f32>,
        bias: Option<CudaSlice<f32>>,
        in_features: usize,
        out_features: usize,
        blaslt: CudaBlasLT,
    ) -> Self {
        Self {
            weight,
            bias,
            in_features,
            out_features,
            blaslt,
        }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let batch_size = input.shape[0];
        let stream = self.blaslt.stream();

        let mut output = stream.alloc_zeros::<f32>(batch_size * self.out_features)?;

        dbg!(
            batch_size,
            self.out_features,
            self.in_features,
            self.in_features as i64, // lda
            self.in_features as i64, // ldb
            self.out_features as i64 // ldc
        );

        unsafe {
            let bias = self.bias.as_ref().map(|b| b.as_view());

            self.blaslt.matmul(
                MatmulConfig {
                    transa: false,
                    transb: true,
                    m: batch_size as u64,
                    n: self.out_features as u64,
                    k: self.in_features as u64,
                    alpha: 1.0,
                    beta: 0.0,
                    lda: self.in_features as i64,
                    ldb: self.out_features as i64,
                    ldc: self.out_features as i64,
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                },
                &input.view(),
                &self.weight.as_view(),
                &mut output,
                bias.as_ref(),
                None,
            )?;
        }

        Ok(Tensor::new(output, vec![batch_size, self.out_features], vec![self.out_features, 1]))
    }
}
