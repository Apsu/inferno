// linear.rs
use crate::tensor::Tensor;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig, MatmulShared};
use cudarc::driver::CudaSlice;

pub struct Linear {
    pub weight: CudaSlice<f32>,       // shape [in, out]
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

        // Allocate output tensor with column-major layout
        let mut output = stream.alloc_zeros::<f32>(batch_size * self.out_features)?;

        unsafe {
            // Validate tensor layouts
            self.validate_shapes(input)?;

            // Perform cuBLASLt matmul
            self.blaslt.matmul(
                MatmulConfig {
                    transa: false,
                    transb: false,
                    m: batch_size as u64,
                    n: self.out_features as u64,
                    k: self.in_features as u64,
                    alpha: 1.0,
                    beta: 0.0,
                    lda: batch_size as i64, // Column-major: rows of input
                    ldb: self.in_features as i64, // Column-major: rows of weight
                    ldc: batch_size as i64, // Column-major: rows of output
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: Some(1),
                    batch_size: None,
                },
                &input.view(),
                &self.weight.as_view(),
                &mut output,
                self.bias.as_ref().map(|b| b.as_view()).as_ref(),
                None,
            )?;
        }

        Ok(Tensor::new(
            output,
            vec![batch_size, self.out_features],
            vec![1, batch_size], // Column-major strides
        ))
    }

    fn validate_shapes(&self, input: &Tensor<f32>) -> anyhow::Result<()> {
        assert_eq!(
            input.strides,
            vec![1, input.shape[0]],
            "Input must be column-major"
        );
        assert_eq!(
            self.weight.len(),
            self.in_features * self.out_features,
            "Weight matrix size mismatch"
        );
        Ok(())
    }
}
