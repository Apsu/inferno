use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor<f32>) -> anyhow::Result<Tensor<f32>>;
}
