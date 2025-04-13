use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor<T> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn view(&self) -> CudaView<T> {
        self.data.as_view()
    }

    pub fn view_mut(&mut self) -> CudaViewMut<T> {
        self.data
            .as_view_mut()
    }
}

impl Tensor<f32> {
    pub fn from_host(
        stream: Arc<CudaStream>,
        data: &[f32],
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> anyhow::Result<Self> {
        let mut dst = unsafe { stream.alloc(data.len())? };
        stream.memcpy_htod(data, &mut dst)?;
        Ok(Self::new(dst, shape, strides))
    }

    pub fn from_zeros(
        stream: Arc<CudaStream>,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> anyhow::Result<Self> {
        let dst = stream.alloc_zeros::<f32>(len)?;
        Ok(Self::new(dst, shape, strides))
    }
}
