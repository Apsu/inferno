use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor<T> {
    pub data: Arc<CudaSlice<T>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn view(&self) -> CudaView<T> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> CudaViewMut<T> {
        Arc::get_mut(&mut self.data)
            .expect("Cannot get mutable view — tensor has multiple owners")
            .view_mut()
    }
}
