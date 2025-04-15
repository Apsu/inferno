use candle_core::{Layout, Shape};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor<T> {
    data: CudaSlice<T>,
    layout: Layout,
    shape: Shape,
}

impl<T> Tensor<T> {
    pub fn new_strided<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
        stride: Vec<usize>,
        start_offset: usize,
    ) -> Self {
        let shape = shape.into();
        Self {
            data,
            layout: Layout::new(shape.clone(), stride, start_offset),
            shape,
        }
    }

    pub fn new_contiguous<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
        start_offset: usize,
    ) -> Self {
        let shape = shape.into();
        Self {
            data,
            layout: Layout::contiguous_with_offset(shape.clone(), start_offset),
            shape,
        }
    }

    pub fn view(&self) -> CudaView<T> {
        self.data.as_view()
    }

    pub fn view_mut(&mut self) -> CudaViewMut<T> {
        self.data.as_view_mut()
    }

    pub fn into_slice(self) -> CudaSlice<T> {
        self.data
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl Tensor<f32> {
    pub fn from_host<S: Into<Shape>>(
        stream: Arc<CudaStream>,
        data: &[f32],
        shape: S,
        stride: Vec<usize>,
    ) -> anyhow::Result<Self> {
        let dst = stream.memcpy_stod(data)?;
        Ok(Self::new_strided(dst, shape, stride, 0))
    }

    pub fn from_zeros<S: Into<Shape>>(
        stream: Arc<CudaStream>,
        len: usize,
        shape: S,
        stride: Vec<usize>,
    ) -> anyhow::Result<Self> {
        let dst = stream.alloc_zeros::<f32>(len)?;
        Ok(Self::new_strided(dst, shape, stride, 0))
    }
}
