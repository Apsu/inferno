use candle_core::{D, Layout, Shape, shape::Dim};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor<T: DeviceRepr> {
    data: CudaSlice<T>,
    layout: Layout,
    stream: Arc<CudaStream>,
}

impl<T: DeviceRepr> Tensor<T> {
    pub fn new_strided<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
        stride: Vec<usize>,
        start_offset: usize,
    ) -> Self {
        let shape = shape.into();
        Self {
            stream: data.stream().clone(),
            data,
            layout: Layout::new(shape, stride, start_offset),
        }
    }

    pub fn new_contiguous<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
        start_offset: usize,
    ) -> Self {
        let shape = shape.into();
        Self {
            stream: data.stream().clone(),
            data,
            layout: Layout::contiguous_with_offset(shape.clone(), start_offset),
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
        self.layout.shape()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn transpose(&self, a: D, b: D) -> anyhow::Result<Self> {
        let layout = self.layout.transpose(
            a.to_index(self.shape(), "transpose")?,
            b.to_index(self.shape(), "transpose")?,
        )?;
        Ok(Self {
            stream: self.stream.clone(),
            data: self.data.clone(),
            layout,
        })
    }

    pub fn to_vec(&self) -> anyhow::Result<Vec<T>> {
        let out_host = self.stream.memcpy_dtov(&self.view()).unwrap();
        todo!()
    }

    pub(crate) fn verify_all_same_device(&self, others: &[&Tensor<T>]) -> anyhow::Result<()> {
        let ord = self.stream.context().ordinal();
        if others.iter().any(|x| x.stream.context().ordinal() != ord) {
            anyhow::bail!("All tensors must be on the same device ordinal.")
        }
        Ok(())
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
