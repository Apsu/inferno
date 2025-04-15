use candle_core::{Layout, Shape, shape::Dim};
use cudarc::{
    driver::{
        CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, LaunchConfig, PushKernelArg,
        ValidAsZeroBits,
    },
    nvrtc::Ptx,
};
use half::{bf16, f16};
use std::sync::Arc;

use crate::modules;

pub trait DTypeLike: DeviceRepr + ValidAsZeroBits + Clone + Copy {
    const DTYPE: &'static str;
}

macro_rules! instantiate_dtypelike {
    ($t:ty) => {
        impl DTypeLike for $t {
            const DTYPE: &'static str = stringify!($t);
        }
    };
}
instantiate_dtypelike!(f32);
instantiate_dtypelike!(f16);
instantiate_dtypelike!(bf16);
instantiate_dtypelike!(u8);
instantiate_dtypelike!(u16);
instantiate_dtypelike!(u32);
instantiate_dtypelike!(u64);
instantiate_dtypelike!(i8);
instantiate_dtypelike!(i16);
instantiate_dtypelike!(i32);
instantiate_dtypelike!(i64);

#[derive(Debug, Clone)]
pub struct Tensor<T: DTypeLike> {
    data: CudaSlice<T>,
    layout: Layout,
    stream: Arc<CudaStream>,
}

impl<T: DTypeLike> Tensor<T> {
    pub fn from_raw(data: CudaSlice<T>, layout: Layout) -> anyhow::Result<Self> {
        if data.len() != layout.shape().elem_count() {
            anyhow::bail!(
                "from_raw expects slice len {} to match layout elem count {}",
                data.len(),
                layout.shape().elem_count()
            );
        }
        Ok(Self {
            stream: data.stream().clone(),
            data,
            layout,
        })
    }

    pub fn from_vec<S: Into<Shape>>(
        stream: Arc<CudaStream>,
        data: &[T],
        shape: S,
    ) -> anyhow::Result<Self> {
        let layout = Layout::contiguous(shape);
        if data.len() != layout.shape().elem_count() {
            anyhow::bail!(
                "from_vec expects host data len {} to match layout elem count {}",
                data.len(),
                layout.shape().elem_count()
            );
        }
        let dst = stream.memcpy_stod(data)?;
        Self::from_raw(dst, layout)
    }

    /// This returns a pre-sliced view of the tensor data.
    pub fn view(&self) -> CudaView<T> {
        self.data.as_view().slice(self.layout.start_offset()..)
    }

    /// This returns a pre-sliced view of the tensor data.
    pub fn view_mut(&mut self) -> CudaViewMut<T> {
        self.data
            .as_view_mut()
            .slice_mut(self.layout.start_offset()..)
    }

    pub fn into_slice(self) -> CudaSlice<T> {
        self.data
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn stride(&self) -> &[usize] {
        self.layout.stride()
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

    pub fn transpose<A: Dim, B: Dim>(&self, a: A, b: B) -> anyhow::Result<Self> {
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

    pub fn contiguous(&self) -> anyhow::Result<Self> {
        if self.layout.is_contiguous() {
            Ok(self.clone())
        } else {
            let elem_count = self.shape().elem_count();
            let rank = self.shape().rank();
            let x = self.data.as_view().slice(self.layout.start_offset()..);

            let mut out = self.stream.alloc_zeros::<T>(elem_count)?;
            let info = self
                .stream
                .memcpy_stod(&[self.shape().dims(), self.layout().stride()].concat())?;

            let module = self
                .stream
                .context()
                .load_module(Ptx::from_src(modules::UNARY))?;
            let func = module.load_function(&format!("ucopy_{}", T::DTYPE))?;
            let mut builder = self.stream.launch_builder(&func);

            builder.arg(&elem_count);
            builder.arg(&rank);
            builder.arg(&info);
            builder.arg(&x);
            builder.arg(&mut out);
            unsafe { builder.launch(LaunchConfig::for_num_elems(elem_count as u32))? };

            Self::from_raw(out, Layout::contiguous(self.shape()))
        }
    }

    pub fn to_vec(&self) -> anyhow::Result<Vec<T>> {
        let res = self.contiguous()?;
        let out_host = self.stream.memcpy_dtov(&res.view()).unwrap();
        Ok(out_host)
    }

    pub(crate) fn verify_all_same_device(&self, others: &[&Tensor<T>]) -> anyhow::Result<()> {
        let ord = self.stream.context().ordinal();
        if others.iter().any(|x| x.stream.context().ordinal() != ord) {
            anyhow::bail!("All tensors must be on the same device ordinal.")
        }
        Ok(())
    }
}
