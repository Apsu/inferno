use anyhow::Result;

const KERNEL_FILES: &[&str] = &["kernels/unary.cu"];
const INCLUDE_FILES: &[&str] = &["kernels/common.cuh"];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(KERNEL_FILES.to_vec())
        .include_paths(INCLUDE_FILES.to_vec());
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/kernels.rs").unwrap();
    Ok(())
}
