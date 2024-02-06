// //! # Build C++ Scripts for Rust Project Using `autocxx`
// //!
// //! This script is responsible for setting up the build process for a Rust project
// //! which uses `autocxx` for C++ interoperability.
// //!
// //! ## Functionality
// //! - It specifies the include paths for C++ headers.
// //! - It initializes and configures the `autocxx` build process.
// //! - It sets compiler flags and compiles the generated bindings.
// //! - It includes instructions for Cargo to re-run the build script upon changes.

fn main() -> miette::Result<()> {
    // Existing include path for C++ headers in your Rust project
    let include_path = std::path::PathBuf::from("src/cpp");

    // Relative path to the Eigen headers
    let eigen_include_path = std::path::PathBuf::from("./pmp-library/external/eigen-3.4.0");

    // Initializes the `autocxx` build process with both include paths
    let mut builder = autocxx_build::Builder::new("src/lib.rs", &[&include_path, &eigen_include_path])
        .build()?;

    // Compiles the generated bindings with C++17 standards
    // and suppresses specific warnings
    builder
        .flag("-std=c++17")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .compile("mesh_cartography");

    // Instructs Cargo to re-run this script if `lib.rs` changes
    println!("cargo:rerun-if-changed=src/lib.rs");

    Ok(())
}
