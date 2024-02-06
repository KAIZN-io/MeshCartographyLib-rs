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

fn main() -> miette::Result<()>  {
    // Sets the include path for C++ headers
    let include_path = std::path::PathBuf::from("src/cpp");

    // Initializes the `autocxx` build process
    let mut builder = autocxx_build::Builder::new("src/lib.rs", &[&include_path])
        .build()?;

    // Compiles the generated bindings with C++17 standards
    builder.flag("-std=c++17").compile("mesh_cartography");

    // Instructs Cargo to re-run this script if `main.rs` changes
    println!("cargo:rerun-if-changed=src/lib.rs");

    Ok(())
}
