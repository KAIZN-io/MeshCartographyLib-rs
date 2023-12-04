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
    // Instructs Cargo to re-run this script if `main.rs` changes
    println!("cargo:rerun-if-changed=src/lib.rs");

    Ok(())
}
