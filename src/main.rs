//! # Example of how to use a PyTorch exported script module in Rust
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-15
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use std::path::PathBuf;
use std::error::Error;
use std::env;

extern crate new_king_lib;
use new_king_lib::io;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logger
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Expect at least one argument: the file path
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_mesh_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];

    // Load mesh from file
    let mesh_path = PathBuf::from(file_path);
    let surface_closed = io::load_mesh_from_obj(mesh_path.clone()).unwrap();

    let mut processor = new_king_lib::MeshProcessor::new(surface_closed);
    let mut uv_mesh = processor.create_uv_surface(file_path);
    let tessellation_mesh = processor.create_tessellation_mesh(&mut uv_mesh);

    Ok(())
}
