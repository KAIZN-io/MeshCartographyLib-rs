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

extern crate mesh_cartography_lib;
use mesh_cartography_lib::io;
use log::info;

fn main() -> Result<(), Box<dyn Error>> {
    mesh_cartography_lib::init_logger();
    info!("Logger initialized.");

    let args: Vec<String> = env::args().collect();

    // Expect at least one argument: the file path
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_mesh_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];

    // Load mesh from file
    let mesh_path = PathBuf::from(file_path);
    let surface_closed = match io::load_mesh_from_obj(mesh_path.clone()) {
        Ok(mesh) => mesh,
        Err(e) => {
            eprintln!("Failed to load mesh: {}", e);
            return Err(e.into());
        }
    };

    let mut processor = mesh_cartography_lib::MeshProcessor::from_mesh(surface_closed);
    let mut uv_mesh = processor.create_uv_surface(file_path);
    info!("UV mesh created");

    // // Get the uv mesh path
    // let uv_mesh_path = processor.mesh_uv_path.clone();
    // let uv_mesh_path = uv_mesh_path.into_os_string().into_string().unwrap();
    // // Open OBJ path and return file content as a string
    // let data = fs::read_to_string(uv_mesh_path).expect("Unable to read file");

    processor.create_tessellation_mesh(&mut uv_mesh);

    Ok(())
}
