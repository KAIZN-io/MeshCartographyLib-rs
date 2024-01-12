//! # Input/Output
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-11
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use std::fs::File;
use std::io::{Write, Result};
use std::path::PathBuf;
use csv::ReaderBuilder;
use std::error::Error;
use nalgebra_sparse::{CsrMatrix, coo::CooMatrix};
use nalgebra::DMatrix;
use std::env;
use std::io::Read;

use wavefront_obj::obj::{self, Primitive};
use three_d_asset::{Positions, TriMesh as ThreeDTriMesh};
use tri_mesh::*;
use tri_mesh::Mesh;

use crate::mesh_definition;

pub struct Vector3Custom {
    x: f64,
    y: f64,
    z: f64,
}

pub struct TriMesh {
    positions: Vec<Vector3Custom>,
    normals: Option<Vec<Vector3Custom>>,
    uvs: Option<Vec<Vector3Custom>>,
    indices: Vec<u32>,
}

fn create_tri_mesh(obj_set: obj::ObjSet) -> std::result::Result<TriMesh, String> {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    for object in obj_set.objects {
        // Directly add all vertices from the .obj file to the positions vector
        for vertex in object.vertices {
            positions.push(Vector3Custom { x: vertex.x, y: vertex.y, z: vertex.z });
        }

        // Build faces using indices from the .obj file
        for geometry in object.geometry {
            for shape in geometry.shapes {
                match shape.primitive {
                    Primitive::Triangle(v1, v2, v3) => {
                        // Indices in .obj files are 1-based, so subtract 1 to convert to 0-based
                        indices.push(v1.0 as u32);
                        indices.push(v2.0 as u32);
                        indices.push(v3.0 as u32);
                    }
                    _ => return Err("Unsupported primitive type".to_string()),
                }
            }
        }
    }

    Ok(TriMesh {
        positions,
        normals: None,
        uvs: None,
        indices,
    })
}

fn convert_to_tri_mesh_mesh(tri_mesh: TriMesh) -> std::result::Result<Mesh, String> {
    // Convert positions to the format expected by three_d_asset::TriMesh
    let positions = Positions::F64(tri_mesh.positions.iter().map(|v| {
        vec3(v.x, v.y, v.z) // Assuming vec3 is from three_d_asset or a similar crate
    }).collect());

    // Convert indices to the format expected by three_d_asset::TriMesh
    let indices = three_d_asset::Indices::U32(tri_mesh.indices);

    // Create the three_d_asset::TriMesh
    let three_d_tri_mesh = ThreeDTriMesh {
        positions,
        indices,
        ..Default::default() // Add normals, uvs, etc., if available
    };

    // Create the tri_mesh::Mesh
    let mesh = Mesh::new(&three_d_tri_mesh);
    Ok(mesh)
}

#[allow(dead_code)]
pub fn load_mesh_from_obj(path: PathBuf) -> std::result::Result<Mesh, String> {
    // Open the file using the PathBuf
    let mut file = File::open(&path).map_err(|e| e.to_string())?;

    // Read the entire file into a string
    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(|e| e.to_string())?;

    // Parse the OBJ data
    let obj_set = obj::parse(&contents).map_err(|e| e.to_string())?;

    // Create a TriMesh from the OBJ data
    let pre_surface_mesh = create_tri_mesh(obj_set).unwrap();
    let surface_mesh = convert_to_tri_mesh_mesh(pre_surface_mesh).unwrap();

    Ok(surface_mesh)
}

pub fn load_mesh_from_js_obj(obj_data: String) -> std::result::Result<Mesh, String> {
    // Parse the OBJ data from the string
    let obj_set = obj::parse(&obj_data).map_err(|e| e.to_string())?;

    // Create a TriMesh from the OBJ data
    let pre_surface_mesh = create_tri_mesh(obj_set)?;
    let surface_mesh = convert_to_tri_mesh_mesh(pre_surface_mesh)?;

    Ok(surface_mesh)
}

#[allow(dead_code)]
pub fn save_mesh_as_obj(mesh: &tri_mesh::Mesh, file_path: PathBuf) -> Result<()> {
    let mut file = File::create(file_path)?;

    // Add meta data
    writeln!(file, "# Generated by MeshCartographyLib")?;

    // Write vertices
    for vertex_id in mesh.vertex_iter() {
        let vertex = mesh.vertex_position(vertex_id);
        writeln!(file, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
    }

    // Write faces
    for face_id in mesh.face_iter() {
        let face = mesh.face_vertices(face_id);

        // OBJ indices start at 1, so we need to add 1 to each index
        let f0 = face.0.to_string().parse::<i32>().unwrap() + 1;
        let f1 = face.1.to_string().parse::<i32>().unwrap() + 1;
        let f2 = face.2.to_string().parse::<i32>().unwrap() + 1;

        writeln!(file, "f {} {} {}", f0, f1, f2)?;
    }

    Ok(())
}

#[allow(dead_code)]
pub fn save_uv_mesh_as_obj(mesh: &tri_mesh::Mesh, mesh_tex_coords: &mesh_definition::MeshTexCoords, file_path: PathBuf) -> Result<()> {
    let mut file = File::create(file_path)?;

    // Add meta data
    writeln!(file, "# Generated by MeshCartographyLib")?;

    // Write vertices
    for vertex_id in mesh.vertex_iter() {
        if let Some(tex_coord) = mesh_tex_coords.get_tex_coord(vertex_id) {
            writeln!(file, "v {} {} {}", tex_coord.0, tex_coord.1, 0)?;
        }
    }

    // Write faces
    for face_id in mesh.face_iter() {
        let face = mesh.face_vertices(face_id);

        // OBJ indices start at 1, so we need to add 1 to each index
        let f0 = face.0.to_string().parse::<i32>().unwrap() + 1;
        let f1 = face.1.to_string().parse::<i32>().unwrap() + 1;
        let f2 = face.2.to_string().parse::<i32>().unwrap() + 1;

        writeln!(file, "f {} {} {}", f0, f1, f2)?;
    }

    Ok(())
}

pub fn convert_uv_mesh_to_string(mesh: &tri_mesh::Mesh, mesh_tex_coords: &mesh_definition::MeshTexCoords) -> std::result::Result<String, String> {
    let mut obj_data = String::new();

    // Add meta data
    obj_data.push_str("# Generated by MeshCartographyLib\n");

    // Write vertices
    for vertex_id in mesh.vertex_iter() {
        if let Some(tex_coord) = mesh_tex_coords.get_tex_coord(vertex_id) {
            let vertex_line = format!("v {} {} {}\n", tex_coord.0, tex_coord.1, 0.0);
            obj_data.push_str(&vertex_line);
        }
    }

    // Write faces
    for face_id in mesh.face_iter() {
        let face = mesh.face_vertices(face_id);
        let f0 = face.0.to_string().parse::<i32>().unwrap() + 1;
        let f1 = face.1.to_string().parse::<i32>().unwrap() + 1;
        let f2 = face.2.to_string().parse::<i32>().unwrap() + 1;

        let face_line = format!("f {} {} {}\n", f0, f1, f2);
        obj_data.push_str(&face_line);
    }

    Ok(obj_data)
}

#[allow(dead_code)]
pub fn load_test_mesh() -> Mesh {
    let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
    let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
    let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");
    load_mesh_from_obj(new_path).unwrap()
}

#[allow(dead_code)]
pub fn load_test_mesh_closed() -> Mesh {
    let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
    let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
    let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4.obj");
    load_mesh_from_obj(new_path).unwrap()
}

#[allow(dead_code)]
pub fn load_sparse_csv_data_to_csr_matrix(file_path: &str) -> std::result::Result<CsrMatrix<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(file_path)?;

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut max_row_index = 0;
    let mut max_col_index = 0;

    for result in reader.records() {
        let record = result?;
        let row_index: usize = record[0].trim().parse()?;
        let col_index: usize = record[1].trim().parse()?;
        let value: f64 = record[2].trim().parse()?;

        row_indices.push(row_index - 1); // Assuming 1-based indices in CSV
        col_indices.push(col_index - 1);
        values.push(value);

        max_row_index = max_row_index.max(row_index);
        max_col_index = max_col_index.max(col_index);
    }

    // Use try_from_triplets with matrix dimensions
    // A COO Sparse matrix stores entries in coordinate-form, that is triplets (i, j, v), where i and j correspond to row and column indices of the entry, and v to the value of the entry
    let coo_matrix = CooMatrix::try_from_triplets(max_row_index, max_col_index, row_indices, col_indices, values)?;

    // Convert the CooMatrix to a CsrMatrix
    Ok(CsrMatrix::from(&coo_matrix))
}

#[allow(dead_code)]
pub fn load_csv_to_dmatrix(file_path: &str) -> std::result::Result<DMatrix<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(file_path)?;

    let mut data = Vec::new();
    let mut nrows = 0;
    let mut ncols = 0;

    for result in reader.records() {
        let record = result?;
        nrows += 1;
        ncols = record.len();

        for field in record.iter() {
            let value: f64 = field.trim().parse()?;
            data.push(value);
        }
    }

    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

#[allow(dead_code)]
pub fn load_csv_to_bool_vec(file_path: &str) -> std::result::Result<Vec<bool>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(file_path)?;

    let mut bools = Vec::new();
    for result in reader.records() {
        let record = result?;
        if let Some(field) = record.get(0) {
            let value: u8 = field.trim().parse()?;
            bools.push(value != 0);
        }
    }

    Ok(bools)
}
