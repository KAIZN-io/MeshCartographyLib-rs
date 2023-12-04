//! # Laplace Matrix
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-22
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** None known at this time.
//! - **Todo:** Further development tasks to be determined.

use nalgebra::DMatrix;
use nalgebra::{Point3, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

extern crate tri_mesh;
use tri_mesh::Mesh;


/// Get the Laplace matrix of a Surface Mesh.
#[allow(non_snake_case)]
pub fn build_laplace_matrix(mesh: &Mesh, clamp: bool) -> CsrMatrix<f64> {
    let num_vertices = mesh.no_vertices();
    let mut coo = CooMatrix::new(num_vertices, num_vertices);

    for face in mesh.face_iter() {
        let (vertex1, vertex2, vertex3) = mesh.face_vertices(face);

        // Use each vertex individually instead of iterating over a tuple
        let vertices = [vertex1, vertex2, vertex3];

        // Create an array of Point3<f64> for the triangle
        let triangle = [
            Point3::new(mesh.vertex_position(vertex1).x, mesh.vertex_position(vertex1).y, mesh.vertex_position(vertex1).z),
            Point3::new(mesh.vertex_position(vertex2).x, mesh.vertex_position(vertex2).y, mesh.vertex_position(vertex2).z),
            Point3::new(mesh.vertex_position(vertex3).x, mesh.vertex_position(vertex3).y, mesh.vertex_position(vertex3).z),
        ];

        // Compute local Laplace matrix for the triangle
        let laplace_matrix = calculate_laplacian_matrix(&triangle);

        // Assemble local matrices into global matrix
        for (j, &vertex_j) in vertices.iter().enumerate() {
            for (k, &vertex_k) in vertices.iter().enumerate() {
                let index_as_u32: u32 = *vertex_j;
                let idx_j: usize = index_as_u32 as usize;

                let index_as_u32: u32 = *vertex_k;
                let idx_k: usize = index_as_u32 as usize;

                let value = laplace_matrix[(k, j)];
                coo.push(idx_j, idx_k, value);
            }
        }
    }

    // Convert COO to CSR format
    let mut L = CsrMatrix::from(&coo);

    // Clamping negative off-diagonal entries to zero
    if clamp {
        for (i, j, value) in L.triplet_iter_mut() {
            if i != j && *value < 0.0 {
                *value = 0.0;
            }
        }
    }

    L
}

fn calculate_laplacian_matrix(polygon: &[Point3<f64>]) -> DMatrix<f64> {
    let a = polygon[0];
    let b = polygon[1];
    let c = polygon[2];

    let ab = b - a;
    let bc = c - b;
    let ca = a - c;

    let cot_cab = cotangent_angle(&ca, &-bc);
    let cot_bca = cotangent_angle(&bc, &-ab);

    DMatrix::from_row_slice(3, 3, &[
        -cot_cab, cot_cab, 0.0,
        cot_cab, -(cot_cab + cot_bca), cot_bca,
        0.0, cot_bca, -cot_bca
    ])
}

fn cotangent_angle(v0: &Vector3<f64>, v1: &Vector3<f64>) -> f64 {
    let dot = v0.dot(v1);
    let cross = v0.cross(v1).norm();

    dot / cross
}



#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};
    use std::env;
    use std::path::PathBuf;
    use crate::io;

    #[test]
    fn test_cotangent_angle() {
        let v0 = Vector3::new(1.0, 0.0, 0.0);
        let v1 = Vector3::new(0.0, 1.0, 0.0);

        let cotangent = cotangent_angle(&v0, &v1);
        assert_eq!(cotangent, 0.0); // Cotangent of 90 degrees is 0
    }

    #[test]
    fn test_calculate_laplacian_matrix() {
        // Define a simple equilateral triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.86602540378, 0.0), // sin(60 degrees) = ~0.866
        ];

        let laplace_matrix = calculate_laplacian_matrix(&triangle);
        // Check if the matrix has the expected properties, e.g., symmetry, non-zero values at certain positions, etc.
        // As an example, checking symmetry:
        assert_eq!(laplace_matrix[(0, 1)], laplace_matrix[(1, 0)]);
        assert_eq!(laplace_matrix[(1, 2)], laplace_matrix[(2, 1)]);
        assert_eq!(laplace_matrix[(2, 0)], laplace_matrix[(0, 2)]);
    }

    #[test]
    fn test_cotangent_angle_acute() {
        let v0 = Vector3::new(1.0, 0.0, 0.0);
        let v1 = Vector3::new(1.0, 1.0, 0.0);

        let cotangent = cotangent_angle(&v0, &v1);
        assert!(cotangent > 0.0); // Cotangent of an acute angle is positive
    }

    #[test]
    fn test_cotangent_angle_obtuse() {
        let v0 = Vector3::new(1.0, 0.0, 0.0);
        let v1 = Vector3::new(-1.0, 1.0, 0.0);

        let cotangent = cotangent_angle(&v0, &v1);
        assert!(cotangent < 0.0); // Cotangent of an obtuse angle is negative
    }

    #[test]
    fn test_calculate_laplacian_matrix_values() {
        // Define a simple right-angle triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];

        let laplace_matrix = calculate_laplacian_matrix(&triangle);

        assert_eq!(laplace_matrix[(0, 1)], 1.0);
        assert_eq!(laplace_matrix[(1, 0)], 1.0);
        assert_eq!(laplace_matrix[(1, 2)], 1.0);
        assert_eq!(laplace_matrix[(2, 1)], 1.0);
        assert_eq!(laplace_matrix[(0, 0)], -1.0);
        assert_eq!(laplace_matrix[(1, 1)], -2.0);
        assert_eq!(laplace_matrix[(2, 2)], -1.0);
    }

    #[test]
    fn test_calculate_laplacian_matrix_isosceles_triangle() {
        // Define an isosceles triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];

        let laplace_matrix = calculate_laplacian_matrix(&triangle);

        assert_eq!(laplace_matrix[(0, 0)], 0.0);
        assert_eq!(laplace_matrix[(0, 1)], 0.0);
        assert_eq!(laplace_matrix[(0, 2)], 0.0);
        assert_eq!(laplace_matrix[(1, 0)], 0.0);
        assert_eq!(laplace_matrix[(1, 1)], -1.0);
        assert_eq!(laplace_matrix[(1, 2)], 1.0);
        assert_eq!(laplace_matrix[(2, 0)], 0.0);
        assert_eq!(laplace_matrix[(2, 1)], 1.0);
        assert_eq!(laplace_matrix[(2, 2)], -1.0);
    }

    #[test]
    fn test_laplace_matrix_diagonal_elements() {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");

        let surface_mesh = io::load_obj_mesh(new_path);
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        for (i, j, value) in laplace_matrix.triplet_iter() {
            if i == j {
                assert!(*value < 0.0);
            } else {
                assert!(*value >= 0.0);
            }
        }
    }

    #[test]
    fn test_laplace_matrix_number_nonzero_elements() {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");

        let surface_mesh = io::load_obj_mesh(new_path);
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        assert_eq!(laplace_matrix.nnz(), 32845)
    }

    #[test]
    fn test_laplace_matrix_format() {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");

        let surface_mesh = io::load_obj_mesh(new_path);
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        // Verify dimensions
        let nv = surface_mesh.no_vertices();
        assert_eq!(laplace_matrix.nrows(), nv);
        assert_eq!(laplace_matrix.ncols(), nv);
    }

    #[test]
    fn test_laplace_matrix_symmetry() {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");

        let surface_mesh = io::load_obj_mesh(new_path);
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        assert_eq!(laplace_matrix, laplace_matrix.transpose());
    }

    #[test]
    fn test_laplace_matrix_row_sum() {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");

        let surface_mesh = io::load_obj_mesh(new_path);
        let laplace_matrix = build_laplace_matrix(&surface_mesh, false);

        let mut row_sums = vec![0.0; laplace_matrix.nrows()];

        for (i, _, value) in laplace_matrix.triplet_iter() {
            row_sums[i] += *value;
        }
        for row_sum in row_sums {
            assert!(row_sum.abs() < 1e-6);
        }
    }
}
