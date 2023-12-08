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
use nalgebra::{Point3, DVector, Vector3};
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

        // collect polygon vertices
        let vertices = [vertex1, vertex2, vertex3];

        // collect their positions
        let triangle = [
            Point3::new(mesh.vertex_position(vertex1).x, mesh.vertex_position(vertex1).y, mesh.vertex_position(vertex1).z),
            Point3::new(mesh.vertex_position(vertex2).x, mesh.vertex_position(vertex2).y, mesh.vertex_position(vertex2).z),
            Point3::new(mesh.vertex_position(vertex3).x, mesh.vertex_position(vertex3).y, mesh.vertex_position(vertex3).z),
        ];

        // setup local laplace matrix for the triangle
        let laplace_matrix: DMatrix<f64> = polygon_laplace_matrix(&triangle);

        // ! collect the triplets
        // assemble local matrices into global matrix
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

fn polygon_laplace_matrix(polygon: &[Point3<f64>]) -> DMatrix<f64> {
    // Ensure the polygon is a triangle
    if polygon.len() != 3 {
        panic!("polygon_laplace_matrix is designed to handle triangles only");
    }

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

fn normalize_matrix(matrix: &mut CsrMatrix<f64>) {
    // Find the maximum positive value in the matrix
    let mut max_value = 0.0;
    for value in matrix.values() {
        if *value > 0.0 && *value > max_value {
            max_value = *value;
        }
    }

    if max_value > 0.0 {
        // Normalize and clamp positive values
        let values = matrix.values_mut();
        for value in values.iter_mut() {
            if *value > 0.0 {
                *value = (*value / max_value).clamp(0.0, 1.0);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};
    use std::env;
    use std::path::PathBuf;
    use crate::io;
    use nalgebra_sparse::CsrMatrix;
    use nalgebra::DMatrix;
    use csv::ReaderBuilder;
    use std::error::Error;

    fn load_test_mesh() -> Mesh {
        let mesh_cartography_lib_dir_str = env::var("Meshes_Dir").expect("MeshCartographyLib_DIR not set");
        let mesh_cartography_lib_dir = PathBuf::from(mesh_cartography_lib_dir_str);
        let new_path = mesh_cartography_lib_dir.join("ellipsoid_x4_open.obj");
        io::load_obj_mesh(new_path)
    }

    fn load_sparse_csv_data_to_csr_matrix(file_path: &str) -> Result<CsrMatrix<f64>, Box<dyn Error>> {
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

    fn load_csv_to_dmatrix(file_path: &str) -> Result<DMatrix<f64>, Box<dyn Error>> {
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

    #[test]
    fn test_cotangent_angle() {
        let v0 = Vector3::new(1.0, 0.0, 0.0);
        let v1 = Vector3::new(0.0, 1.0, 0.0);

        let cotangent = cotangent_angle(&v0, &v1);
        assert_eq!(cotangent, 0.0); // Cotangent of 90 degrees is 0
    }

    #[test]
    fn test_polygon_laplace_matrix() {
        // Define a simple equilateral triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.86602540378, 0.0), // sin(60 degrees) = ~0.866
        ];

        let laplace_matrix = polygon_laplace_matrix(&triangle);
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
    fn test_polygon_laplace_matrix_values() {
        // Define a simple right-angle triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];

        let laplace_matrix = polygon_laplace_matrix(&triangle);

        assert_eq!(laplace_matrix[(0, 1)], 1.0);
        assert_eq!(laplace_matrix[(1, 0)], 1.0);
        assert_eq!(laplace_matrix[(1, 2)], 1.0);
        assert_eq!(laplace_matrix[(2, 1)], 1.0);
        assert_eq!(laplace_matrix[(0, 0)], -1.0);
        assert_eq!(laplace_matrix[(1, 1)], -2.0);
        assert_eq!(laplace_matrix[(2, 2)], -1.0);
    }

    #[test]
    fn test_polygon_laplace_matrix_isosceles_triangle() {
        // Define an isosceles triangle
        let triangle = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];

        let laplace_matrix = polygon_laplace_matrix(&triangle);

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
        let surface_mesh = load_test_mesh();
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
        let surface_mesh = load_test_mesh();
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        assert_eq!(laplace_matrix.nnz(), 32845)
    }

    #[test]
    fn test_laplace_matrix_format() {
        let surface_mesh = load_test_mesh();
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        // Verify dimensions
        let nv = surface_mesh.no_vertices();
        assert_eq!(laplace_matrix.nrows(), nv);
        assert_eq!(laplace_matrix.ncols(), nv);
    }

    #[test]
    fn test_laplace_matrix_symmetry() {
        let surface_mesh = load_test_mesh();
        let laplace_matrix = build_laplace_matrix(&surface_mesh, true);

        assert_eq!(laplace_matrix, laplace_matrix.transpose());
    }

    #[test]
    fn test_laplace_matrix_row_sum() {
        let surface_mesh = load_test_mesh();
        let laplace_matrix = build_laplace_matrix(&surface_mesh, false);

        let mut row_sums = vec![0.0; laplace_matrix.nrows()];

        for (i, _, value) in laplace_matrix.triplet_iter() {
            row_sums[i] += *value;
        }
        for row_sum in row_sums {
            assert!(row_sum.abs() < 1e-6);
        }
    }

    #[test]
    fn test_laplace_matrix() {
        let surface_mesh: Mesh = load_test_mesh();
        let laplace_matrix: CsrMatrix<f64> = build_laplace_matrix(&surface_mesh, true);

        let file_path = "mocked_data/L_sparse.csv";
        let L_sparse = load_sparse_csv_data_to_csr_matrix(file_path).expect("Failed to load matrix");

        // Count the number of explicitly stored entries in the matrix
        assert_eq!(laplace_matrix.nnz(), L_sparse.nnz());
    }
}
