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
        let laplace_matrix: DMatrix<f64> = calculate_laplacian_matrix(&triangle);

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

// Helper function to compute the cross product and return its dot product with another vector
fn dot_cross(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> f64 {
    a.cross(b).dot(&c.cross(b))
}

// compute virtual vertex per polygon, represented by affine weights,
// such that the resulting triangle fan minimizes the sum of squared triangle areas
pub fn compute_virtual_vertex(poly: &DMatrix<f64>) -> DVector<f64> {
    let n = poly.nrows();
    let mut x = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);

    // Setup array of positions and edges
    for i in 0..n {
        x.push(Vector3::new(poly[(i, 0)], poly[(i, 1)], poly[(i, 2)]));
    }
    for i in 0..n {
        d.push(x[(i + 1) % n] - x[i]);
    }

    // Setup matrix A and rhs b
    let mut A = DMatrix::zeros(n + 1, n);
    let mut b = DVector::zeros(n + 1);
    for j in 0..n {
        for i in j..n {
            let mut Aij = 0.0;
            let mut bi = 0.0;
            for k in 0..n {
                Aij += dot_cross(&x[j], &d[k], &x[i]);
                bi += dot_cross(&x[i], &d[k], &x[k]);
            }
            A[(i, j)] = Aij;
            A[(j, i)] = Aij; // Symmetric entry
            b[i] = bi;
        }
    }
    for j in 0..n {
        A[(n, j)] = 1.0;
    }
    b[n] = 1.0;

    // Solving the linear system
    let svd = A.svd(true, true);
    let solution = svd.solve(&b, 1e-6).expect("SVD solve failed");

    // Extracting the top 'n' rows of the solution
    solution.rows(0, n).into()
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
    fn test_dot_cross() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        let v3 = Vector3::new(0.0, 0.0, 1.0);

        let result = dot_cross(&v1, &v2, &v3);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_cross_different_vectors() {
        let v4 = Vector3::new(1.0, 2.0, 3.0);
        let v5 = Vector3::new(-1.0, 0.5, 2.0);
        let v1 = Vector3::new(1.0, 0.0, 0.0);

        let result = dot_cross(&v4, &v5, &v1);
        assert_eq!(result, 11.25);
    }

    #[test]
    fn test_dot_cross_orthogonal_vectors() {
        let v6 = Vector3::new(0.0, 1.0, -1.0);
        let v7 = Vector3::new(1.0, 1.0, 1.0);
        let v1 = Vector3::new(1.0, 0.0, 0.0);

        let result = dot_cross(&v6, &v7, &v1);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_cross_parallel_vectors() {
        let v8 = Vector3::new(1.0, 2.0, 3.0);
        let v9 = Vector3::new(2.0, 4.0, 6.0);
        let v1 = Vector3::new(1.0, 0.0, 0.0);

        let result = dot_cross(&v8, &v9, &v1);
        assert_eq!(result, 0.0);
    }

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

        let file_path = "mocked_data/L.csv";
        let L_dense: DMatrix<f64> = load_csv_to_dmatrix(file_path).expect("Failed to load matrix");
        let L_sparse = CsrMatrix::from(&L_dense);  // Convert to CSR Sparse matrix

        // Count the number of explicitly stored entries in the matrix
        // assert_eq!(laplace_matrix.nnz(), L_sparse.nnz());
    }
}
