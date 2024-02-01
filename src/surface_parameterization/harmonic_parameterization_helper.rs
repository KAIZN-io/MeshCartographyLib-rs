//! # Harmonic Parameterization Helper
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-24
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** Improve the speed of the QR decomposition.

use nalgebra::{DMatrix, Cholesky};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use std::{collections::HashMap, ops::AddAssign};
use tri_mesh::Mesh;

use crate::mesh_definition::{TexCoord, MeshTexCoords};
use crate::surface_parameterization::{laplacian_matrix, boundary_matrix};

/// Represents a triplet in a sparse matrix.
struct Triplet<T> {
    row: usize,
    col: usize,
    value: T,
}

/// Performs harmonic parameterization on a mesh.
pub fn harmonic_parameterization(mesh: &Mesh, mesh_tex_coords: &mut MeshTexCoords, use_uniform_weights: bool) {
    // Set which vertices are constrained (i.e. on the boundary)
    let mut is_constrained = Vec::new();
    for vertex_id in mesh.vertex_iter() {
        is_constrained.push(mesh.is_vertex_on_boundary(vertex_id));
    }

    // build system matrix (clamp negative cotan weights to zero)
    // 1. Get the local geometry and relationships between the mesh vertices
    let l_mtx = laplacian_matrix::build_laplace_matrix(mesh, use_uniform_weights);

    // 2. Inject Boundary Constraints -> sets fixed boundary vertices
    let b_mtx = boundary_matrix::set_boundary_constraints(mesh, mesh_tex_coords);

    // 3. Solve the linear equation system
    let result = solve_using_qr_decomposition(&l_mtx, &b_mtx, is_constrained);

    match result {
        Ok(x_mtx) => {
            for (vertex_id, row) in mesh.vertex_iter().zip(x_mtx.row_iter()) {
                let tex_coord = TexCoord(row[0], row[1]);
                // println!("tex_coord: {:?} {:?}", row[0], row[1]);
                mesh_tex_coords.set_tex_coord(vertex_id, tex_coord);
            }
        }
        Err(e) => {
            println!("An error occurred: {}", e);
        }
    }
}

pub fn solve_using_qr_decomposition(l_mtx: &CsrMatrix<f64>, b_mtx: &DMatrix<f64>, is_constrained: Vec<bool>) -> Result<DMatrix<f64>, String> {
    let nrows = l_mtx.nrows();
    let mut idx = vec![usize::MAX; nrows];
    let mut n_dofs = 0;
    let mut bb_mtx = DMatrix::zeros(nrows, b_mtx.ncols());
    for i in 0..nrows {
        if !is_constrained[i]{
            idx[i] = n_dofs;
            bb_mtx.set_row(n_dofs, &b_mtx.row(i));
            n_dofs += 1;
        }
    }

    bb_mtx.resize_mut(n_dofs, b_mtx.ncols(), 0.0); // Resize BB after filling it

    // collect entries for reduced matrix
    // update rhs with constraints
    let sparse_matrix_triplets: Vec<Triplet<f64>> = get_tripplets(&l_mtx, &b_mtx, &mut bb_mtx, &idx);

    let dense_mtx = build_dense_matrix(&sparse_matrix_triplets, bb_mtx.nrows());

    // Solve the system Lxx = BB using Cholesky decomposition
    let cholesky = Cholesky::new(dense_mtx).unwrap();
    let xx = cholesky.solve(&bb_mtx);

    // Fill in the solution X
    let mut x_mtx = DMatrix::zeros(b_mtx.nrows(), b_mtx.ncols());
    for i in 0..l_mtx.nrows() {
        for j in 0..b_mtx.ncols() {
            x_mtx[(i, j)] = if idx[i] == usize::MAX { b_mtx[(i, j)] } else { xx[(idx[i], j)] };
        }
    }

    Ok(x_mtx)
}


/// A COO Sparse matrix stores entries in coordinate-form, that is triplets (i, j, v), where i and j correspond to row and column indices of the entry, and v to the value of the entry
fn get_tripplets(l_mtx: &CsrMatrix<f64>, b_mtx: &DMatrix<f64>, bb_mtx: &mut DMatrix<f64>, idx: &[usize]) -> Vec<Triplet<f64>> {
    let mut sparse_matrix_triplets: Vec<Triplet<f64>> = Vec::new();
    for triplet in l_mtx.triplet_iter() {
        let i = triplet.0;
        let j = triplet.1;
        let v = triplet.2;

        if idx[i] != usize::MAX { // row is dof
            if idx[j] != usize::MAX { // col is dof
                sparse_matrix_triplets.push(Triplet { row: idx[i], col: idx[j], value: *v });
            } else { // col is constraint
                // Update B
                for col in 0..b_mtx.ncols() {
                    bb_mtx[(idx[i], col)] -= v * b_mtx[(j, col)];
                }
            }
        }
    }

    sparse_matrix_triplets
}


// Function to convert custom triplets to a CSR matrix
fn build_csr_matrix<T: Copy + nalgebra::Scalar + Zero + AddAssign>(nrows: usize, ncols: usize, sparse_matrix_triplets: &[Triplet<T>]) -> CsrMatrix<T> {
    // Collect the entries
    let entries = collect_entries(sparse_matrix_triplets);

    // Convert the sorted entries to vectors for CSR matrix construction
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0; nrows + 1];

    for ((row, col), value) in entries {
        values.push(value);
        row_indices.push(col);  // Note: col indices for each row
        col_ptrs[row + 1] += 1;
    }

    // Compute the starting index of each row
    for i in 1..=nrows {
        col_ptrs[i] += col_ptrs[i - 1];
    }

    // Create the CSR matrix
    let csr_matrix = CsrMatrix::try_from_csr_data(nrows, ncols, col_ptrs, row_indices, values)
        .expect("Failed to create CSR matrix");

    csr_matrix
}

fn build_dense_matrix(sparse_matrix_triplets: &[Triplet<f64>], n_dofs: usize) -> DMatrix<f64> {
    let csr_matrix = build_csr_matrix(n_dofs, n_dofs, &sparse_matrix_triplets);

    // Convert CSR matrix to dense matrix
    DMatrix::from(&csr_matrix)
}

fn collect_entries<T: Copy + nalgebra::Scalar + Zero + AddAssign>(sparse_matrix_triplets: &[Triplet<T>]) -> Vec<((usize, usize), T)> {
    // ? Oder ist das hier der Bug, wegen der Verwendung von HashMap?
    let mut entries: HashMap<(usize, usize), T> = HashMap::new();
    for triplet in sparse_matrix_triplets {
        let key = (triplet.row, triplet.col);
        *entries.entry(key).or_insert_with(Zero::zero) += triplet.value;
    }

    // Sort entries: first by row, then by column
    let mut sorted_entries: Vec<_> = entries.into_iter().collect();
    sorted_entries.sort_by_key(|&((row, col), _)| (row, col));

    sorted_entries
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_csr_matrix() {
        let nrows = 3;
        let ncols = 3;
        let sparse_matrix_triplets = vec![
            Triplet { row: 0, col: 0, value: 1.0 },
            Triplet { row: 1, col: 1, value: 2.0 },
            Triplet { row: 2, col: 2, value: 3.0 },
        ];

        // Expected result
        let expected_values = vec![1.0, 2.0, 3.0];
        let expected_col_indices = vec![0, 1, 2];

        // Invoke the function
        let csr_matrix = build_csr_matrix(nrows, ncols, &sparse_matrix_triplets);

        // Assert results
        assert_eq!(csr_matrix.values(), &expected_values);
        assert_eq!(csr_matrix.col_indices(), &expected_col_indices);
    }

    #[test]
    fn test_build_csr_matrix_complex() {
        let nrows = 4;
        let ncols = 4;
        let sparse_matrix_triplets = vec![
            Triplet { row: 0, col: 0, value: 1.0 },
            Triplet { row: 0, col: 3, value: 2.0 },
            Triplet { row: 1, col: 1, value: 3.0 },
            Triplet { row: 2, col: 0, value: 4.0 },
            Triplet { row: 2, col: 2, value: 5.0 },
            Triplet { row: 3, col: 3, value: 6.0 },
        ];

        // Expected result
        let expected_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_col_indices = vec![0, 3, 1, 0, 2, 3];

        // Invoke the function
        let csr_matrix = build_csr_matrix(nrows, ncols, &sparse_matrix_triplets);

        // Assert results
        assert_eq!(csr_matrix.values(), &expected_values);
        assert_eq!(csr_matrix.col_indices(), &expected_col_indices);
    }
}
