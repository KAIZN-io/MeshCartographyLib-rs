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

use nalgebra::{DMatrix, LU};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::AddAssign;

extern crate tri_mesh;
use tri_mesh::Mesh;

use crate::mesh_definition;
use crate::SurfaceParameterization::laplacian_matrix;

use crate::mesh_definition::TexCoord;

struct Triplet<T> {
    row: usize,
    col: usize,
    value: T,
}


#[allow(non_snake_case)]
pub fn harmonic_parameterization(mesh: &Mesh, mesh_tex_coords: &mut mesh_definition::MeshTexCoords, use_uniform_weights: bool) {
    // Set which vertices are constrained (i.e. on the boundary)
    let mut is_constrained = Vec::new();
    for vertex_id in mesh.vertex_iter() {
        is_constrained.push(mesh.is_vertex_on_boundary(vertex_id));
    }

    // build system matrix (clamp negative cotan weights to zero)
    // 1. Get the local geometry and relationships between the mesh vertices
    let L = laplacian_matrix::build_laplace_matrix(mesh, use_uniform_weights);

    // 2. Inject Boundary Constraints -> sets fixed boundary vertices
    let B = set_boundary_constraints(mesh, mesh_tex_coords);

    // 3. Solve the linear equation system
    let result = solve_using_qr_decomposition(&L, &B, is_constrained);

    match result {
        Ok(X) => {
            for (vertex_id, row) in mesh.vertex_iter().zip(X.row_iter()) {
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


pub fn set_boundary_constraints(mesh: &Mesh, mesh_tex_coords: &mut mesh_definition::MeshTexCoords) -> DMatrix<f64> {
    // Build the RHS vector B
    const DIM: usize = 2;
    let mut B = DMatrix::zeros(mesh.no_vertices(), DIM);
    for vertex_id in mesh.vertex_iter() {
        if mesh.is_vertex_on_boundary(vertex_id) {
            if let Some(tex_coord) = mesh_tex_coords.get_tex_coord(vertex_id) {
                let index_as_u32: u32 = *vertex_id; // Dereference to get u32
                let index_as_usize: usize = index_as_u32 as usize; // Cast u32 to usize
                B.set_row(index_as_usize, &nalgebra::RowVector2::new(tex_coord.0, tex_coord.1));
            }
        }
    }

    B
}

#[allow(non_snake_case)]
fn solve_using_qr_decomposition(L: &CsrMatrix<f64>, B: &DMatrix<f64>, is_constrained: Vec<bool>) -> Result<DMatrix<f64>, String> {
    let nrows = L.nrows();
    assert_eq!(4725, nrows);

    let mut idx = vec![usize::MAX; nrows];
    let mut n_dofs = 0;
    let mut BB = DMatrix::zeros(nrows, B.ncols());
    for i in 0..nrows {
        if !is_constrained[i]{
            idx[i] = n_dofs;
            BB.set_row(n_dofs, &B.row(i));
            n_dofs += 1;
        }
    }

    assert_eq!(n_dofs, 4613);  // ! Test for Ellipsoid
    BB.resize_mut(n_dofs, B.ncols(), 0.0); // Resize BB after filling it

    // collect entries for reduced matrix
    // update rhs with constraints
    let triplets = get_tripplets(&L, &B, &mut BB, &idx);

    // ? Build the dense matrix of the inner part of the mesh
    let dense_matrix = build_dense_matrix(&triplets, BB.nrows());

    // Solve the system Lxx = BB using LU decomposition
    let lu = LU::new(dense_matrix.clone());
    let xx = lu.solve(&BB)
        .ok_or("Failed to solve the system using LU decomposition")?;

    // Fill in the solution X
    let mut X = DMatrix::zeros(B.nrows(), B.ncols());
    for i in 0..L.nrows() {
        for j in 0..B.ncols() {
            X[(i, j)] = if idx[i] == usize::MAX { B[(i, j)] } else { xx[(idx[i], j)] };
        }
    }

    Ok(X)
}


fn get_tripplets(L: &CsrMatrix<f64>, B: &DMatrix<f64>, BB: &mut DMatrix<f64>, idx: &[usize]) -> Vec<Triplet<f64>> {
    let mut triplets: Vec<Triplet<f64>> = Vec::new();
    for triplet in L.triplet_iter() {
        let i = triplet.0;
        let j = triplet.1;
        let v = triplet.2;

        if idx[i] != usize::MAX { // row is dof
            if idx[j] != usize::MAX { // col is dof
                triplets.push(Triplet { row: idx[i], col: idx[j], value: *v });
            } else { // col is constraint
                // Update B
                for col in 0..B.ncols() {
                    BB[(idx[i], col)] -= v * B[(j, col)];
                }
            }
        }
    }

    triplets
}


// Function to convert custom triplets to a CSR matrix
fn build_csr_matrix<T: Copy + nalgebra::Scalar + Zero + AddAssign>(nrows: usize, ncols: usize, triplets: &[Triplet<T>]) -> CsrMatrix<T> {
    // Collect the entries
    let entries = collect_entries(triplets);

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

fn build_dense_matrix(triplets: &[Triplet<f64>], n_dofs: usize) -> DMatrix<f64> {
    let csr_matrix = build_csr_matrix(n_dofs, n_dofs, &triplets);

    // Convert CSR matrix to dense matrix
    let mut dense_matrix = DMatrix::zeros(csr_matrix.nrows(), csr_matrix.ncols());
    for triplet in csr_matrix.triplet_iter() {
        let i = triplet.0;
        let j = triplet.1;
        let v = *triplet.2;

        dense_matrix[(i, j)] = v;
    }

    dense_matrix
}


fn collect_entries<T: Copy + nalgebra::Scalar + Zero + AddAssign>(triplets: &[Triplet<T>]) -> Vec<((usize, usize), T)> {
    // ? Oder ist das hier der Bug, wegen der Verwendung von HashMap?
    let mut entries: HashMap<(usize, usize), T> = HashMap::new();
    for triplet in triplets {
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
        let triplets = vec![
            Triplet { row: 0, col: 0, value: 1.0 },
            Triplet { row: 1, col: 1, value: 2.0 },
            Triplet { row: 2, col: 2, value: 3.0 },
        ];

        // Expected result
        let expected_values = vec![1.0, 2.0, 3.0];
        let expected_col_indices = vec![0, 1, 2];

        // Invoke the function
        let csr_matrix = build_csr_matrix(nrows, ncols, &triplets);

        // Assert results
        assert_eq!(csr_matrix.values(), &expected_values);
        assert_eq!(csr_matrix.col_indices(), &expected_col_indices);
    }

    #[test]
    fn test_build_csr_matrix_complex() {
        let nrows = 4;
        let ncols = 4;
        let triplets = vec![
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
        let csr_matrix = build_csr_matrix(nrows, ncols, &triplets);

        // Assert results
        assert_eq!(csr_matrix.values(), &expected_values);
        assert_eq!(csr_matrix.col_indices(), &expected_col_indices);
    }
}






// idx[4716]: 4604
// idx[4717]: 4605
// idx[4718]: 4606
// idx[4719]: 4607
// idx[4720]: 4608
// idx[4721]: 4609
// idx[4722]: 4610
// idx[4723]: 4611
// idx[4724]: 4612
// println!("idx[{}]: {}", i, idx[i]);

// should be:
// idx[4655] = 4598
// idx[4656] = 4599
// idx[4657] = 4600
// idx[4658] = 4601
// idx[4659] = 4602
// idx[4660] = 4603
// idx[4661] = 4604
// idx[4662] = 4605
// idx[4663] = 4606
// idx[4664] = 4607
// idx[4665] = 4608
// idx[4666] = 4609
// idx[4667] = 4610
// idx[4668] = 4611
// idx[4669] = 4612
