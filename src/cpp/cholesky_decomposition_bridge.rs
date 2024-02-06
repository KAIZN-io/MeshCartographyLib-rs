use nalgebra::DMatrix;
use autocxx::prelude::*;
use crate::surface_parameterization::harmonic_parameterization_helper::Triplet;

// Including the C++ header file
include_cpp! {
    #include "input.h"
    safety!(unsafe_ffi)
    generate!("eigen_operations")
    generate!("cholesky_decomposition")
}

pub fn solve_cholesky(bb_mtx: &DMatrix<f64>, sparse_matrix_triplets: &[Triplet<f64>]) -> Result<DMatrix<f64>, String> {
    // Convert the triplets into separate vectors for rows, columns, and values
    let rows: Vec<usize> = sparse_matrix_triplets.iter().map(|t| t.row).collect();
    let cols: Vec<usize> = sparse_matrix_triplets.iter().map(|t| t.col).collect();
    let values: Vec<f64> = sparse_matrix_triplets.iter().map(|t| t.value).collect();

    // Assuming you have already created a dense matrix and have its pointer, nrows, and ncols
    let ptr = bb_mtx.as_ptr();
    let nrows = bb_mtx.nrows() as u64; // Cast to u64 first
    let ncols = bb_mtx.ncols() as u64; // Cast to u64 first

    // Convert nrows and ncols to autocxx::c_ulong if required
    let mtx_nrows_c = autocxx::c_ulong::from(nrows);
    let mtx_ncols_c = autocxx::c_ulong::from(ncols);

    // Call the C++ function
    let rows_c_ulong: Vec<autocxx::c_ulong> = rows.iter().map(|&r| autocxx::c_ulong::from(r as u64)).collect();
    let cols_c_ulong: Vec<autocxx::c_ulong> = cols.iter().map(|&c| autocxx::c_ulong::from(c as u64)).collect();

    let rows_ptr = rows_c_ulong.as_ptr();
    let cols_ptr = cols_c_ulong.as_ptr();
    let mut output: Vec<f64> = vec![0.0; bb_mtx.nrows() * bb_mtx.ncols()];

    #[cfg(target_pointer_width = "64")]
    unsafe {
        ffi::cholesky_decomposition(
            ptr,
            mtx_nrows_c,
            mtx_ncols_c,
            rows_ptr,
            cols_ptr,
            values.as_ptr(),
            autocxx::c_ulong::from(rows.len() as u64),
            output.as_mut_ptr()
        )
    };

    // Convert the output to a DMatrix
    let mut xx = DMatrix::zeros(nrows as usize, ncols as usize);
    for i in 0..nrows {
        for j in 0..ncols {
            xx[(i as usize, j as usize)] = output[(i * ncols + j) as usize];
        }
    }

    Ok(xx)
}
