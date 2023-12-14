//! # Boundary Matrix
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-04
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

use nalgebra::DMatrix;

extern crate tri_mesh;
use tri_mesh::Mesh;

use crate::mesh_definition;
use crate::io;

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



#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_definition::TexCoord;

    #[test]
    fn test_set_boundary_constraints() {
        // Load B matrix
        let file_path = "test/data/B.csv";
        let B_dense = io::load_csv_to_dmatrix(file_path).expect("Failed to load matrix");

        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = create_mocked_mesh_tex_coords();
        let _B = set_boundary_constraints(&surface_mesh, &mut mesh_tex_coords);

        // Compare B_dense and B with tolerance
        let epsilon = 1e-6;

        // Compare B_dense and B
        for vertex_id in surface_mesh.vertex_iter() {
            // convert vertex_id to usize
            let index_as_u32: u32 = *vertex_id;
            let index_as_usize: usize = index_as_u32 as usize;
            let row_data: Vec<f64> = _B.row(index_as_usize).iter().cloned().collect();
            let row_data_mocked: Vec<f64> = B_dense.row(index_as_usize).iter().cloned().collect();

            assert!(
                (row_data[1] - row_data_mocked[0]).abs() <= epsilon,
                "row_data[1] and row_data_mocked[0] are not equal within tolerance: {} != {}",
                row_data[1],
                row_data_mocked[0]
            );
            assert!(
                (row_data[0] - row_data_mocked[1]).abs() <= epsilon,
                "row_data[0] and row_data_mocked[1] are not equal within tolerance: {} != {}",
                row_data[0],
                row_data_mocked[1]
            );
        }
    }

    fn create_mocked_mesh_tex_coords() -> mesh_definition::MeshTexCoords {
        let surface_mesh = io::load_test_mesh();
        let mut mesh_tex_coords = mesh_definition::MeshTexCoords::new(&surface_mesh);

        for vertex_id in surface_mesh.vertex_iter() {
            mesh_tex_coords.set_tex_coord(vertex_id, TexCoord(0.0, 0.0)); // Initialize to the origin
        }

        // Insert mocked data
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4466).unwrap(), TexCoord(0.0, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4670).unwrap(), TexCoord(0.03571428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4671).unwrap(), TexCoord(0.07142857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4672).unwrap(), TexCoord(0.10714285714285715, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4673).unwrap(), TexCoord(0.14285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4674).unwrap(), TexCoord(0.17857142857142855, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4675).unwrap(), TexCoord(0.2142857142857143, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4676).unwrap(), TexCoord(0.25, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4677).unwrap(), TexCoord(0.2857142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4678).unwrap(), TexCoord(0.3214285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4679).unwrap(), TexCoord(0.3571428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4680).unwrap(), TexCoord(0.39285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4681).unwrap(), TexCoord(0.4285714285714286, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4682).unwrap(), TexCoord(0.46428571428571425, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4683).unwrap(), TexCoord(0.5, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4684).unwrap(), TexCoord(0.5357142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4685).unwrap(), TexCoord(0.5714285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4686).unwrap(), TexCoord(0.6071428571428571, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4687).unwrap(), TexCoord(0.6428571428571428, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4688).unwrap(), TexCoord(0.6785714285714286, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4689).unwrap(), TexCoord(0.7142857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4690).unwrap(), TexCoord(0.75, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4691).unwrap(), TexCoord(0.7857142857142857, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4692).unwrap(), TexCoord(0.8214285714285714, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4693).unwrap(), TexCoord(0.8571428571428572, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4694).unwrap(), TexCoord(0.8928571428571429, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4695).unwrap(), TexCoord(0.9285714285714285, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4696).unwrap(), TexCoord(0.9642857142857142, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4697).unwrap(), TexCoord(1.0, 0.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4698).unwrap(), TexCoord(1.0, 0.035714285714285664));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4699).unwrap(), TexCoord(1.0, 0.07142857142857133));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4700).unwrap(), TexCoord(1.0, 0.10714285714285715));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4701).unwrap(), TexCoord(1.0, 0.14285714285714282));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4702).unwrap(), TexCoord(1.0, 0.17857142857142846));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4703).unwrap(), TexCoord(1.0, 0.2142857142857143));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4704).unwrap(), TexCoord(1.0, 0.24999999999999994));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4705).unwrap(), TexCoord(1.0, 0.28571428571428564));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4706).unwrap(), TexCoord(1.0, 0.32142857142857145));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4707).unwrap(), TexCoord(1.0, 0.3571428571428571));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4708).unwrap(), TexCoord(1.0, 0.3928571428571428));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4709).unwrap(), TexCoord(1.0, 0.42857142857142844));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4710).unwrap(), TexCoord(1.0, 0.46428571428571425));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4711).unwrap(), TexCoord(1.0, 0.4999999999999999));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4712).unwrap(), TexCoord(1.0, 0.5357142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4713).unwrap(), TexCoord(1.0, 0.5714285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4714).unwrap(), TexCoord(1.0, 0.6071428571428571));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4715).unwrap(), TexCoord(1.0, 0.6428571428571427));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4716).unwrap(), TexCoord(1.0, 0.6785714285714284));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4717).unwrap(), TexCoord(1.0, 0.7142857142857144));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4718).unwrap(), TexCoord(1.0, 0.75));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4719).unwrap(), TexCoord(1.0, 0.7857142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4720).unwrap(), TexCoord(1.0, 0.8214285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4721).unwrap(), TexCoord(1.0, 0.857142857142857));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4722).unwrap(), TexCoord(1.0, 0.8928571428571427));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4723).unwrap(), TexCoord(1.0, 0.9285714285714284));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4724).unwrap(), TexCoord(1.0, 0.9642857142857143));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1897).unwrap(), TexCoord(1.0, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3800).unwrap(), TexCoord(0.9642857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1857).unwrap(), TexCoord(0.9285714285714287, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1858).unwrap(), TexCoord(0.892857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1859).unwrap(), TexCoord(0.8571428571428573, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1860).unwrap(), TexCoord(0.8214285714285714, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1861).unwrap(), TexCoord(0.7857142857142857, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1862).unwrap(), TexCoord(0.75, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1863).unwrap(), TexCoord(0.7142857142857144, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1908).unwrap(), TexCoord(0.6785714285714287, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1909).unwrap(), TexCoord(0.642857142857143, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1910).unwrap(), TexCoord(0.6071428571428574, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1867).unwrap(), TexCoord(0.5714285714285714, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1834).unwrap(), TexCoord(0.5357142857142857, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1801).unwrap(), TexCoord(0.5000000000000001, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1802).unwrap(), TexCoord(0.4642857142857144, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1803).unwrap(), TexCoord(0.42857142857142877, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1915).unwrap(), TexCoord(0.3928571428571431, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1917).unwrap(), TexCoord(0.3571428571428571, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(1918).unwrap(), TexCoord(0.32142857142857145, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2069).unwrap(), TexCoord(0.2857142857142858, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2071).unwrap(), TexCoord(0.2500000000000001, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2178).unwrap(), TexCoord(0.21428571428571447, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2123).unwrap(), TexCoord(0.1785714285714288, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2239).unwrap(), TexCoord(0.14285714285714315, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2291).unwrap(), TexCoord(0.10714285714285715, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2540).unwrap(), TexCoord(0.0714285714285715, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(2713).unwrap(), TexCoord(0.03571428571428583, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3099).unwrap(), TexCoord(0.0, 1.0));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3189).unwrap(), TexCoord(0.0, 0.9642857142857146));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3245).unwrap(), TexCoord(0.0, 0.9285714285714287));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3458).unwrap(), TexCoord(0.0, 0.8928571428571433));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3524).unwrap(), TexCoord(0.0, 0.8571428571428573));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3522).unwrap(), TexCoord(0.0, 0.8214285714285721));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3521).unwrap(), TexCoord(0.0, 0.785714285714286));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3553).unwrap(), TexCoord(0.0, 0.75));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(3585).unwrap(), TexCoord(0.0, 0.7142857142857147));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(61).unwrap(), TexCoord(0.0, 0.6785714285714287));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(59).unwrap(), TexCoord(0.0, 0.6428571428571433));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(56).unwrap(), TexCoord(0.0, 0.6071428571428574));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(57).unwrap(), TexCoord(0.0, 0.5714285714285714));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(52).unwrap(), TexCoord(0.0, 0.535714285714286));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(101).unwrap(), TexCoord(0.0, 0.5000000000000001));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(100).unwrap(), TexCoord(0.0, 0.46428571428571475));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(99).unwrap(), TexCoord(0.0, 0.42857142857142877));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(47).unwrap(), TexCoord(0.0, 0.39285714285714346));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(45).unwrap(), TexCoord(0.0, 0.35714285714285743));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(43).unwrap(), TexCoord(0.0, 0.32142857142857145));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(41).unwrap(), TexCoord(0.0, 0.28571428571428614));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(39).unwrap(), TexCoord(0.0, 0.2500000000000001));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(37).unwrap(), TexCoord(0.0, 0.2142857142857148));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(92).unwrap(), TexCoord(0.0, 0.1785714285714288));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(91).unwrap(), TexCoord(0.0, 0.1428571428571435));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(90).unwrap(), TexCoord(0.0, 0.10714285714285748));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4336).unwrap(), TexCoord(0.0, 0.0714285714285715));
        mesh_tex_coords.set_tex_coord(surface_mesh.vertex_iter().nth(4467).unwrap(), TexCoord(0.0, 0.03571428571428616));

        mesh_tex_coords
    }
}
