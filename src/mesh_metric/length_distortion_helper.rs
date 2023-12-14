//! # Calculate the length distortion of the mesh
//! A parameterization is length-preserving if it is both angle- and area-preserving.
//! In this case the first fundamental form is the identity, i.e., σ1 = σ2 = 1.
//! Only developable surfaces, where these surfaces have zero Gaussian curvature everywhere, admit a perfect length-preserving parameterization.
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Dec-14
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** -
//! - **Todo:** -

extern crate tri_mesh;
use tri_mesh::{Mesh, HalfEdgeID};
use nalgebra::Point3;
use std::collections::HashSet;

pub struct LengthDistortionHelper<'a> {
    mesh_open: &'a Mesh,
    mesh_uv: &'a Mesh,
}

impl<'a> LengthDistortionHelper<'a> {
    pub fn new(mesh_open: &'a Mesh, mesh_uv: &'a Mesh) -> Self {
        LengthDistortionHelper {
            mesh_open,
            mesh_uv,
        }
    }

    pub fn compute_length_distortion(&self) -> f64 {
        let mut total_distortion = 0.0;
        let mut processed_edges = HashSet::new();

        for halfedge in self.mesh_open.halfedge_iter() {
            if !processed_edges.insert(halfedge) {
                // If the edge (represented by this halfedge) has already been processed, skip it
                continue;
            }

            if self.mesh_open.is_edge_on_boundary(halfedge) {
                // Skip boundary edges if necessary
                continue;
            }

            let length_open = self.edge_length(self.mesh_open, halfedge);
            let length_uv = self.edge_length(self.mesh_uv, halfedge);

            total_distortion += (length_open - length_uv).abs();
        }

        total_distortion / self.mesh_open.no_edges() as f64  // Average length distortion
    }


    fn edge_length(&self, mesh: &Mesh, edge: HalfEdgeID) -> f64 {
        let (vertex1, vertex2) = mesh.edge_vertices(edge);

        let pt1_vec = mesh.position(vertex1);
        let pt2_vec = mesh.position(vertex2);

        let pt1 = Point3::new(pt1_vec.x, pt1_vec.y, pt1_vec.z);
        let pt2 = Point3::new(pt2_vec.x, pt2_vec.y, pt2_vec.z);

        (pt1 - pt2).norm()  // The magnitude or length of the vector
    }
}
