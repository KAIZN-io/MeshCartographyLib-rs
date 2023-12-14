//! # Calculate the face distortion of the mesh
//! Since the area of a mapped patch x(U), U ⊂ parameter space Ω, is computed as ∫ U √det(I)dA, the
//! parameterization is area-preserving if det I = 1, or equivalently σ1σ2 = 1, for all points u ∈ Ω
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
use tri_mesh::{Mesh, FaceID};
use nalgebra::{Point3, Vector3};

pub struct FaceDistortionHelper<'a> {
    mesh_open: &'a Mesh,
    mesh_uv: &'a Mesh,
}

impl<'a> FaceDistortionHelper<'a> {
    pub fn new(mesh_open: &'a Mesh, mesh_uv: &'a Mesh) -> Self {
        FaceDistortionHelper {
            mesh_open,
            mesh_uv,
        }
    }

    pub fn compute_face_distortion(&self) -> f64 {
        let mut total_distortion = 0.0;

        for f in self.mesh_open.face_iter() {
            let area_open = self.triangle_area(self.mesh_open, f);
            let area_uv = self.triangle_area(self.mesh_uv, f);
            total_distortion += (area_open - area_uv).abs();
        }

        total_distortion / self.mesh_open.no_faces() as f64  // Average face distortion
    }

    fn triangle_area(&self, mesh: &Mesh, face: FaceID) -> f64 {
        let (vertex1, vertex2, vertex3) = mesh.face_vertices(face);
        let pt1_vec = mesh.position(vertex1);
        let pt2_vec = mesh.position(vertex2);
        let pt3_vec = mesh.position(vertex3);

        let pt1 = Point3::new(pt1_vec.x, pt1_vec.y, pt1_vec.z);
        let pt2 = Point3::new(pt2_vec.x, pt2_vec.y, pt2_vec.z);
        let pt3 = Point3::new(pt3_vec.x, pt3_vec.y, pt3_vec.z);

        // Compute the vectors representing two sides of the triangle
        let v1 = pt2 - pt1;
        let v2 = pt3 - pt1;

        let cross_product = v1.cross(&v2);
        0.5 * cross_product.norm()  // Half the magnitude of the cross product is the area of the triangle
    }
}
