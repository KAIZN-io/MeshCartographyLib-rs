extern crate tri_mesh;
use tri_mesh::{Mesh, FaceID};
use nalgebra::Point3;

pub struct AngleDistortionHelper<'a> {
    mesh_open: &'a Mesh,
    mesh_uv: &'a Mesh,
}

impl<'a> AngleDistortionHelper<'a> {
    pub fn new(mesh_open: &'a Mesh, mesh_uv: &'a Mesh) -> Self {
        AngleDistortionHelper {
            mesh_open,
            mesh_uv,
        }
    }

    pub fn compute_angle_distortion(&self) -> f64 {
        let mut total_distortion = 0.0;

        for f in self.mesh_open.face_iter() {
            let angles_open = self.triangle_angles(self.mesh_open, f);
            let angles_uv = self.triangle_angles(self.mesh_uv, f);

            for i in 0..3 {
                total_distortion += (angles_open[i] - angles_uv[i]).abs();
            }
        }

        total_distortion / (3.0 * self.mesh_open.no_faces() as f64)  // Average angle distortion
    }

    fn triangle_angles(&self, mesh: &Mesh, face: FaceID) -> Vec<f64> {
        let (vertex1, vertex2, vertex3) = mesh.face_vertices(face);
        let pt1_vec = mesh.vertex_position(vertex1);
        let pt2_vec = mesh.vertex_position(vertex2);
        let pt3_vec = mesh.vertex_position(vertex3);

        let pt1 = Point3::new(pt1_vec.x, pt1_vec.y, pt1_vec.z);
        let pt2 = Point3::new(pt2_vec.x, pt2_vec.y, pt2_vec.z);
        let pt3 = Point3::new(pt3_vec.x, pt3_vec.y, pt3_vec.z);

        let mut angles = Vec::with_capacity(3);
        angles.push(self.compute_angle(pt1, pt2, pt3));
        angles.push(self.compute_angle(pt2, pt3, pt1));
        angles.push(self.compute_angle(pt3, pt1, pt2));

        angles
    }

    fn compute_angle(&self, a: Point3<f64>, b: Point3<f64>, c: Point3<f64>) -> f64 {
        let ca = a - c;
        let cb = b - c;

        let dot_product = ca.dot(&cb);
        let magnitude_product = ca.norm() * cb.norm();

        dot_product.acos() / magnitude_product
    }
}
