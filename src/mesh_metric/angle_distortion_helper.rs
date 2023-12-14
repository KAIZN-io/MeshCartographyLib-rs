extern crate tri_mesh;
use tri_mesh::{Mesh, FaceID};
use nalgebra::Point3;

struct AngleDistortionHelper<'a> {
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

    fn triangle_angles(&self, mesh: &Mesh, f: FaceID) -> Vec<f64> {
        let pts: Vec<Point3<f64>> = mesh.vertices_around_face(f).iter().map(|&v| mesh.vertex_position(v)).collect();

        let mut angles = Vec::with_capacity(3);
        angles.push(self.compute_angle(pts[0], pts[1], pts[2]));
        angles.push(self.compute_angle(pts[1], pts[2], pts[0]));
        angles.push(self.compute_angle(pts[2], pts[0], pts[1]));

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
