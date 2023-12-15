//! # Create the tesselation of the UV mesh monotile
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

use nalgebra::{DMatrix, DVector, Vector2, Matrix2, SVD};

pub struct Tessellation;

impl Tessellation {
    pub fn calculate_angle(&self, border1: &[Vector2<f64>], border2: &[Vector2<f64>]) -> f64 {
        let dir1 = self.fit_line(border1);
        let dir2 = self.fit_line(border2);

        let dot = dir1.dot(&dir2);
        let det = dir1.x * dir2.y - dir1.y * dir2.x;

        let angle = det.atan2(dot);
        let angle_in_degrees = angle * (180.0 / std::f64::consts::PI);

        // Normalize to [0, 360)
        if angle_in_degrees < 0.0 {
            angle_in_degrees + 360.0
        } else {
            angle_in_degrees
        }
    }

    fn fit_line(&self, points: &[Vector2<f64>]) -> Vector2<f64> {
        let n = points.len() as f64;
        let mean = points.iter().sum::<Vector2<f64>>() / n;

        let mut cov = Matrix2::zeros();
        for p in points {
            let centered = p - mean;
            cov += centered * centered.transpose();
        }
        cov /= n;

        // Find the eigenvector of the covariance matrix corresponding to the largest eigenvalue
        let svd = SVD::new(cov, true, true);
        svd.v_t.unwrap().column(0).into()
    }

    fn custom_rotate(&self, pt: Vector2<f64>, angle_radians: f64) -> Vector2<f64> {
        let cos_theta = angle_radians.cos();
        let sin_theta = angle_radians.sin();

        let x_prime = pt.x * cos_theta - pt.y * sin_theta;
        let y_prime = pt.x * sin_theta + pt.y * cos_theta;

        Vector2::new(x_prime, y_prime)
    }

    pub fn order_data(&self, vec: &mut Vec<Vector2<f64>>) {
        let size = vec.len();
        let mut x = DVector::from_iterator(size, vec.iter().map(|v| v.x));
        let y = DVector::from_iterator(size, vec.iter().map(|v| v.y));

        // Creating the design matrix for linear regression
        let a = DVector::from_element(size, 1.0);
        let b = DMatrix::from_columns(&[x.clone(), a]);

        // Perform linear regression using SVD
        let svd = b.svd(true, true);
        let coeffs = svd.solve(&y, std::f64::EPSILON).unwrap();
        let m = coeffs[0];
        let coeff_b = coeffs[1]; // Renamed to avoid conflict with 'b' in the closure

        // Check if all x-values are the same (vertical line)
        let vertical_line = x.max() - x.min() < std::f64::EPSILON;

        // Sort the vector based on the parameter t
        vec.sort_by(|a, b| {
            if vertical_line {
                a.y.partial_cmp(&b.y).unwrap()
            } else {
                let ta = (a.x + m * (a.y - coeff_b)) / (1.0 + m * m).sqrt();
                let tb = (b.x + m * (b.y - coeff_b)) / (1.0 + m * m).sqrt();
                ta.partial_cmp(&tb).unwrap()
            }
        });
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use nalgebra::Vector2;

    #[test]
    fn test_order_data() {
        // Initialize a Tessellation instance
        let tessellation = Tessellation;

        let mut points = vec![
            Vector2::new(1.0, 2.0),
            Vector2::new(2.0, 3.0),
            Vector2::new(0.5, 1.5),
        ];

        // Call the order_data function
        tessellation.order_data(&mut points);

        // check if the points are sorted by x-coordinate:
        assert!(points.windows(2).all(|w| w[0].x <= w[1].x));
    }

    // #[test]
    // fn test_fit_line() {
    //     let tessellation = Tessellation;
    //     let points = vec![
    //         Vector2::new(1.0, 2.0),
    //         Vector2::new(2.0, 3.0),
    //         Vector2::new(3.0, 4.0),
    //     ];
    //     let fitted_line = tessellation.fit_line(&points);
    //     // Expect the line to have a certain direction (example values)
    //     assert!((fitted_line.x - 0.7).abs() < 1e-5);
    //     assert!((fitted_line.y - 0.7).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_90_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(0.0, 1.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 90 degrees
    //     assert!((angle - 90.0).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_180_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(-1.0, 0.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 180 degrees
    //     assert!((angle - 180.0).abs() < 1e-5);
    // }

    // #[test]
    // fn test_calculate_angle_45_degrees() {
    //     let tessellation = Tessellation;
    //     let border1 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    //     let border2 = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 1.0)];
    //     let angle = tessellation.calculate_angle(&border1, &border2);
    //     // Expect angle to be close to 45 degrees

    //     println!("angle: {}", angle);
    //     assert!((angle - 45.0).abs() < 1e-5);
    // }

    #[test]
    fn test_custom_rotate_90_degrees() {
        let tessellation = Tessellation;
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI / 2.0); // Rotate 90 degrees
        assert!((rotated_point.x.abs() < 1e-5) && ((rotated_point.y - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_custom_rotate_180_degrees() {
        let tessellation = Tessellation;
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI); // Rotate 180 degrees
        assert!(((rotated_point.x + 1.0).abs() < 1e-5) && (rotated_point.y.abs() < 1e-5));
    }

    #[test]
    fn test_custom_rotate_45_degrees() {
        let tessellation = Tessellation;
        let point = Vector2::new(1.0, 0.0);
        let rotated_point = tessellation.custom_rotate(point, PI / 4.0); // Rotate 45 degrees
        let sqrt2_over_2 = (2.0f64).sqrt() / 2.0;
        assert!(((rotated_point.x - sqrt2_over_2).abs() < 1e-5) && ((rotated_point.y - sqrt2_over_2).abs() < 1e-5));
    }
}
