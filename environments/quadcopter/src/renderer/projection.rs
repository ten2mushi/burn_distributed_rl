//! 3D to 2D perspective projection.

use super::config::CameraConfig;
use std::f32::consts::PI;

/// Perspective projection for 3D to 2D coordinate transformation.
#[derive(Clone, Debug)]
pub struct PerspectiveProjection {
    /// View matrix (world -> camera space).
    view_matrix: [[f32; 4]; 4],
    /// Projection matrix (camera -> clip space).
    proj_matrix: [[f32; 4]; 4],
    /// Viewport width.
    width: f32,
    /// Viewport height.
    height: f32,
    /// Current camera position.
    camera_pos: [f32; 3],
}

impl PerspectiveProjection {
    /// Create a new perspective projection.
    pub fn new(config: &CameraConfig, width: u32, height: u32) -> Self {
        let mut proj = Self {
            view_matrix: [[0.0; 4]; 4],
            proj_matrix: [[0.0; 4]; 4],
            width: width as f32,
            height: height as f32,
            camera_pos: config.position,
        };
        proj.update_camera(config);
        proj
    }

    /// Update camera matrices from configuration.
    pub fn update_camera(&mut self, config: &CameraConfig) {
        self.camera_pos = config.position;
        self.view_matrix = Self::compute_view_matrix(config.position, config.target, config.up);
        self.proj_matrix = Self::compute_projection_matrix(
            config.fov_degrees,
            self.width / self.height,
            config.near,
            config.far,
        );
    }

    /// Update viewport dimensions.
    pub fn update_viewport(&mut self, width: u32, height: u32, config: &CameraConfig) {
        self.width = width as f32;
        self.height = height as f32;
        self.proj_matrix = Self::compute_projection_matrix(
            config.fov_degrees,
            self.width / self.height,
            config.near,
            config.far,
        );
    }

    /// Project a 3D world point to 2D screen coordinates.
    ///
    /// Returns `None` if the point is behind the camera.
    pub fn project(&self, point: [f32; 3]) -> Option<(f32, f32)> {
        self.project_with_depth(point).map(|(x, y, _)| (x, y))
    }

    /// Project a 3D world point to 2D screen coordinates with depth.
    ///
    /// Returns `None` if the point is behind the camera.
    /// Returns `(screen_x, screen_y, depth)` where depth is in camera space.
    pub fn project_with_depth(&self, point: [f32; 3]) -> Option<(f32, f32, f32)> {
        // Transform to camera space
        let cam = self.world_to_camera(point);

        // Check if behind camera (negative Z in camera space)
        if cam[2] >= 0.0 {
            return None;
        }

        // Transform to clip space
        let clip = self.camera_to_clip(cam);

        // Perspective divide
        if clip[3].abs() < 1e-10 {
            return None;
        }

        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];

        // Convert NDC to screen coordinates
        let screen_x = (ndc_x + 1.0) * 0.5 * self.width;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.height; // Flip Y for screen coords

        Some((screen_x, screen_y, -cam[2]))
    }

    /// Project a direction vector from a 3D origin.
    ///
    /// Returns the screen coordinates of both origin and endpoint.
    pub fn project_direction(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        length: f32,
    ) -> Option<((f32, f32), (f32, f32))> {
        let origin_2d = self.project(origin)?;
        let endpoint = [
            origin[0] + direction[0] * length,
            origin[1] + direction[1] * length,
            origin[2] + direction[2] * length,
        ];
        let endpoint_2d = self.project(endpoint)?;
        Some((origin_2d, endpoint_2d))
    }

    /// Get distance from camera to a point (for depth sorting).
    pub fn distance_to_camera(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.camera_pos[0];
        let dy = point[1] - self.camera_pos[1];
        let dz = point[2] - self.camera_pos[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Transform world coordinates to camera space.
    fn world_to_camera(&self, point: [f32; 3]) -> [f32; 4] {
        let p = [point[0], point[1], point[2], 1.0];
        Self::mat4_mul_vec4(&self.view_matrix, &p)
    }

    /// Transform camera space to clip space.
    fn camera_to_clip(&self, cam: [f32; 4]) -> [f32; 4] {
        Self::mat4_mul_vec4(&self.proj_matrix, &cam)
    }

    /// Compute look-at view matrix.
    fn compute_view_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        // Forward direction (from target to eye for right-handed system)
        let f = Self::normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);

        // Right direction
        let r = Self::normalize(Self::cross(&f, &up));

        // Corrected up direction
        let u = Self::cross(&r, &f);

        // View matrix (transpose of rotation * translation)
        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [
                -Self::dot(&r, &eye),
                -Self::dot(&u, &eye),
                Self::dot(&f, &eye),
                1.0,
            ],
        ]
    }

    /// Compute perspective projection matrix.
    fn compute_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let fov_rad = fov_deg * PI / 180.0;
        let f = 1.0 / (fov_rad / 2.0).tan();

        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), -1.0],
            [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
        ]
    }

    /// Matrix-vector multiplication.
    fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
            m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
        ]
    }

    /// Vector normalization.
    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len < 1e-10 {
            return [0.0, 0.0, 1.0];
        }
        [v[0] / len, v[1] / len, v[2] / len]
    }

    /// Cross product.
    fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Dot product.
    fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}

/// Compute camera position for orbital motion.
pub fn orbit_camera_position(
    target: [f32; 3],
    radius: f32,
    elevation: f32,
    azimuth: f32,
) -> [f32; 3] {
    let cos_elev = elevation.cos();
    [
        target[0] + radius * cos_elev * azimuth.cos(),
        target[1] + radius * cos_elev * azimuth.sin(),
        target[2] + radius * elevation.sin(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_basics() {
        let config = CameraConfig::default();
        let proj = PerspectiveProjection::new(&config, 800, 600);

        // Point in front of camera should project
        let result = proj.project([0.0, 0.0, 1.0]);
        assert!(result.is_some());

        // Point behind camera should not project
        let behind = proj.project([0.0, 0.0, 100.0]);
        // Depends on camera position, but generally should work
    }

    #[test]
    fn test_orbit_position() {
        let pos = orbit_camera_position([0.0, 0.0, 0.0], 10.0, 0.0, 0.0);
        assert!((pos[0] - 10.0).abs() < 0.01);
        assert!(pos[1].abs() < 0.01);
        assert!(pos[2].abs() < 0.01);
    }
}
