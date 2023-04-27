use super::super::coordinates::{CoordinateSystem, Vector3D};

pub struct Constraint {
    nz: usize,
    ny: usize,
    nx: usize,
    dist_min2: f32,
    dist_max2: f32,
}

#[derive(PartialEq)]
pub enum CheckResult {
    SMALL,
    LARGE,
    OK
}

impl Constraint {
    pub fn new(nz: usize, ny: usize, nx: usize, dist_min2: f32, dist_max2: f32) -> Self {
        Self {nz, ny, nx, dist_min2, dist_max2}
    }

    pub fn fast_check_longitudinal(
        &self,
        coord: &CoordinateSystem<f32>,
        end_point: &Vector3D<f32>,
        y0: f32,
    ) -> CheckResult {
        let point_0y0 = coord.at(0.0, y0, 0.0);
        let end = end_point.clone();
        let dist2_00 = (point_0y0 - end).length2();
        let dist2_01 = (coord.at(0.0, y0, (self.nx - 1) as f32) - end).length2();
        let dist2_10 = (coord.at((self.nz - 1) as f32, y0, 0.0) - end).length2();
        let dist2_11 = (coord.at((self.nz - 1) as f32, y0, (self.nx - 1) as f32) - end).length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }

        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
        // is invalid.
        if point_0y0.point_to_plane_distance2(&coord.ey, end_point) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }

    pub fn fast_check_lateral(
        &self,
        coord: &CoordinateSystem<f32>,
        end_point: &Vector3D<f32>,
        x0: usize,
    ) -> CheckResult {
        let point_00x = coord.at(0.0, 0.0, x0 as f32);
        let end = end_point.clone();
        let dist2_00 = (point_00x - end).length2();
        let dist2_01 = (coord.at(0.0, (self.ny - 1) as f32, x0 as f32) - end).length2();
        let dist2_10 = (coord.at((self.nz - 1) as f32, 0.0, x0 as f32) - end).length2();
        let dist2_11 =
            (coord.at((self.nz - 1) as f32, (self.ny - 1) as f32, x0 as f32) - end)
                .length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }
        if point_00x.point_to_plane_distance2(&coord.ex, end_point) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }

    pub fn check_constraint(
        &self,
        start_point: &Vector3D<f32>,
        end_point: &Vector3D<f32>,
    ) -> bool {
        let dist2 = (start_point - end_point).length2();
        if dist2 < self.dist_min2 || dist2 > self.dist_max2 {
            return true;
        }
        return false;
    }
}

pub struct AngleConstraint {
    nz: usize,
    ny: usize,
    nx: usize,
    dist_min2: f32,
    dist_max2: f32,
    cos_max: f32,
}

impl AngleConstraint {
    pub fn new(
        nz: usize,
        ny: usize,
        nx: usize,
        dist_min2: f32,
        dist_max2: f32,
        cos_max: f32,
    ) -> Self {
        Self { nz, ny, nx, dist_min2, dist_max2, cos_max }
    }

    pub fn fast_check_longitudinal(
        &self,
        coord: &CoordinateSystem<f32>,
        end_point: &Vector3D<f32>,
        y0: f32,
    ) -> CheckResult {
        let point_0y0 = coord.at(0.0, y0, 0.0);
        let end = end_point.clone();
        let dist2_00 = (point_0y0 - end).length2();
        let dist2_01 = (coord.at(0.0, y0, (self.nx - 1) as f32) - end).length2();
        let dist2_10 = (coord.at((self.nz - 1) as f32, y0, 0.0) - end).length2();
        let dist2_11 = (coord.at((self.nz - 1) as f32, y0, (self.nx - 1) as f32) - end).length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }

        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
        // is invalid.
        if point_0y0.point_to_plane_distance2(&coord.ey, end_point) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }
    pub fn fast_check_lateral(
        &self,
        coord: &CoordinateSystem<f32>,
        end_point: &Vector3D<f32>,
        x0: usize,
    ) -> CheckResult {
        let point_00x = coord.at(0.0, 0.0, x0 as f32);
        let end = end_point.clone();
        let dist2_00 = (point_00x - end).length2();
        let dist2_01 = (coord.at(0.0, (self.ny - 1) as f32, x0 as f32) - end).length2();
        let dist2_10 = (coord.at((self.nz - 1) as f32, 0.0, x0 as f32) - end).length2();
        let dist2_11 =
            (coord.at((self.nz - 1) as f32, (self.ny - 1) as f32, x0 as f32) - end)
                .length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }
        if point_00x.point_to_plane_distance2(&coord.ex, end_point) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }

    pub fn check_constraint(
        &self,
        start_point: &Vector3D<f32>,
        end_point: &Vector3D<f32>,
        origin_vector: &Vector3D<f32>,
        origin_dist2: f32,
    ) -> bool {
        let vec = end_point - start_point;
        let dist2 = vec.length2();
        if dist2 < self.dist_min2 || dist2 > self.dist_max2 {
            return true;
        }

        // Use formula: a.dot(b) = |a|*|b|*cos(C)
        let ab = (dist2 * origin_dist2).sqrt();
        let cos = (vec.dot(origin_vector)).abs() / ab;

        if cos < self.cos_max {
            // check angle of displacement vector of origins and that of
            // points of interests. Smaller cosine means larger skew.
            return true;
        }
        return false;
    }
}
