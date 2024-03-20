use super::super::coordinates::{CoordinateSystem, Vector3D};

/// Struct that represents longitudinal constraints.
pub struct Constraint {
    zmax: f32,
    ymax: f32,
    xmax: f32,
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
    pub fn new(nz: f32, ny: f32, nx: f32, dist_min2: f32, dist_max2: f32) -> Self {
        Self {
            zmax: nz - 1.0,
            ymax: ny - 1.0,
            xmax: nx - 1.0,
            dist_min2,
            dist_max2,
        }
    }

    pub fn fast_check_longitudinal(
        &self,
        coord: &CoordinateSystem<f32>,
        end: &Vector3D<f32>,
        y0: f32,
    ) -> CheckResult {
        let point_0y0 = coord.at(0.0, y0, 0.0);
        let dist2_00 = (point_0y0 - end).length2();
        let dist2_01 = (coord.at(0.0, y0, self.xmax) - end).length2();
        let dist2_10 = (coord.at(self.zmax, y0, 0.0) - end).length2();
        let dist2_11 = (coord.at(self.zmax, y0, self.xmax) - end).length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }

        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
        // is invalid.
        if point_0y0.point_to_plane_distance2(&coord.ey, end) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }

    pub fn fast_check_lateral(
        &self,
        coord: &CoordinateSystem<f32>,
        end: &Vector3D<f32>,
        x0: f32,
    ) -> CheckResult {
        let point_00x = coord.at(0.0, 0.0, x0);
        let dist2_00 = (point_00x - end).length2();
        let dist2_01 = (coord.at(0.0, self.ymax, x0) - end).length2();
        let dist2_10 = (coord.at(self.zmax, 0.0, x0) - end).length2();
        let dist2_11 = (coord.at(self.zmax, self.ymax, x0) - end).length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }
        if point_00x.point_to_plane_distance2(&coord.ex, end) > self.dist_max2 {
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
    zmax: f32,
    ymax: f32,
    xmax: f32,
    dist_min2: f32,
    dist_max2: f32,
    cos_max: f32,
}

impl AngleConstraint {
    pub fn new(
        nz: f32,
        ny: f32,
        nx: f32,
        dist_min2: f32,
        dist_max2: f32,
        cos_max: f32,
    ) -> Self {
        Self {
            zmax: nz - 1.0,
            ymax: ny - 1.0,
            xmax: nx - 1.0,
            dist_min2,
            dist_max2,
            cos_max,
        }
    }

    pub fn fast_check_longitudinal(
        &self,
        coord: &CoordinateSystem<f32>,
        end: &Vector3D<f32>,
        y0: f32,
    ) -> CheckResult {
        let point_0y0 = coord.at(0.0, y0, 0.0);
        let dist2_00 = (point_0y0 - end).length2();
        let dist2_01 = (coord.at(0.0, y0, self.xmax) - end).length2();
        let dist2_10 = (coord.at(self.zmax, y0, 0.0) - end).length2();
        let dist2_11 = (coord.at(self.zmax, y0, self.xmax) - end).length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }

        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
        // is invalid.
        if point_0y0.point_to_plane_distance2(&coord.ey, end) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }
    pub fn fast_check_lateral(
        &self,
        coord: &CoordinateSystem<f32>,
        end: &Vector3D<f32>,
        x0: f32,
    ) -> CheckResult {
        let point_00x = coord.at(0.0, 0.0, x0);
        let dist2_00 = (point_00x - end).length2();
        let dist2_01 = (coord.at(0.0, self.ymax, x0) - end).length2();
        let dist2_10 = (coord.at(self.zmax, 0.0, x0) - end).length2();
        let dist2_11 =
            (coord.at(self.zmax, self.ymax, x0) - end)
                .length2();
        if dist2_00 < self.dist_min2 && dist2_01 < self.dist_min2
            && dist2_10 < self.dist_min2 && dist2_11 < self.dist_min2
        {
            return CheckResult::SMALL;
        }
        if point_00x.point_to_plane_distance2(&coord.ex, end) > self.dist_max2 {
            return CheckResult::LARGE;
        }
        return CheckResult::OK;
    }

    /// Check if the displacement vector of the end point is within the constraint.
    /// start_point is in the i-th coordinate system, and end_point is in the (i+1)-th
    /// coordinate system. origin_vector is the displacement vector of the origin of the
    /// (i+1)-th coordinate system and the origin of the i-th coordinate system.
    /// origin_dist2 is simply the square length of origin_vector (just for the
    /// efficiency).
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
