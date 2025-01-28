pub mod traits;
pub mod basic;
pub mod defective;
pub mod filamentous;

pub use traits::{GraphTrait, CylindricGraphTrait};
pub use basic::CylindricGraph;
pub use defective::DefectiveCylindricGraph;
pub use filamentous::FilamentousGraph;
