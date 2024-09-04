#[derive(Clone, PartialEq, Eq)]
/// Current state of the annealing model
pub enum OptimizationState {
    NotConverged,  // Optimization is not converged yet
    Converged,  // Optimization converged
    Failed,  // Optimization failed due to wrong parameters
}
