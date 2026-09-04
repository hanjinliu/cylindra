#[derive(Clone)]
pub struct Reservoir {
    temperature_diff: f32,
    // temperature: f32,
    num_iters: usize,
    time_constant: f32,
}

impl Reservoir {
    pub fn new(temperature: f32, time_constant: f32) -> Self {
        if time_constant <= 0.0 {
            panic!("Time constant must be positive.");
        }
        let initial_temperature = temperature;
        let temperature0 = initial_temperature;
        Self {
            temperature_diff: temperature0,
            num_iters: 0,
            time_constant,
        }
    }

    /// Cool the reservoir to state t=n.
    pub fn cool(&mut self, n: usize) {
        self.num_iters = n;
    }

    /// Calculate the probability of accepting a state with energy difference de.
    pub fn prob(&self, de: f32) -> f32 {
        if de <= 0.0 {
            1.0
        } else {
            (-de / self.temperature()).exp()
        }
    }

    /// Return the current temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature_diff * (-(self.num_iters as f32) / self.time_constant).exp()
    }

    /// Initialize the reservoir.
    pub fn initialize(&mut self) {
        self.num_iters = 0;
    }

    pub fn time_constant(&self) -> f32 {
        self.time_constant
    }
}
