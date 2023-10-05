#[derive(Clone)]
pub struct Reservoir {
    temperature_diff: f32,
    temperature: f32,
    time_constant: f32,
    min_temperature: f32,
}

impl Reservoir {
    pub fn new(temperature: f32, time_constant: f32, min_temperature: f32) -> Self {
        if min_temperature < 0.0 {
            panic!("Minimum temperature must be positive");
        } else if temperature < min_temperature {
            panic!("Initial temperature must be greater than minimum temperature");
        } else if time_constant <= 0.0 {
            panic!("Time constant must be positive.");
        }
        let initial_temperature = temperature;
        let temperature0 = initial_temperature - min_temperature;
        Self {
            temperature_diff: temperature0,
            temperature,
            time_constant,
            min_temperature,
        }
    }

    /// Cool the reservoir to state t=n.
    pub fn cool(&mut self, n: usize) {
        self.temperature =
            self.temperature_diff * (-(n as f32) / self.time_constant).exp()
            + self.min_temperature;
    }

    /// Calculate the probability of accepting a state with energy difference de.
    pub fn prob(&self, de: f32) -> f32 {
        if de < 0.0 {
            1.0
        } else {
            (-de / self.temperature).exp()
        }
    }

    /// Return the current temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Initialize the reservoir.
    pub fn initialize(&mut self) {
        self.temperature = self.temperature_diff + self.min_temperature;
    }

    pub fn time_constant(&self) -> f32 {
        self.time_constant
    }
}
