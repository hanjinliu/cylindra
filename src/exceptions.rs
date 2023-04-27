#[macro_export]
macro_rules! value_error {
    ($msg:expr) => {
        Err(pyo3::exceptions::PyValueError::new_err(
            ($msg).to_string()
        ))
    }
}

#[macro_export]
macro_rules! index_error {
    ($msg:expr) => {
        Err(pyo3::exceptions::PyIndexError::new_err(
            ($msg).to_string()
        ))
    }
}
