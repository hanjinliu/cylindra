#[macro_export]
/// Macro to create a `ValueError` with a given message.
macro_rules! value_error {
    ($msg:expr) => {
        Err(pyo3::exceptions::PyValueError::new_err(
            ($msg).to_string()
        ))
    }
}

#[macro_export]
/// Macro to create a `IndexError` with a given message.
macro_rules! index_error {
    ($msg:expr) => {
        Err(pyo3::exceptions::PyIndexError::new_err(
            ($msg).to_string()
        ))
    }
}
