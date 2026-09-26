use core::fmt;

#[derive(Debug)]
pub enum XrfError {
    /// There was a panic during parallel training or prediction
    ParallelCodePanic,
    /// The series of `Walk` steps was not a correct depth-first traversal of the forest
    WalkAggregationFailure,
}

impl fmt::Display for XrfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XrfError::ParallelCodePanic => write!(f, "Panic in the parallel code"),
            XrfError::WalkAggregationFailure => {
                write!(f, "Walk is a corrupted representation of a forest")
            }
        }
    }
}

impl std::error::Error for XrfError {}
