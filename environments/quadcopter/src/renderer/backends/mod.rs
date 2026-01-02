//! Rendering backends for quadcopter visualization.

mod image;

pub use self::image::ImageBackend;

/// Result type for rendering operations.
pub type RenderResult<T> = Result<T, RenderError>;

/// Rendering error type.
#[derive(Debug)]
pub enum RenderError {
    /// Backend initialization failed.
    InitError(String),
    /// Rendering operation failed.
    RenderError(String),
    /// I/O error.
    IoError(std::io::Error),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::InitError(msg) => write!(f, "Initialization error: {}", msg),
            RenderError::RenderError(msg) => write!(f, "Render error: {}", msg),
            RenderError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<std::io::Error> for RenderError {
    fn from(e: std::io::Error) -> Self {
        RenderError::IoError(e)
    }
}
