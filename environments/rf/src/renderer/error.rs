//! Renderer error types.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// Errors that can occur during rendering operations.
#[derive(Debug)]
pub enum RenderError {
    /// I/O error (file operations)
    Io(io::Error),
    /// Image encoding error
    ImageEncoding(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// History buffer is empty
    EmptyHistory,
    /// Plot type not supported by backend
    UnsupportedPlotType(String),
    /// Window creation failed (realtime feature)
    WindowCreation(String),
    /// GIF encoding error
    GifEncoding(String),
    /// Path error
    InvalidPath(PathBuf),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Io(e) => write!(f, "I/O error: {}", e),
            RenderError::ImageEncoding(msg) => write!(f, "Image encoding error: {}", msg),
            RenderError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            RenderError::EmptyHistory => write!(f, "History buffer is empty"),
            RenderError::UnsupportedPlotType(name) => {
                write!(f, "Unsupported plot type: {}", name)
            }
            RenderError::WindowCreation(msg) => write!(f, "Window creation failed: {}", msg),
            RenderError::GifEncoding(msg) => write!(f, "GIF encoding error: {}", msg),
            RenderError::InvalidPath(path) => write!(f, "Invalid path: {:?}", path),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RenderError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for RenderError {
    fn from(err: io::Error) -> Self {
        RenderError::Io(err)
    }
}

#[cfg(feature = "render")]
impl From<image::ImageError> for RenderError {
    fn from(err: image::ImageError) -> Self {
        RenderError::ImageEncoding(err.to_string())
    }
}

/// Result type alias for render operations.
pub type RenderResult<T> = Result<T, RenderError>;
