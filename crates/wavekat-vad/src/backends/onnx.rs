//! Shared helpers for ONNX-based VAD backends.

use crate::error::VadError;
use ort::session::Session;

/// Create an ONNX Runtime session from a model file on disk.
pub(crate) fn session_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Session, VadError> {
    Session::builder()
        .map_err(|e| VadError::BackendError(format!("failed to create session builder: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| VadError::BackendError(format!("failed to set intra threads: {e}")))?
        .commit_from_file(path)
        .map_err(|e| VadError::BackendError(format!("failed to load ONNX model: {e}")))
}

/// Create an ONNX Runtime session from model bytes in memory.
pub(crate) fn session_from_memory(model_bytes: &[u8]) -> Result<Session, VadError> {
    Session::builder()
        .map_err(|e| VadError::BackendError(format!("failed to create session builder: {e}")))?
        .with_intra_threads(1)
        .map_err(|e| VadError::BackendError(format!("failed to set intra threads: {e}")))?
        .commit_from_memory(model_bytes)
        .map_err(|e| VadError::BackendError(format!("failed to load ONNX model: {e}")))
}
