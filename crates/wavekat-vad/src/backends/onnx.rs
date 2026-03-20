//! Shared helpers for ONNX-based VAD backends.

use crate::error::VadError;
use ort::session::Session;

/// Create an ONNX Runtime session from a model file on disk.
pub(crate) fn session_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Session, VadError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_from_file_nonexistent_path() {
        let result = session_from_file("/nonexistent/path/to/model.onnx");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, VadError::BackendError(_)),
            "Expected BackendError, got {err:?}"
        );
    }

    #[test]
    fn session_from_memory_invalid_bytes() {
        let result = session_from_memory(b"not a valid onnx model");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, VadError::BackendError(_)),
            "Expected BackendError, got {err:?}"
        );
    }

    #[test]
    fn session_from_memory_empty_bytes() {
        let result = session_from_memory(b"");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, VadError::BackendError(_)),
            "Expected BackendError, got {err:?}"
        );
    }
}
