//! Kaldi-format CMVN (Cepstral Mean-Variance Normalization) parser.
//!
//! Reads a Kaldi binary-format `cmvn.ark` file containing accumulated
//! feature statistics (sums and sums-of-squares), then computes per-dimension
//! mean and inverse standard deviation vectors for normalization.
//!
//! The normalization formula is: `normalized = (feature - mean) * inv_std`

use crate::error::VadError;

/// Per-dimension CMVN statistics for feature normalization.
pub(crate) struct CmvnStats {
    /// Per-dimension means (80-dim).
    pub means: Vec<f32>,
    /// Per-dimension inverse standard deviations (80-dim).
    pub inv_stds: Vec<f32>,
}

impl CmvnStats {
    /// Parse CMVN statistics from a Kaldi binary matrix file.
    ///
    /// The file format is:
    /// - Header: ` BDM ` (space, B, D/F, M, space) for double/float matrix
    /// - 4 bytes: rows (i32, little-endian) — expected 2
    /// - 4 bytes: cols (i32, little-endian) — expected dim+1
    /// - Row 0: accumulated sums (dim values) + count (last value)
    /// - Row 1: accumulated sums of squares (dim values) + 0
    ///
    /// Computes:
    /// - `mean[d] = sums[d] / count`
    /// - `variance[d] = (sum_sq[d] / count) - mean[d]^2`
    /// - `inv_std[d] = 1 / sqrt(max(variance, 1e-20))`
    pub fn from_kaldi_binary(data: &[u8]) -> Result<Self, VadError> {
        // Minimum size: header (5) + row_tag (1+4) + col_tag (1+4) + at least some data
        if data.len() < 15 {
            return Err(VadError::BackendError("CMVN file too small".into()));
        }

        // Skip the initial space-separated token (e.g., empty key before binary data).
        // Kaldi ark files may have " BFM " or " BDM " as a header.
        // Find the 'B' marker that starts the binary section.
        // Format: [optional key] \0 B [type marker] [data]
        // Or for standalone files: ' ' B [type] M ' '
        // We need to find '\0B' or ' B' followed by 'F'/'D' and 'M'
        let b_pos = data
            .iter()
            .position(|&b| b == b'B')
            .ok_or_else(|| VadError::BackendError("CMVN: no binary marker 'B' found".into()))?;
        let mut pos = b_pos + 1;

        if pos >= data.len() {
            return Err(VadError::BackendError("CMVN: truncated after 'B'".into()));
        }

        // Type marker: 'F' = float32, 'D' = float64
        let type_marker = data[pos];
        pos += 1;
        let is_double = match type_marker {
            b'F' => false,
            b'D' => true,
            _ => {
                return Err(VadError::BackendError(format!(
                    "CMVN: unexpected type marker '{}'",
                    type_marker as char
                )))
            }
        };

        // 'M' for matrix
        if pos >= data.len() || data[pos] != b'M' {
            return Err(VadError::BackendError("CMVN: expected 'M' marker".into()));
        }
        pos += 1;

        // Skip space after 'M' if present
        if pos < data.len() && data[pos] == b' ' {
            pos += 1;
        }

        // Read row tag byte (0x04) + rows (i32 LE)
        if pos + 5 > data.len() {
            return Err(VadError::BackendError("CMVN: truncated at rows".into()));
        }
        pos += 1; // skip tag byte (0x04)
        let rows = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        if rows != 2 {
            return Err(VadError::BackendError(format!(
                "CMVN: expected 2 rows, got {rows}"
            )));
        }

        // Read col tag byte (0x04) + cols (i32 LE)
        if pos + 5 > data.len() {
            return Err(VadError::BackendError("CMVN: truncated at cols".into()));
        }
        pos += 1; // skip tag byte
        let cols = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        if cols < 2 {
            return Err(VadError::BackendError(format!(
                "CMVN: expected cols >= 2, got {cols}"
            )));
        }
        let dim = (cols - 1) as usize;

        // Read matrix data
        let elem_size = if is_double { 8 } else { 4 };
        let total_elems = 2 * cols as usize;
        let data_size = total_elems * elem_size;

        if pos + data_size > data.len() {
            return Err(VadError::BackendError(format!(
                "CMVN: file too small for {total_elems} elements"
            )));
        }

        // Parse row 0 (sums + count) and row 1 (sums of squares)
        let read_f64 = |offset: usize| -> f64 {
            if is_double {
                f64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ])
            } else {
                f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as f64
            }
        };

        // Row 0: sums[0..dim], count[dim]
        let row0_start = pos;
        let row1_start = pos + cols as usize * elem_size;

        let count = read_f64(row0_start + dim * elem_size);
        if count < 1.0 {
            return Err(VadError::BackendError(format!(
                "CMVN: count must be >= 1, got {count}"
            )));
        }

        let floor: f64 = 1e-20;
        let mut means = Vec::with_capacity(dim);
        let mut inv_stds = Vec::with_capacity(dim);

        for d in 0..dim {
            let sum = read_f64(row0_start + d * elem_size);
            let sum_sq = read_f64(row1_start + d * elem_size);

            let mean = sum / count;
            let variance = (sum_sq / count) - mean * mean;
            let variance = if variance < floor { floor } else { variance };
            let istd = 1.0 / variance.sqrt();

            means.push(mean as f32);
            inv_stds.push(istd as f32);
        }

        Ok(Self { means, inv_stds })
    }

    /// Apply CMVN normalization to a feature vector in-place.
    #[inline]
    pub fn normalize(&self, features: &mut [f32]) {
        for (i, feat) in features.iter_mut().enumerate() {
            *feat = (*feat - self.means[i]) * self.inv_stds[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedded CMVN file for testing.
    const CMVN_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/firered_cmvn.ark"));

    #[test]
    fn parse_cmvn_dimensions() {
        let stats = CmvnStats::from_kaldi_binary(CMVN_BYTES).unwrap();
        assert_eq!(stats.means.len(), 80);
        assert_eq!(stats.inv_stds.len(), 80);
    }

    #[test]
    fn parse_cmvn_values_match_python() {
        let stats = CmvnStats::from_kaldi_binary(CMVN_BYTES).unwrap();

        // Reference values from Python (firered_reference.py)
        let ref_means: [f64; 5] = [
            10.42295174919564,
            10.862097411631494,
            11.764544378124809,
            12.490164701573908,
            13.25983008289003,
        ];
        let ref_inv_stds: [f64; 5] = [
            0.2494980879825924,
            0.23563235243542163,
            0.23145152525802104,
            0.2332233926481505,
            0.23182660283718737,
        ];

        for i in 0..5 {
            let mean_diff = (stats.means[i] as f64 - ref_means[i]).abs();
            assert!(
                mean_diff < 1e-4,
                "mean[{i}]: rust={}, python={}, diff={mean_diff}",
                stats.means[i],
                ref_means[i]
            );

            let istd_diff = (stats.inv_stds[i] as f64 - ref_inv_stds[i]).abs();
            assert!(
                istd_diff < 1e-4,
                "inv_std[{i}]: rust={}, python={}, diff={istd_diff}",
                stats.inv_stds[i],
                ref_inv_stds[i]
            );
        }
    }

    #[test]
    fn normalize_applies_correctly() {
        let stats = CmvnStats {
            means: vec![1.0, 2.0],
            inv_stds: vec![0.5, 0.25],
        };

        let mut features = vec![3.0, 6.0];
        stats.normalize(&mut features);

        // (3.0 - 1.0) * 0.5 = 1.0
        // (6.0 - 2.0) * 0.25 = 1.0
        assert!((features[0] - 1.0).abs() < 1e-6);
        assert!((features[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_invalid_data() {
        assert!(CmvnStats::from_kaldi_binary(b"").is_err());
        assert!(CmvnStats::from_kaldi_binary(b"too short").is_err());
    }
}
