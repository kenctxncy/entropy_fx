#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
use rand::Rng;

/// Binary matrix: Vec<Vec<bool>> where true = 1, false = 0
pub type BinaryMatrix = Vec<Vec<bool>>;

/// Codeword: vector of bits
pub type Codeword = Vec<bool>;

/// Message: vector of k bits
pub type Message = Vec<bool>;

/// Syndrome: vector of p bits
pub type Syndrome = Vec<bool>;

/// Information about error correction result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorInfo {
    /// No error detected
    NoError,
    /// Single error at given position (0-indexed)
    SingleError(usize),
    /// Multiple errors or uncorrectable error pattern
    Uncorrectable,
}

/// Systematic error-correcting code
#[derive(Debug, Clone)]
pub struct SystematicCode {
    /// Generator matrix P = [`U_k` | `H_p`] (k rows, n columns)
    pub generator: BinaryMatrix,
    /// Parity check matrix H = [`H_p^T` | `I_p`] (p rows, n columns)
    pub parity_check: BinaryMatrix,
    /// Message length (number of information bits)
    pub k: usize,
    /// Codeword length (total number of bits)
    pub n: usize,
    /// Number of parity bits
    pub p: usize,
}

/// Build `k`×`k` identity matrix
///
/// # Arguments
///
/// * `k` - Size of the identity matrix
///
/// # Returns
///
/// Identity matrix of size `k` × `k`
#[must_use]
pub fn build_identity_matrix(k: usize) -> BinaryMatrix {
    let mut matrix = vec![vec![false; k]; k];
    for (i, row) in matrix.iter_mut().enumerate().take(k) {
        row[i] = true;
    }
    matrix
}

/// Build parity submatrix `H_p` (`k` rows, `p` columns)
///
/// Uses algorithm from Python: increment binary numbers starting from 3,
/// skipping those with only one bit set
///
/// # Arguments
///
/// * `k` - Number of rows
/// * `p` - Number of columns
///
/// # Returns
///
/// Parity submatrix `H_p` of size `k` × `p`
#[must_use]
pub fn build_parity_submatrix(k: usize, p: usize) -> BinaryMatrix {
    let mut h_p = Vec::with_capacity(k);
    let mut i = 3_u32;

    while h_p.len() < k {
        let bin_str = format!("{i:b}");
        let bit_count = bin_str.chars().filter(|&c| c == '1').count();

        // Skip numbers with only one bit set
        if bit_count > 1 {
            // Pad binary string to p characters on the left (like Python's zfill)
            // If longer than p, take the least significant p bits (rightmost)
            let padded = match bin_str.len().cmp(&p) {
                std::cmp::Ordering::Less => {
                    format!("{}{}", "0".repeat(p - bin_str.len()), bin_str)
                }
                std::cmp::Ordering::Greater => {
                    // Take the rightmost p characters (LSBs)
                    bin_str.chars().skip(bin_str.len() - p).collect::<String>()
                }
                std::cmp::Ordering::Equal => bin_str.clone(),
            };
            let mut row = vec![false; p];
            // Convert each character to bool (leftmost bit is MSB, rightmost is LSB)
            for (idx, ch) in padded.chars().enumerate() {
                if ch == '1' {
                    row[idx] = true;
                }
            }
            h_p.push(row);
        }
        i += 1;
    }

    h_p
}

/// Transpose a binary matrix
///
/// # Arguments
///
/// * `matrix` - Matrix to transpose
///
/// # Returns
///
/// Transposed matrix
#[must_use]
fn transpose_matrix(matrix: &BinaryMatrix) -> BinaryMatrix {
    if matrix.is_empty() {
        return vec![];
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![false; rows]; cols];

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            transposed[j][i] = val;
        }
    }

    transposed
}

/// Build generator matrix P = [`U_k` | `H_p`] and parity check matrix H = [`H_p^T` | `I_p`]
///
/// # Arguments
///
/// * `k` - Message length (number of information bits)
/// * `n` - Codeword length (total number of bits)
///
/// # Returns
///
/// A `SystematicCode` structure with generator and parity check matrices
#[must_use]
pub fn build_generator_matrix(k: usize, n: usize) -> SystematicCode {
    let p = n - k;

    // Build identity matrix U_k
    let u_k = build_identity_matrix(k);

    // Build parity submatrix H_p
    let h_p = build_parity_submatrix(k, p);

    // Build generator matrix P = [U_k | H_p]
    let mut generator = Vec::with_capacity(k);
    for (i, u_row) in u_k.iter().enumerate() {
        let mut row = u_row.clone();
        row.extend_from_slice(&h_p[i]);
        generator.push(row);
    }

    // Build parity check matrix H = [H_p^T | I_p]
    let h_p_transposed = transpose_matrix(&h_p);
    let i_p = build_identity_matrix(p);

    let mut parity_check = Vec::with_capacity(p);
    for (i, h_row) in h_p_transposed.iter().enumerate() {
        let mut row = h_row.clone();
        row.extend_from_slice(&i_p[i]);
        parity_check.push(row);
    }

    SystematicCode {
        generator,
        parity_check,
        k,
        n,
        p,
    }
}

/// Build parity check matrix from `H_p`
///
/// # Arguments
///
/// * `h_p` - Parity submatrix `H_p`
///
/// # Returns
///
/// Parity check matrix H = [`H_p^T` | `I_p`]
#[must_use]
pub fn build_parity_check_from_hp(h_p: &BinaryMatrix) -> BinaryMatrix {
    let p = h_p[0].len();
    let h_p_transposed = transpose_matrix(h_p);
    let i_p = build_identity_matrix(p);

    let mut parity_check = Vec::with_capacity(p);
    for (i, h_row) in h_p_transposed.iter().enumerate() {
        let mut row = h_row.clone();
        row.extend_from_slice(&i_p[i]);
        parity_check.push(row);
    }

    parity_check
}

/// Encode message using generator matrix or `H_p^T` (as in Python)
///
/// Python uses `H_t` (transpose of `H_p`) to compute checksums
///
/// # Arguments
///
/// * `message` - Message bits to encode
/// * `code` - Systematic code structure
///
/// # Returns
///
/// Encoded codeword
#[must_use]
pub fn encode_message(message: &Message, code: &SystematicCode) -> Codeword {
    // Extract H_p from generator matrix (last p columns)
    let h_p: BinaryMatrix = code
        .generator
        .iter()
        .map(|row| Vec::from(&row[code.k..]))
        .collect();

    // Transpose H_p to get H_t
    let h_t = transpose_matrix(&h_p);

    // Compute checksums: for each row in H_t, XOR the message bits where H_t[row][j] == 1
    let mut checksums = Vec::with_capacity(code.p);
    for row in &h_t {
        let mut checksum = false;
        for (j, &bit) in message.iter().enumerate() {
            if row[j] {
                checksum ^= bit;
            }
        }
        checksums.push(checksum);
    }

    // Codeword = message + checksums
    let mut codeword = message.clone();
    codeword.extend_from_slice(&checksums);
    codeword
}

/// Compute syndrome: H * codeword (mod 2)
///
/// # Arguments
///
/// * `parity_check` - Parity check matrix H
/// * `codeword` - Received codeword
///
/// # Returns
///
/// Syndrome vector
#[must_use]
pub fn compute_syndrome(parity_check: &BinaryMatrix, codeword: &Codeword) -> Syndrome {
    let mut syndrome = Vec::with_capacity(parity_check.len());

    for row in parity_check {
        let mut sum = false;
        for (j, &bit) in codeword.iter().enumerate() {
            if row[j] {
                sum ^= bit;
            }
        }
        syndrome.push(sum);
    }

    syndrome
}

/// Correct error in received codeword
///
/// # Arguments
///
/// * `parity_check` - Parity check matrix H
/// * `received` - Received codeword (possibly with errors)
///
/// # Returns
///
/// A tuple `(corrected, error_info)` where:
/// - `corrected` is the corrected codeword
/// - `error_info` contains information about detected/corrected errors
#[must_use]
pub fn correct_error(parity_check: &BinaryMatrix, received: &Codeword) -> (Codeword, ErrorInfo) {
    let syndrome = compute_syndrome(parity_check, received);

    // Check if syndrome is all zeros (no error)
    if syndrome.iter().all(|&b| !b) {
        return (received.clone(), ErrorInfo::NoError);
    }

    // Try to find matching column in parity check matrix
    for j in 0..parity_check[0].len() {
        let mut matches = true;
        for (i, &syndrome_bit) in syndrome.iter().enumerate() {
            if parity_check[i][j] != syndrome_bit {
                matches = false;
                break;
            }
        }
        if matches {
            // Found single error at position j
            let mut corrected = received.clone();
            corrected[j] = !corrected[j];
            return (corrected, ErrorInfo::SingleError(j));
        }
    }

    // No matching column found - multiple errors or uncorrectable
    (received.clone(), ErrorInfo::Uncorrectable)
}

/// Inject a single error into codeword with given probability
///
/// # Arguments
///
/// * `codeword` - Original codeword
/// * `error_probability` - Probability of injecting an error (0.0 to 1.0)
///
/// # Returns
///
/// A tuple `(modified, error_position)` where:
/// - `modified` is the codeword with possibly injected error
/// - `error_position` is `Some(position)` if error was injected, `None` otherwise
#[must_use]
pub fn inject_single_error(
    codeword: &Codeword,
    error_probability: f64,
) -> (Codeword, Option<usize>) {
    let mut rng = rand::rng();
    let should_inject = rng.random_range(0.0..1.0) < error_probability;

    if should_inject {
        let error_pos = rng.random_range(0..codeword.len());
        let mut modified = codeword.clone();
        modified[error_pos] = !modified[error_pos];
        (modified, Some(error_pos))
    } else {
        (codeword.clone(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::common::compute_n_from_k;

    #[test]
    fn test_compute_n_from_k() {
        // For k=4, should get n=7, p=3 (example from methodical text)
        let (n, p) = compute_n_from_k(4);
        assert_eq!(n, 7);
        assert_eq!(p, 3);

        // For k=53 (from Python), should get reasonable n
        let (n, p) = compute_n_from_k(53);
        assert!(n >= 53);
        assert_eq!(p, n - 53);
    }

    #[test]
    fn test_build_identity_matrix() {
        let matrix = build_identity_matrix(3);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert!(matrix[0][0]);
        assert!(!matrix[0][1]);
        assert!(!matrix[0][2]);
        assert!(!matrix[1][0]);
        assert!(matrix[1][1]);
        assert!(!matrix[1][2]);
        assert!(!matrix[2][0]);
        assert!(!matrix[2][1]);
        assert!(matrix[2][2]);
    }

    #[test]
    fn test_build_parity_submatrix() {
        let h_p = build_parity_submatrix(4, 3);
        assert_eq!(h_p.len(), 4);
        assert_eq!(h_p[0].len(), 3);

        // First row should be binary representation of 3 = 011
        assert!(!h_p[0][0]);
        assert!(h_p[0][1]);
        assert!(h_p[0][2]);
    }

    #[test]
    fn test_build_generator_matrix() {
        let code = build_generator_matrix(4, 7);
        assert_eq!(code.k, 4);
        assert_eq!(code.n, 7);
        assert_eq!(code.p, 3);
        assert_eq!(code.generator.len(), 4);
        assert_eq!(code.generator[0].len(), 7);
        assert_eq!(code.parity_check.len(), 3);
        assert_eq!(code.parity_check[0].len(), 7);

        // Generator matrix should start with identity
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(code.generator[i][j], i == j);
            }
        }
    }

    #[test]
    fn test_encode_message() {
        let code = build_generator_matrix(4, 7);
        let message = vec![true, false, true, false];
        let codeword = encode_message(&message, &code);

        assert_eq!(codeword.len(), 7);
        // First k bits should match message
        assert_eq!(codeword[0..4], message[..]);
    }

    #[test]
    fn test_compute_syndrome_no_error() {
        let code = build_generator_matrix(4, 7);
        let message = vec![true, false, true, false];
        let codeword = encode_message(&message, &code);
        let syndrome = compute_syndrome(&code.parity_check, &codeword);

        // Syndrome should be all zeros for valid codeword
        assert!(syndrome.iter().all(|&b| !b));
    }

    #[test]
    fn test_correct_error() {
        let code = build_generator_matrix(4, 7);
        let message = vec![true, false, true, false];
        let codeword = encode_message(&message, &code);

        // Inject error at position 2
        let mut received = codeword.clone();
        received[2] = !received[2];

        let (corrected, error_info) = correct_error(&code.parity_check, &received);

        match error_info {
            ErrorInfo::SingleError(pos) => {
                assert_eq!(pos, 2);
                assert_eq!(corrected, codeword);
            }
            _ => panic!("Expected single error"),
        }
    }

    #[test]
    fn test_correct_error_no_error() {
        let code = build_generator_matrix(4, 7);
        let message = vec![true, false, true, false];
        let codeword = encode_message(&message, &code);

        let (corrected, error_info) = correct_error(&code.parity_check, &codeword);

        assert_eq!(error_info, ErrorInfo::NoError);
        assert_eq!(corrected, codeword);
    }
}
