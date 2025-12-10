#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
use rand::Rng;

/// Information about error detection/correction result for Hamming code
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HammingErrorInfo {
    /// No error detected
    NoError,
    /// Single error at given position (1-indexed, as per Hamming code convention)
    SingleError(usize),
    /// Double error detected (can be detected but not corrected)
    DoubleError,
}

/// Hamming code structure
#[derive(Debug, Clone)]
pub struct HammingCode {
    /// Number of information bits
    pub k: usize,
    /// Total codeword length (without parity bit for double error detection)
    pub n: usize,
    /// Number of parity bits
    pub p: usize,
}

/// Compute n and p from k for Hamming code
/// For Hamming code: n = 2^p - 1, and we need k <= n - p
/// We find the smallest p such that 2^p >= k + p + 1
///
/// # Panics
///
/// Panics if the algorithm fails to converge for the given `k` value
/// (i.e., if `p` exceeds 32). This should never happen for realistic
/// lab workload values of `k`.
#[must_use]
pub fn compute_hamming_n_from_k(k: usize) -> (usize, usize) {
    let mut p = 1;
    loop {
        let n = (1_usize << p) - 1; // 2^p - 1
        if k <= n - p {
            return (n, p);
        }
        p += 1;
        // Safety check to avoid infinite loop
        assert!(
            p <= 32,
            "compute_hamming_n_from_k: failed to converge for k = {k}"
        );
    }
}

/// Encode message using Hamming code
/// Places parity bits at positions that are powers of 2 (1, 2, 4, 8, ...)
/// Places information bits at other positions (3, 5, 6, 7, 9, 10, 11, ...)
#[must_use]
pub fn encode_hamming(message: &[bool], code: &HammingCode) -> Vec<bool> {
    let mut codeword = vec![false; code.n];
    let mut msg_idx = 0;

    // Place information bits at non-power-of-2 positions
    for i in 1..=code.n {
        if !is_power_of_two(i) && msg_idx < message.len() {
            codeword[i - 1] = message[msg_idx];
            msg_idx += 1;
        }
    }

    // Calculate parity bits
    for i in 0..code.p {
        let parity_pos = 1 << i; // Position 1, 2, 4, 8, ...
        codeword[parity_pos - 1] = compute_parity_check(&codeword, parity_pos, code.n);
    }

    codeword
}

/// Add parity bit for double error detection
/// This is the overall parity bit (XOR of all bits)
#[must_use]
pub fn add_parity_bit(codeword: &[bool]) -> Vec<bool> {
    let mut extended = codeword.to_vec();
    let parity = codeword.iter().fold(false, |acc, &bit| acc ^ bit);
    extended.push(parity);
    extended
}

/// Compute parity check for a given parity position
/// XORs all bits whose position has 1 in the i-th bit of binary representation
#[must_use]
fn compute_parity_check(codeword: &[bool], parity_pos: usize, n: usize) -> bool {
    (1..=n)
        .filter(|&j| (j & parity_pos) != 0)
        .fold(false, |acc, j| acc ^ codeword[j - 1])
}

/// Compute syndrome for Hamming code
/// Returns syndrome vector (p bits) and overall parity check result
#[must_use]
pub fn compute_syndrome_hamming(codeword: &[bool], code: &HammingCode) -> (Vec<bool>, bool) {
    let syndrome: Vec<bool> = (0..code.p)
        .map(|i| {
            let parity_pos = 1 << i; // Position 1, 2, 4, 8, ...
            compute_parity_check(codeword, parity_pos, code.n)
        })
        .collect();

    // Compute overall parity (for double error detection)
    let overall_parity = codeword.iter().fold(false, |acc, &bit| acc ^ bit);

    (syndrome, overall_parity)
}

/// Convert syndrome to error position (read right to left, 1-indexed)
#[must_use]
fn syndrome_to_position(syndrome: &[bool]) -> usize {
    syndrome
        .iter()
        .enumerate()
        .fold(0, |acc, (i, &bit)| if bit { acc | (1 << i) } else { acc })
}

/// Correct single error at given position
#[must_use]
fn correct_single_error(codeword: &[bool], position: usize) -> Vec<bool> {
    let mut corrected = codeword.to_vec();
    corrected[position - 1] = !corrected[position - 1];
    corrected
}

/// Decode Hamming code and detect/correct errors
/// Returns corrected codeword and error information
#[must_use]
pub fn decode_hamming(
    received: &[bool],
    code: &HammingCode,
    has_parity_bit: bool,
) -> (Vec<bool>, HammingErrorInfo) {
    let (syndrome, overall_parity) = if has_parity_bit {
        // Extract codeword without parity bit
        let codeword = &received[..code.n];
        let received_parity = received[code.n];
        let (syndrome, computed_parity) = compute_syndrome_hamming(codeword, code);
        // Overall parity check: received_parity should equal computed_parity
        let overall_parity = received_parity == computed_parity;
        (syndrome, overall_parity)
    } else {
        compute_syndrome_hamming(received, code)
    };

    // Check if all syndrome bits are zero
    let all_zero = syndrome.is_empty() || syndrome.iter().all(|&b| !b);

    // Extract codeword part (without parity bit if present)
    let codeword_part = if has_parity_bit {
        &received[..code.n]
    } else {
        received
    };

    // Handle no errors case
    if all_zero && (!has_parity_bit || overall_parity) {
        return (codeword_part.to_vec(), HammingErrorInfo::NoError);
    }

    // Handle double error detection (only for modified Hamming code)
    if has_parity_bit && !all_zero && overall_parity {
        return (codeword_part.to_vec(), HammingErrorInfo::DoubleError);
    }

    // Try to correct single error
    if !all_zero {
        let position = syndrome_to_position(&syndrome);
        if position > 0 && position <= code.n {
            return (
                correct_single_error(codeword_part, position),
                HammingErrorInfo::SingleError(position),
            );
        }
    }

    // Uncorrectable error
    (codeword_part.to_vec(), HammingErrorInfo::DoubleError)
}

/// Inject errors with given multiplicity (0, 1, or 2)
/// Returns modified codeword and list of error positions (1-indexed)
#[must_use]
pub fn inject_errors(codeword: &[bool], error_multiplicity: usize) -> (Vec<bool>, Vec<usize>) {
    let mut modified = codeword.to_vec();
    let mut error_positions = Vec::new();
    let mut rng = rand::rng();

    if error_multiplicity == 0 {
        return (modified, error_positions);
    }

    // Generate unique random positions using Fisher-Yates shuffle approach
    let max_errors = error_multiplicity.min(codeword.len());
    let mut available_positions: Vec<usize> = (0..codeword.len()).collect();

    for _ in 0..max_errors {
        let idx = rng.random_range(0..available_positions.len());
        let pos = available_positions.swap_remove(idx);
        error_positions.push(pos + 1); // 1-indexed
        modified[pos] = !modified[pos];
    }

    (modified, error_positions)
}

/// Generate random error multiplicity (0, 1, or 2)
#[must_use]
pub fn generate_error_multiplicity() -> usize {
    let mut rng = rand::rng();
    rng.random_range(0..3) // 0, 1, or 2
}

/// Check if a number is a power of two
#[must_use]
const fn is_power_of_two(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hamming_n_from_k() {
        // For k=9, should get n=15, p=4 (2^4 - 1 = 15, and 9 <= 15 - 4 = 11)
        let (n, p) = compute_hamming_n_from_k(9);
        assert_eq!(n, 15);
        assert_eq!(p, 4);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(8));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(5));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(7));
    }

    #[test]
    fn test_encode_hamming() {
        let code = HammingCode { k: 9, n: 15, p: 4 };
        let message = vec![true, false, false, true, true, false, false, true, true];
        let codeword = encode_hamming(&message, &code);

        assert_eq!(codeword.len(), 15);
        // Check that information bits are at correct positions
        // Positions 3, 5, 6, 7, 9, 10, 11, 12, 13 should contain message bits
    }

    #[test]
    fn test_decode_hamming_no_error() {
        let code = HammingCode { k: 9, n: 15, p: 4 };
        let message = vec![true, false, false, true, true, false, false, true, true];
        let codeword = encode_hamming(&message, &code);
        let (corrected, error_info) = decode_hamming(&codeword, &code, false);

        assert_eq!(error_info, HammingErrorInfo::NoError);
        assert_eq!(corrected, codeword);
    }

    #[test]
    fn test_decode_hamming_single_error() {
        let code = HammingCode { k: 9, n: 15, p: 4 };
        let message = vec![true, false, false, true, true, false, false, true, true];
        let codeword = encode_hamming(&message, &code);

        // Inject error at position 5
        let mut received = codeword.clone();
        received[4] = !received[4]; // Position 5 (1-indexed) = index 4 (0-indexed)

        let (corrected, error_info) = decode_hamming(&received, &code, false);

        match error_info {
            HammingErrorInfo::SingleError(pos) => {
                assert_eq!(pos, 5);
                assert_eq!(corrected, codeword);
            }
            _ => panic!("Expected single error"),
        }
    }

    #[test]
    fn test_add_parity_bit() {
        let codeword = vec![true, false, true, false];
        let extended = add_parity_bit(&codeword);
        assert_eq!(extended.len(), codeword.len() + 1);
        // Parity should be XOR of all bits: true ^ false ^ true ^ false = false
        assert!(!extended[4]);
    }

    #[test]
    fn test_decode_hamming_with_parity_double_error() {
        let code = HammingCode { k: 9, n: 15, p: 4 };
        let message = vec![true, false, false, true, true, false, false, true, true];
        let codeword = encode_hamming(&message, &code);
        let extended = add_parity_bit(&codeword);

        // Inject two errors
        let mut received = extended;
        received[4] = !received[4]; // Position 5
        received[6] = !received[6]; // Position 7

        let (_corrected, error_info) = decode_hamming(&received, &code, true);

        assert_eq!(error_info, HammingErrorInfo::DoubleError);
    }
}
