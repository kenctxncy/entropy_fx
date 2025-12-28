#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
use rand::Rng;

/// Binary polynomial represented as a vector of coefficients
/// coefficients[i] is the coefficient for x^i
/// For example: [1, 0, 1, 1] represents x^3 + x + 1
pub type Polynomial = Vec<bool>;

/// Cyclic code structure
#[derive(Debug, Clone)]
pub struct CyclicCode {
    /// Message length (number of information bits)
    pub k: usize,
    /// Codeword length (total number of bits)
    pub n: usize,
    /// Number of parity bits
    pub p: usize,
    /// Generator polynomial P(x) of degree p
    pub generator_poly: Polynomial,
    /// Syndrome table: maps syndrome (as polynomial) to error position
    pub syndrome_table: Vec<(Polynomial, usize)>,
}

/// Information about error correction result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CyclicErrorInfo {
    /// No error detected
    NoError,
    /// Single error at given position (1-indexed)
    SingleError(usize),
    /// Multiple errors or uncorrectable error pattern
    Uncorrectable,
}

/// Primitive polynomials for degrees 1-8
/// Each entry is a tuple representing coefficients from highest to lowest degree
/// For example: (1, 0, 1, 1) represents x^3 + x + 1
const PRIMITIVE_POLYNOMIALS: &[&[u8]] = &[
    &[1, 1],                      // x + 1
    &[1, 1, 1],                   // x^2 + x + 1
    &[1, 0, 1, 1],                // x^3 + x + 1
    &[1, 0, 0, 1, 1],             // x^4 + x + 1
    &[1, 0, 1, 0, 0, 1],          // x^5 + x^2 + 1
    &[1, 0, 0, 0, 0, 1, 1],       // x^6 + x + 1
    &[1, 0, 0, 0, 1, 1, 0, 1],    // x^7 + x^4 + x^3 + x + 1
    &[1, 0, 0, 0, 1, 0, 0, 1, 1], // x^8 + x^4 + x^3 + x + 1
];

use super::common::compute_n_from_k;

/// Convert polynomial coefficients from u8 array to bool vector
#[must_use]
fn poly_from_coeffs(coeffs: &[u8]) -> Polynomial {
    coeffs.iter().map(|&c| c != 0).collect()
}

/// Normalize polynomial by removing leading zeros
#[must_use]
fn normalize_poly(mut poly: Polynomial) -> Polynomial {
    while poly.last() == Some(&false) {
        poly.pop();
    }
    if poly.is_empty() { vec![false] } else { poly }
}

/// Multiply polynomial by x^p (shift left by p positions)
#[must_use]
fn multiply_by_xp(poly: &Polynomial, p: usize) -> Polynomial {
    let mut result = vec![false; p];
    result.extend_from_slice(poly);
    normalize_poly(result)
}

/// Add two polynomials in GF(2) (XOR)
#[must_use]
fn add_polynomials(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let max_len = a.len().max(b.len());
    let mut result = vec![false; max_len];
    for (i, &coeff) in a.iter().enumerate() {
        result[i] ^= coeff;
    }
    for (i, &coeff) in b.iter().enumerate() {
        result[i] ^= coeff;
    }
    normalize_poly(result)
}

/// Divide polynomial `dividend` by `divisor` in GF(2), return remainder
#[must_use]
fn polynomial_modulo(dividend: &Polynomial, divisor: &Polynomial) -> Polynomial {
    let mut remainder = dividend.clone();
    let divisor_degree = divisor.len() - 1;

    while remainder.len() > divisor.len() {
        let remainder_degree = remainder.len() - 1;
        if !remainder[remainder_degree] {
            remainder.pop();
            continue;
        }

        // Perform polynomial division step
        let shift = remainder_degree - divisor_degree;
        let mut temp = vec![false; shift];
        temp.extend_from_slice(divisor);

        // Extend temp to match remainder length
        while temp.len() < remainder.len() {
            temp.push(false);
        }

        // XOR (subtract in GF(2))
        for (i, &coeff) in temp.iter().enumerate() {
            if i < remainder.len() {
                remainder[i] ^= coeff;
            }
        }

        remainder = normalize_poly(remainder);
    }

    // Final reduction if needed
    if remainder.len() == divisor.len() {
        let mut can_reduce = true;
        for (i, &div_coeff) in divisor.iter().enumerate() {
            if remainder[i] && !div_coeff {
                can_reduce = false;
                break;
            }
        }
        if can_reduce {
            let mut reduced = vec![false; divisor.len()];
            for (i, &div_coeff) in divisor.iter().enumerate() {
                reduced[i] = remainder[i] ^ div_coeff;
            }
            remainder = normalize_poly(reduced);
        }
    }

    normalize_poly(remainder)
}

/// Convert bit vector to polynomial (bits from left to right = high to low degree)
///
/// # Arguments
///
/// * `bits` - Bit vector to convert
///
/// # Returns
///
/// Polynomial representation of the bit vector
#[must_use]
pub fn bits_to_polynomial(bits: &[bool]) -> Polynomial {
    Vec::from(bits)
}

/// Convert polynomial to bit vector
///
/// # Arguments
///
/// * `poly` - Polynomial to convert
/// * `length` - Desired length of the bit vector
///
/// # Returns
///
/// Bit vector representation of the polynomial
#[must_use]
pub fn polynomial_to_bits(poly: &Polynomial, length: usize) -> Vec<bool> {
    let mut bits = vec![false; length];
    for (i, &coeff) in poly.iter().enumerate().take(length) {
        bits[i] = coeff;
    }
    bits
}

/// Create a new cyclic code
///
/// # Arguments
///
/// * `k` - Message length (number of information bits)
///
/// # Returns
///
/// A `CyclicCode` structure with generator polynomial and syndrome table
///
/// # Panics
///
/// Panics if `p` (computed from `k`) is not in range [1, 8] for primitive polynomials
#[must_use]
pub fn create_cyclic_code(k: usize) -> CyclicCode {
    let (n, p) = compute_n_from_k(k);

    // Get primitive polynomial for degree p
    assert!(
        (1..=8).contains(&p),
        "p = {p} is out of range [1, 8] for primitive polynomials"
    );
    let generator_coeffs = PRIMITIVE_POLYNOMIALS[p - 1];
    let generator_poly = poly_from_coeffs(generator_coeffs);

    // Build syndrome table
    let syndrome_table = build_syndrome_table(n, &generator_poly);

    CyclicCode {
        k,
        n,
        p,
        generator_poly,
        syndrome_table,
    }
}

/// Build syndrome table: for each error position, compute the syndrome
///
/// # Arguments
///
/// * `n` - Codeword length
/// * `generator_poly` - Generator polynomial
///
/// # Returns
///
/// Vector of tuples `(syndrome, error_position)` for each possible error position
#[must_use]
fn build_syndrome_table(n: usize, generator_poly: &Polynomial) -> Vec<(Polynomial, usize)> {
    let mut table = Vec::new();

    for pos in 0..n {
        // Create error vector with error at position pos
        let mut error_vec = vec![false; n];
        error_vec[pos] = true;

        // Compute syndrome: remainder of error polynomial divided by generator
        let error_poly = bits_to_polynomial(&error_vec);
        let syndrome = polynomial_modulo(&error_poly, generator_poly);

        table.push((syndrome, pos));
    }

    table
}

/// Encode message using cyclic code
///
/// Algorithm: F(x) = x^`p` * G(x) + R(x), where R(x) = (x^`p` * G(x)) mod P(x)
///
/// # Arguments
///
/// * `message` - Message bits to encode
/// * `code` - Cyclic code structure
///
/// # Returns
///
/// Encoded codeword
///
/// # Panics
///
/// Panics if `message.len()` does not match `code.k`
#[must_use]
pub fn encode_cyclic(message: &[bool], code: &CyclicCode) -> Vec<bool> {
    assert_eq!(
        message.len(),
        code.k,
        "Message length {} does not match code k = {}",
        message.len(),
        code.k
    );

    // G(x) - message polynomial
    let g_poly = bits_to_polynomial(message);

    // x^p * G(x)
    let xp_g_poly = multiply_by_xp(&g_poly, code.p);

    // R(x) = (x^p * G(x)) mod P(x)
    let r_poly = polynomial_modulo(&xp_g_poly, &code.generator_poly);

    // F(x) = x^p * G(x) + R(x)
    let f_poly = add_polynomials(&xp_g_poly, &r_poly);

    // Convert to bit vector of length n
    polynomial_to_bits(&f_poly, code.n)
}

/// Compute syndrome of received codeword
///
/// # Arguments
///
/// * `received` - Received codeword (possibly with errors)
/// * `code` - Cyclic code structure
///
/// # Returns
///
/// Syndrome polynomial
///
/// # Panics
///
/// Panics if `received.len()` does not match `code.n`
#[must_use]
pub fn compute_syndrome_cyclic(received: &[bool], code: &CyclicCode) -> Polynomial {
    assert_eq!(
        received.len(),
        code.n,
        "Received length {} does not match code n = {}",
        received.len(),
        code.n
    );

    let received_poly = bits_to_polynomial(received);
    polynomial_modulo(&received_poly, &code.generator_poly)
}

/// Decode and correct errors in received codeword
///
/// # Arguments
///
/// * `received` - Received codeword (possibly with errors)
/// * `code` - Cyclic code structure
///
/// # Returns
///
/// A tuple `(corrected, error_info)` where:
/// - `corrected` is the corrected codeword
/// - `error_info` contains information about detected/corrected errors
#[must_use]
pub fn decode_cyclic(received: &[bool], code: &CyclicCode) -> (Vec<bool>, CyclicErrorInfo) {
    let syndrome = compute_syndrome_cyclic(received, code);

    // Check if syndrome is zero (no error)
    let is_zero = syndrome.iter().all(|&b| !b);
    if is_zero {
        return (Vec::from(received), CyclicErrorInfo::NoError);
    }

    // Look up error position in syndrome table
    for (syndrome_entry, error_pos) in &code.syndrome_table {
        if syndrome.len() == syndrome_entry.len()
            && syndrome
                .iter()
                .zip(syndrome_entry.iter())
                .all(|(&a, &b)| a == b)
        {
            // Found error position, correct it
            let mut corrected = Vec::from(received);
            corrected[*error_pos] = !corrected[*error_pos];
            return (corrected, CyclicErrorInfo::SingleError(*error_pos + 1));
        }
    }

    // Syndrome not found in table - uncorrectable error
    (Vec::from(received), CyclicErrorInfo::Uncorrectable)
}

/// Inject single error at random position with given probability
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
pub fn inject_single_error_cyclic(
    codeword: &[bool],
    error_probability: f64,
) -> (Vec<bool>, Option<usize>) {
    let mut rng = rand::rng();
    let mut received = Vec::from(codeword);

    if rng.random_range(0.0..1.0) < error_probability {
        let error_pos = rng.random_range(0..codeword.len());
        received[error_pos] = !received[error_pos];
        return (received, Some(error_pos));
    }

    (received, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_n_from_k() {
        let (n, p) = compute_n_from_k(53);
        assert!(n >= 53);
        assert_eq!(p, n - 53);
    }

    #[test]
    fn test_polynomial_operations() {
        // Test: (x^3 + x + 1) mod (x^3 + x + 1) = 0
        let poly1 = vec![true, false, true, true]; // x^3 + x + 1
        let poly2 = vec![true, false, true, true]; // x^3 + x + 1
        let remainder = polynomial_modulo(&poly1, &poly2);
        assert!(remainder.iter().all(|&b| !b) || remainder == vec![false]);
    }

    #[test]
    fn test_encode_decode() {
        let code = create_cyclic_code(4);
        let message = vec![true, true, false, true]; // 1101

        let codeword = encode_cyclic(&message, &code);
        assert_eq!(codeword.len(), code.n);

        // Decode without error
        let (_decoded, error_info) = decode_cyclic(&codeword, &code);
        assert!(matches!(error_info, CyclicErrorInfo::NoError));
    }
}
