use crate::MAX_CODE_LENGTH;

/// Compute `n` and `p` from `k` using inequality `2^k <= 2^n / (1+n)`.
///
/// This function is used by both systematic and cyclic codes.
///
/// # Arguments
///
/// * `k` - Message length (number of information bits)
///
/// # Returns
///
/// A tuple `(n, p)` where:
/// - `n` is the codeword length
/// - `p` is the number of parity bits
///
/// # Panics
///
/// Panics if the algorithm fails to converge (n > `MAX_CODE_LENGTH`)
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn compute_n_from_k(k: usize) -> (usize, usize) {
    let mut n = k;
    let k_u32 = k as u32;

    loop {
        let n_u32 = n as u32;
        let lhs = 2_u128.pow(k_u32);
        let rhs = 2_u128.pow(n_u32) / (n as u128 + 1);

        if lhs <= rhs {
            break;
        }

        n += 1;
        assert!(
            n <= MAX_CODE_LENGTH,
            "compute_n_from_k: failed to converge for k = {k}"
        );
    }

    let p = n - k;
    (n, p)
}
