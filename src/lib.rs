// Finished 1st lab, modified for 2nd lab with noise
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
use rand::Rng;

pub mod coding;

/// Очистка от -0.0 значений
fn clean_zeros(value: f64) -> f64 {
    if value.abs() < 1e-10 { 0.0 } else { value }
}

/// Генерация нормированных вероятностей длиной `count`
#[must_use]
pub fn generate_probabilities(count: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let nums: Vec<f64> = (0..count).map(|_| rng.random_range(1.0..100.0)).collect();
    let sum: f64 = nums.iter().sum();
    nums.iter()
        .map(|v| clean_zeros((v / sum * 100_000.0).round() / 100_000.0))
        .collect()
}

/// Генерация нормированных вероятностей с минимальным порогом
/// Использует `generate_probabilities` и добавляет минимальный порог к каждому элементу
#[must_use]
pub fn generate_probabilities_with_threshold(count: usize, min_threshold: f64) -> Vec<f64> {
    // Генерируем базовые вероятности
    let base_probs = generate_probabilities(count);

    // Добавляем минимальный порог к каждому элементу
    let adjusted_probs: Vec<f64> = base_probs
        .iter()
        .map(|&prob| prob.mul_add(1.0 - min_threshold, min_threshold))
        .collect();

    // Нормализуем результат
    let sum: f64 = adjusted_probs.iter().sum();
    adjusted_probs
        .iter()
        .map(|&v| clean_zeros((v / sum * 100_000.0).round() / 100_000.0))
        .collect()
}

/// Расчёт энтропии
#[must_use]
pub fn calc_entropy(probs: &[f64]) -> f64 {
    -probs.iter().map(|&p| p * p.log2()).sum::<f64>()
}

/// Теоретический максимум энтропии для `count` сигналов
#[must_use]
pub fn max_entropy(count: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        (count as f64).log2()
    }
}

/// Генерация матрицы переходов `p(x_i/y_j)` с учетом помех<br>
/// Диагональные элементы >= `min_threshold`,
/// остальные элементы используют `generate_probabilities` для равномерного распределения
#[must_use]
pub fn generate_transition_matrix(count: usize, min_threshold: f64) -> Vec<Vec<f64>> {
    let mut rng = rand::rng();
    let mut matrix = vec![vec![0.0; count]; count];

    for (i, row) in matrix.iter_mut().enumerate().take(count) {
        // 1. Диагональный элемент: min_threshold <= p(x_i/y_i) <= 1.0
        let diagonal_prob =
            (1.0 - min_threshold).mul_add(rng.random_range(0.0..1.0), min_threshold);
        row[i] = clean_zeros((diagonal_prob * 100_000.0_f64).round() / 100_000.0);

        // 2. Остальные элементы: используем generate_probabilities для равномерного распределения
        let remaining_prob = 1.0 - row[i];
        let non_diagonal_count = count - 1;

        if non_diagonal_count > 0 {
            // Генерируем нормализованные веса для недиагональных элементов
            let normalized_weights = generate_probabilities(non_diagonal_count);

            // Распределяем оставшуюся вероятность пропорционально весам
            let mut col_idx = 0;
            for (j, val) in row.iter_mut().enumerate() {
                if j != i {
                    *val = clean_zeros(
                        (remaining_prob * normalized_weights[col_idx] * 100_000.0_f64).round()
                            / 100_000.0,
                    );
                    col_idx += 1;
                }
            }
        }
    }

    matrix
}

/// Расчет вероятностей на выходе `p(y_j)` = `sum(p(x_i)` * `p(x_i/y_j)`)
#[allow(clippy::needless_range_loop)]
#[must_use]
pub fn calculate_output_probabilities(
    input_probs: &[f64],
    transition_matrix: &[Vec<f64>],
) -> Vec<f64> {
    let count = input_probs.len();
    let mut output_probs = vec![0.0; count];

    for j in 0..count {
        for i in 0..count {
            output_probs[j] += input_probs[i] * transition_matrix[i][j];
        }
        output_probs[j] = clean_zeros((output_probs[j] * 100_000.0_f64).round() / 100_000.0);
    }

    output_probs
}

/// Расчет матрицы совместных вероятностей `p(x_i,y_j)` = `p(x_i)` * `p(y_j|x_i)`
#[allow(clippy::needless_range_loop)]
#[must_use]
pub fn calculate_joint_probabilities(
    input_probs: &[f64],
    output_probs: &[f64],
    transition_matrix: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let count = input_probs.len();
    let mut joint_matrix = vec![vec![0.0; count]; count];

    for i in 0..count {
        for j in 0..count {
            // p(x_i, y_j) = p(x_i) * p(y_j|x_i) = p(x_i) * p(x_i|y_j) * p(y_j) / p(x_i)
            // Упрощаем: p(x_i, y_j) = p(x_i|y_j) * p(y_j)
            joint_matrix[i][j] = transition_matrix[i][j] * output_probs[j];
            joint_matrix[i][j] =
                clean_zeros((joint_matrix[i][j] * 100_000.0_f64).round() / 100_000.0);
        }
    }

    joint_matrix
}

/// Расчет условной энтропии H(X/Y)
#[must_use]
pub fn calculate_conditional_entropy(
    joint_probs: &[Vec<f64>],
    transition_matrix: &[Vec<f64>],
) -> f64 {
    let mut entropy = 0.0;

    for i in 0..joint_probs.len() {
        for j in 0..joint_probs[i].len() {
            if joint_probs[i][j] > 0.0 && transition_matrix[i][j] > 0.0 {
                entropy -= joint_probs[i][j] * transition_matrix[i][j].log2();
            }
        }
    }

    clean_zeros((entropy * 100_000.0_f64).round() / 100_000.0)
}

/// Расчет количества информации I(X,Y) = H(X) - H(X/Y)
#[must_use]
pub fn calculate_mutual_information(input_entropy: f64, conditional_entropy: f64) -> f64 {
    clean_zeros(((input_entropy - conditional_entropy) * 100_000.0_f64).round() / 100_000.0)
}

// ========== Функции для 3-й лабораторной работы ==========

/// Генерация длительностей символов в диапазоне (0, N] мкс
#[must_use]
pub fn generate_symbol_durations(count: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    #[allow(clippy::cast_precision_loss)]
    let count_f64 = count as f64;
    (0..count)
        .map(|_| {
            let duration = rng.random_range(0.0..=count_f64);
            clean_zeros((duration * 100_000.0).round() / 100_000.0)
        })
        .collect()
}

/// Генерация матрицы вероятностей ошибок P[X,Y] для 3-й лабораторной
/// Элементы в диапазоне (0, q], где q = 1/(2*N)
/// Диагональный элемент = 1 - сумма остальных элементов в строке
#[must_use]
pub fn generate_error_probability_matrix(count: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::rng();
    #[allow(clippy::cast_precision_loss)]
    let q = 1.0 / (2.0 * count as f64);
    let mut matrix = vec![vec![0.0; count]; count];

    for (i, row) in matrix.iter_mut().enumerate().take(count) {
        // Генерируем вероятности ошибок для недиагональных элементов
        let mut error_sum = 0.0;
        for (j, val) in row.iter_mut().enumerate() {
            if j != i {
                let error_prob = rng.random_range(0.0..=q);
                *val = clean_zeros((error_prob * 100_000.0_f64).round() / 100_000.0);
                error_sum += *val;
            }
        }

        // Диагональный элемент = 1 - сумма ошибок
        row[i] = clean_zeros(((1.0 - error_sum) * 100_000.0_f64).round() / 100_000.0);
    }

    matrix
}

/// Расчет средней длительности символа τ = Σ(Px[i] * Tx[i])
#[must_use]
pub fn calculate_average_duration(input_probs: &[f64], durations: &[f64]) -> f64 {
    let tau: f64 = input_probs
        .iter()
        .zip(durations.iter())
        .map(|(&p, &t)| p * t)
        .sum();
    clean_zeros((tau * 100_000.0_f64).round() / 100_000.0)
}

/// Расчет скорости передачи информации без помех I(Y) = H(X) / τ
/// `avg_duration` в мкс, результат в бит/с (умножаем на `1_000_000` для перевода мкс → с)
#[must_use]
pub fn calculate_information_rate_no_noise(entropy: f64, avg_duration: f64) -> f64 {
    if avg_duration > 0.0 {
        // 1 мкс = 10^-6 с, поэтому умножаем на 1_000_000
        clean_zeros((entropy / avg_duration * 1_000_000.0 * 100_000.0_f64).round() / 100_000.0)
    } else {
        0.0
    }
}

/// Расчет пропускной способности без помех C = log₂(N) / τ
/// `avg_duration` в мкс, результат в бит/с (умножаем на `1_000_000` для перевода мкс → с)
#[must_use]
pub fn calculate_capacity_no_noise(count: usize, avg_duration: f64) -> f64 {
    if avg_duration > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let max_ent = (count as f64).log2();
        // 1 мкс = 10^-6 с, поэтому умножаем на 1_000_000
        clean_zeros((max_ent / avg_duration * 1_000_000.0 * 100_000.0_f64).round() / 100_000.0)
    } else {
        0.0
    }
}

/// Расчет скорости передачи информации с помехами I(Y,Z) = (H(X) - H(X/Y)) / τ
/// `avg_duration` в мкс, результат в бит/с (умножаем на `1_000_000` для перевода мкс → с)
#[must_use]
pub fn calculate_information_rate_with_noise(
    entropy: f64,
    conditional_entropy: f64,
    avg_duration: f64,
) -> f64 {
    if avg_duration > 0.0 {
        let mutual_info = entropy - conditional_entropy;
        // 1 мкс = 10^-6 с, поэтому умножаем на 1_000_000
        clean_zeros((mutual_info / avg_duration * 1_000_000.0 * 100_000.0_f64).round() / 100_000.0)
    } else {
        0.0
    }
}

/// Расчет пропускной способности с помехами C = (log₂(N) - H(X/Y)) / τ
/// `avg_duration` в мкс, результат в бит/с (умножаем на `1_000_000` для перевода мкс → с)
#[must_use]
pub fn calculate_capacity_with_noise(
    count: usize,
    conditional_entropy: f64,
    avg_duration: f64,
) -> f64 {
    if avg_duration > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let max_ent = (count as f64).log2();
        // 1 мкс = 10^-6 с, поэтому умножаем на 1_000_000
        let capacity = (max_ent - conditional_entropy) / avg_duration * 1_000_000.0;
        clean_zeros((capacity * 100_000.0_f64).round() / 100_000.0)
    } else {
        0.0
    }
}

/// Форматирование скорости/пропускной способности с правильными единицами измерения
#[must_use]
pub fn format_rate(rate_bits_per_sec: f64) -> String {
    if rate_bits_per_sec >= 1_000_000.0 {
        let mbits = rate_bits_per_sec / 1_000_000.0;
        format!("{mbits:.3} Мбит/с")
    } else if rate_bits_per_sec >= 1_000.0 {
        let kbits = rate_bits_per_sec / 1_000.0;
        format!("{kbits:.3} кбит/с")
    } else {
        format!("{rate_bits_per_sec:.3} бит/с")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_sum_to_one() {
        let probs = generate_probabilities(8);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn entropy_is_positive() {
        let probs = generate_probabilities(10);
        let h = calc_entropy(&probs);
        assert!(h > 0.0 && h.is_finite(), "entropy = {h}");
    }

    #[test]
    fn max_entropy_matches_theory() {
        let n = 4;
        #[allow(clippy::cast_precision_loss)]
        let theoretical = (n as f64).log2();
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(max_entropy(n), theoretical);
        }
    }

    #[test]
    fn transition_matrix_diagonal_elements_valid() {
        let matrix = generate_transition_matrix(5, 0.7);
        for (i, row) in matrix.iter().enumerate().take(5) {
            assert!(row[i] >= 0.7, "diagonal element {} < 0.7", row[i]);
            assert!(row[i] <= 1.0, "diagonal element {} > 1.0", row[i]);
        }
    }

    #[test]
    fn transition_matrix_rows_sum_to_one() {
        let matrix = generate_transition_matrix(4, 0.7);
        for (i, row) in matrix.iter().enumerate().take(4) {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} sum = {sum}");
        }
    }

    #[test]
    fn output_probabilities_sum_to_one() {
        let input_probs = generate_probabilities(3);
        let transition_matrix = generate_transition_matrix(3, 0.7);
        let output_probs = calculate_output_probabilities(&input_probs, &transition_matrix);
        let sum: f64 = output_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "output probabilities sum = {sum}");
    }

    #[test]
    fn mutual_information_is_positive() {
        let input_probs = generate_probabilities(4);
        let transition_matrix = generate_transition_matrix(4, 0.7);
        let output_probs = calculate_output_probabilities(&input_probs, &transition_matrix);
        let joint_probs =
            calculate_joint_probabilities(&input_probs, &output_probs, &transition_matrix);

        let input_entropy = calc_entropy(&input_probs);
        let conditional_entropy = calculate_conditional_entropy(&joint_probs, &transition_matrix);
        let mutual_info = calculate_mutual_information(input_entropy, conditional_entropy);

        assert!(mutual_info >= 0.0, "mutual information = {mutual_info}");
    }

    #[test]
    fn symbol_durations_are_positive() {
        let durations = generate_symbol_durations(5);
        for d in durations {
            assert!(d > 0.0 && d <= 5.0, "duration = {d}");
        }
    }

    #[test]
    fn error_probability_matrix_rows_sum_to_one() {
        let matrix = generate_error_probability_matrix(4);
        for (i, row) in matrix.iter().enumerate().take(4) {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} sum = {sum}");
        }
    }

    #[test]
    fn average_duration_is_positive() {
        let probs = generate_probabilities(3);
        let durations = generate_symbol_durations(3);
        let avg = calculate_average_duration(&probs, &durations);
        assert!(avg > 0.0, "average duration = {avg}");
    }

    #[test]
    fn information_rate_no_noise_is_positive() {
        let entropy = 2.5;
        let avg_duration = 1.0;
        let rate = calculate_information_rate_no_noise(entropy, avg_duration);
        assert!(rate > 0.0, "information rate = {rate}");
    }

    #[test]
    fn capacity_no_noise_is_positive() {
        let count = 4;
        let avg_duration = 1.0;
        let capacity = calculate_capacity_no_noise(count, avg_duration);
        assert!(capacity > 0.0, "capacity = {capacity}");
    }
}
