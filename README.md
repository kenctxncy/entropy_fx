# entropy_fx

Библиотека для расчета энтропии источников информации, анализа каналов связи и реализации помехоустойчивых кодов.

## Описание

`entropy_fx` предоставляет набор функций для:

- Расчет энтропии источников информации
- Генерация и анализ матриц переходов каналов связи
- Расчет пропускной способности каналов (с помехами и без)
- Реализация систематических помехоустойчивых кодов
- Реализация кодов Хэмминга с обнаружением и исправлением ошибок

## Использование

Добавьте в `Cargo.toml`:

```toml
[dependencies]
entropy_fx = { path = "../entropy_fx" }
# или из git репозитория
# entropy_fx = { git = "https://github.com/yourusername/entropy_fx.git" }
```

## Основные модули

### Основные функции энтропии

```rust
use entropy_fx::{calc_entropy, max_entropy, generate_probabilities};

// Генерация вероятностей
let probs = generate_probabilities(5);

// Расчет энтропии
let entropy = calc_entropy(&probs);

// Максимальная энтропия
let max = max_entropy(5);
```

### Систематические коды

```rust
use entropy_fx::coding::systematic::{
    SystematicCode, compute_n_from_k, build_generator_matrix,
    encode_message, compute_syndrome, correct_error
};

// Вычисление параметров кода
let (n, p) = compute_n_from_k(9);

// Построение кода
let code = build_generator_matrix(9, n);

// Кодирование сообщения
let message = vec![true, false, true, /* ... */];
let codeword = encode_message(&message, &code);

// Декодирование и коррекция ошибок
let syndrome = compute_syndrome(&code.parity_check, &received);
let (corrected, error_info) = correct_error(&code.parity_check, &received);
```

### Коды Хэмминга

```rust
use entropy_fx::coding::hamming::{
    HammingCode, compute_hamming_n_from_k,
    encode_hamming, decode_hamming, add_parity_bit
};

// Вычисление параметров кода Хэмминга
let (n, p) = compute_hamming_n_from_k(9);
let code = HammingCode { k: 9, n, p };

// Кодирование
let message = vec![true, false, true, /* ... */];
let codeword = encode_hamming(&message, &code);

// Добавление parity bit для обнаружения двукратных ошибок
let codeword_with_parity = add_parity_bit(&codeword);

// Декодирование и коррекция
let (corrected, error_info) = decode_hamming(&received, &code, true);
```

## API Документация

### Функции энтропии

- `generate_probabilities(count: usize) -> Vec<f64>` - генерация нормированных вероятностей
- `calc_entropy(probs: &[f64]) -> f64` - расчет энтропии
- `max_entropy(count: usize) -> f64` - максимальная энтропия
- `calculate_conditional_entropy(...)` - условная энтропия
- `calculate_mutual_information(...)` - взаимная информация

### Функции каналов

- `generate_transition_matrix(...)` - генерация матрицы переходов
- `calculate_capacity_no_noise(...)` - пропускная способность без помех
- `calculate_capacity_with_noise(...)` - пропускная способность с помехами

### Систематические коды

См. модуль `entropy_fx::coding::systematic`

### Коды Хэмминга

См. модуль `entropy_fx::coding::hamming`

## Тестирование

```bash
# Запустить все тесты
cargo test

# Тесты конкретного модуля
cargo test coding::hamming
cargo test coding::systematic
```

## Требования

- Rust 1.91+ (edition 2024)
- `rand = "0.9.2"`

## Проверка кода

```bash
# Clippy с pedantic и nursery флагами
cargo clippy --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery

# Форматирование
cargo fmt
```

## Принципы разработки

Код следует принципам:
- **LIPS**

## Лицензия

Genshtab Public License 1.4

## Автор

Аланов А. + Оля
