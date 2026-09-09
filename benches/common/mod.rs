#![allow(dead_code)]

pub fn square_matrix_data(size: usize, seed: usize) -> Vec<f64> {
    rectangular_matrix_data(size, size, seed)
}

pub fn rectangular_matrix_data(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    let scale = rows.max(cols) as f64 * 0.75 + 1.0;
    let mut data = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let harmonic = (row as f64 + 1.0) * 0.013 + (col as f64 + 1.0) * 0.021;
            let cycle = (((row * 17 + col * 31 + seed) % 23) as f64 - 11.0) / 29.0;
            let diagonal_boost = if row == col { scale } else { 0.0 };
            data[row * cols + col] = diagonal_boost + harmonic + cycle;
        }
    }
    data
}

pub fn vector_data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let cycle = (((index * 19 + seed) % 17) as f64 - 8.0) / 11.0;
            cycle + (index as f64 + 1.0) * 0.017
        })
        .collect()
}

pub fn spd_matrix_data(size: usize) -> Vec<f64> {
    let base = square_matrix_data(size, 41);
    let mut spd = vec![0.0; size * size];
    for row in 0..size {
        for col in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                sum += base[row * size + k] * base[col * size + k];
            }
            if row == col {
                sum += size as f64;
            }
            spd[row * size + col] = sum;
        }
    }
    spd
}
