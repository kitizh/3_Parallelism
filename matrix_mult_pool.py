import numpy as np
from multiprocessing import Pool
import os


def read_matrix(file_path):
    """Считывает матрицу из файла."""
    return np.loadtxt(file_path, dtype=int)


def write_matrix(file_path, matrix):
    """Записывает матрицу в файл."""
    np.savetxt(file_path, matrix, fmt='%d')


def elementwise_multiply(task):
    """Вычисляет один элемент матрицы-произведения."""
    i, j, a, b = task
    return i, j, a * b


def parallel_matrix_multiplication(matrix_a, matrix_b, output_file, num_processes=4):
    """Перемножает две матрицы поэлементно с использованием пула процессов."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Матрицы должны быть одного размера.")

    tasks = [
        (i, j, matrix_a[i, j], matrix_b[i, j])
        for i in range(matrix_a.shape[0])
        for j in range(matrix_a.shape[1])
    ]

    result_matrix = np.zeros_like(matrix_a)

    with Pool(processes=num_processes) as pool:
        for i, j, value in pool.map(elementwise_multiply, tasks):
            result_matrix[i, j] = value

    write_matrix(output_file, result_matrix)


if __name__ == "__main__":
    # Пример использования
    matrix_a = read_matrix("matrix_a.txt")
    matrix_b = read_matrix("matrix_b.txt")
    parallel_matrix_multiplication(matrix_a, matrix_b, "result_matrix.txt", num_processes=4)
