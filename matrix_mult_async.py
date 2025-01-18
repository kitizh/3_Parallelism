import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time
import random


def generate_random_matrix(size, min_value=1, max_value=10):
    """Генерирует случайную квадратную матрицу заданного размера."""
    return np.random.randint(min_value, max_value + 1, size=(size, size))


def write_matrix_append(file_path, matrix):
    """Добавляет матрицу в файл с пропуском одной строки."""
    with open(file_path, "a") as f:
        f.write("\n")
        np.savetxt(f, matrix, fmt='%d')


def elementwise_multiply_and_write(task):
    """Вычисляет элемент матрицы-произведения и записывает в промежуточный файл."""
    i, j, a, b, intermediate_file = task
    value = a * b
    with open(intermediate_file, "a") as f:
        f.write(f"{i},{j},{value}\n")
    return i, j, value


def build_result_matrix(intermediate_file, shape):
    """Формирует итоговую матрицу из промежуточного файла."""
    result_matrix = np.zeros(shape, dtype=int)
    with open(intermediate_file, "r") as f:
        for line in f:
            i, j, value = map(int, line.strip().split(","))
            result_matrix[i, j] = value
    return result_matrix


def parallel_matrix_multiplication_auto_threads(matrix_a, matrix_b, intermediate_file):
    """Перемножает матрицы с автоматическим определением количества потоков."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Матрицы должны быть одного размера.")

    # Печать матриц в консоль
    print("Матрица A:")
    print(matrix_a)
    print("\nМатрица B:")
    print(matrix_b)

    # Формирование задач для параллельного вычисления
    tasks = [
        (i, j, matrix_a[i, j], matrix_b[i, j], intermediate_file)
        for i in range(matrix_a.shape[0])
        for j in range(matrix_a.shape[1])
    ]

    # Очистка промежуточного файла перед началом работы
    open(intermediate_file, "w").close()

    # Определение оптимального количества потоков
    num_processes = cpu_count()

    # Параллельное вычисление элементов матрицы
    with Pool(processes=num_processes) as pool:
        pool.map(elementwise_multiply_and_write, tasks)

    # Построение итоговой матрицы из промежуточного файла
    return build_result_matrix(intermediate_file, matrix_a.shape)


def generator_and_multiplier(result_file, intermediate_file, size=3, iterations=5, delay=2):
    """
    Генерирует случайные матрицы и перемножает их по мере генерации.

    :param result_file: файл для записи результирующих матриц
    :param intermediate_file: промежуточный файл для вычислений
    :param size: размер квадратных матриц
    :param iterations: количество пар матриц для генерации и умножения
    :param delay: задержка между генерацией новых пар матриц (в секундах)
    """
    for _ in range(iterations):
        # Генерация двух случайных матриц
        matrix_a = generate_random_matrix(size)
        matrix_b = generate_random_matrix(size)

        # Вычисление произведения матриц
        result_matrix = parallel_matrix_multiplication_auto_threads(
            matrix_a,
            matrix_b,
            intermediate_file
        )

        # Запись результата в файл с добавлением новой строки
        write_matrix_append(result_file, result_matrix)

        # Задержка перед следующей генерацией
        time.sleep(delay)


if __name__ == "__main__":
    result_file = "result_matrix_async.txt"
    intermediate_file = "intermediate_file_async.txt"

    # Очистка файла с результатами перед началом работы
    if os.path.exists(result_file):
        open(result_file, "w").close()

    # Генерация и умножение матриц
    generator_and_multiplier(result_file, intermediate_file, size=3, iterations=100, delay=2)
