import numpy as np
from multiprocessing import Pool, cpu_count


def read_matrix(file_path):
    """Считывает матрицу из файла."""
    return np.loadtxt(file_path, dtype=int)


def write_matrix(file_path, matrix):
    """Записывает матрицу в файл."""
    np.savetxt(file_path, matrix, fmt='%d')


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


def parallel_matrix_multiplication_auto_threads(matrix_a, matrix_b, intermediate_file, result_file):
    """Перемножает матрицы с автоматическим определением количества потоков."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Матрицы должны быть одного размера.")

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
    result_matrix = build_result_matrix(intermediate_file, matrix_a.shape)

    # Запись итоговой матрицы в результирующий файл
    write_matrix(result_file, result_matrix)


if __name__ == "__main__":
    # Пример использования
    matrix_a = read_matrix("matrix_a.txt")
    matrix_b = read_matrix("matrix_b.txt")
    parallel_matrix_multiplication_auto_threads(
        matrix_a,
        matrix_b,
        intermediate_file="intermediate_file_threads.txt",
        result_file="result_matrix_threads.txt"
    )
