import numpy as np
from scipy import sparse

if __name__ == '__main__':
    # Create a vector as a row
    vector_row = np.array([1, 2, 3])

    # Create a vector as a column
    vector_column = np.array([[1],
                              [2],
                              [3]])

    # Create a matrix
    matrix = np.array([[1, 2],
                       [1, 2],
                       [1, 2]])

    matrix_object = np.mat([[1, 2],
                            [1, 2],
                            [1, 2]])

    matrix_not_sparse = np.array([[0, 0],
                                  [0, 1],
                                  [3, 0]])

    # Create compressed sparse row (CSR) matrix
    matrix_sparse = sparse.csr_matrix(matrix_not_sparse)

    # Create larger matrix
    matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Create compressed sparse row (CSR) matrix
    matrix_large_sparse = sparse.csr_matrix(matrix_large)

    # View original sparse matrix
    print(matrix_sparse)

    # View large sparse matrix
    print(matrix_large_sparse)

    # Create row vector
    vector = np.array([1, 2, 3, 4, 5, 6])

    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Select third element of vector
    print(vector[2])

    # Select second row, second column
    print(matrix[1,1])

    # Select all elements of a vector
    print(vector[:])

    # Select everything up to and including the third element
    print(vector[:3])

    # Select everything after the third element
    print(vector[3:])

    # Select the last element
    print(vector[-1])

    # Select the first two rows and all columns of a matrix
    print(matrix[:2,:])

    # Select all rows and the second column
    print(matrix[:,1:2])

    matrix = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
    ])

    # View number of rows and columns
    print(matrix.shape)

    # View number of elements (rows * columns)
    print(matrix.size)

    # View number of dimensions
    print(matrix.ndim)

    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Create function that adds 100 to something
    add_100 = lambda i: i + 100

    # Create vectorized function
    vectorized_add_100 = np.vectorize(add_100)

    # Apply function to all elements in matrix
    print(vectorized_add_100(matrix))