import numpy as np
from scipy import sparse

if __name__ == '__main__':
    # 1.1
    # Create a vector as a row
    vector_row = np.array([1, 2, 3])

    # Create a vector as a column
    vector_column = np.array([[1],
                              [2],
                              [3]])

    # 1.2
    # Create a matrix
    matrix = np.array([[1, 2],
                       [1, 2],
                       [1, 2]])

    matrix_object = np.mat([[1, 2],
                            [1, 2],
                            [1, 2]])

    # 1.3
    # Create a matrix not sparse
    matrix_not_sparse = np.array([[0, 0],
                                  [0, 1],
                                  [3, 0]])

    # Create compressed sparse row (CSR) matrix
    matrix_sparse = sparse.csr_matrix(matrix_not_sparse)

    # View sparse matrix
    print(matrix_sparse)

    # Create larger matrix
    matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                             ])

    # Create compressed sparse row (CSR) matrix
    matrix_large_sparse = sparse.csr_matrix(matrix_large)

    # View original sparse matrix
    print(matrix_sparse)

    # View large sparse matrix
    print(matrix_large_sparse)

    # 1.4
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
    print(matrix[1, 1])

    # Select all elements of a vector
    print(vector[:])

    # Select everything up to and including the third element
    print(vector[:3])

    # Select everything after the third element
    print(vector[3:])

    # Select the last element
    print(vector[-1])

    # Select the first two rows and all columns of a matrix
    print(matrix[:2, :])

    # Select all rows and the second column
    print(matrix[:, 1:2])

    # 1.5
    # Create matrix
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    # View number of rows and columns
    print(matrix.shape)

    # View number of elements (rows * columns)
    print(matrix.size)

    # View number of dimensions
    print(matrix.ndim)

    # 1.6
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

    # Add 100 to all elements
    print(matrix + 100)

    # 1.7
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Return maximum element
    print(np.max(matrix))

    # Return minimum element
    print(np.min(matrix))

    # Find maximum element in each column
    print(np.max(matrix, axis=0))

    # Find maximum element in each row
    print(np.max(matrix, axis=1))

    # 1.8
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Return mean
    print(np.mean(matrix))

    # Return variance
    print(np.var(matrix))

    # Return standard deviation
    print(np.std(matrix))

    # Find the mean value in each column
    print(np.mean(matrix, axis=0))

    # 1.9
    # Create 4x3 matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    # Reshape matrix into 2x6 matrix
    print(matrix.reshape(2, 6))

    print(matrix.size)

    print(matrix.reshape(1, -1))

    print(matrix.reshape(12))

    # 1.10
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Transpose matrix
    print(matrix.T)

    # Transpose vector
    print(np.array([1, 2, 3, 4, 5, 6]).T)

    # Transpose row vector
    print(np.array([[1, 2, 3, 4, 5, 6]]).T)

    # 1.11
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Flatten matrix
    print(matrix.flatten())

    print(matrix.reshape(1, -1))

    # 1.12
    # Create matrix
    matrix = np.array([
        [1, 1, 1],
        [1, 1, 10],
        [1, 1, 15]
    ])

    # Return matrix rank
    print(np.linalg.matrix_rank(matrix))

    # 1.13
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Return determinant of matrix
    print(np.linalg.det(matrix))

    # 1.14
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 8, 9]
    ])

    # Return diagonal elements
    print(matrix.diagonal())

    # Return diagonal one above the main diagonal
    print(matrix.diagonal(offset=1))

    # Return diagonal one below the mais diagonal
    print(matrix.diagonal(offset=-1))

    # 1.15
    # Create matrix
    matrix = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 8, 9]
    ])

    # Return trace
    print(matrix.trace())

    # Return diagonal and sum elements
    print(sum(matrix.diagonal()))

    # 1.16
    # Create matrix
    matrix = np.array([
        [1, -1, 3],
        [1, 1, 6],
        [3, 8, 9]
    ])

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # View eigenvalues
    print(eigenvalues)

    # View eigenvectors
    print(eigenvectors)

    # 1.17
    # Create two vectors
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])

    # Calculate dot product
    print(np.dot(vector_a, vector_b))

    print(vector_a @ vector_b)

    # 1.18
    # Create matrix
    matrix_a = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,2]
    ])

    # Create matrix
    matrix_b = np.array([
        [1, 3, 1],
        [1, 3, 1],
        [1, 3, 8]
    ])

    # Add two matrices
    print(np.add(matrix_a, matrix_b))

    # Subtract two matrices
    print(np.subtract(matrix_a, matrix_b))

    # Add two matrices
    print(matrix_a + matrix_b)

    # 1.19
    # Create matrix
    matrix_a = np.array([
        [1,1],
        [1,2]
    ])

    # Create matrix
    matrix_b = np.array([
        [1, 3],
        [1, 2]
    ])

    # Multiply two matrices
    print(np.dot(matrix_a, matrix_b))

    # Multiply two matrices
    print(matrix_a @ matrix_b)

    # Multiply two matrices element-wise
    print(matrix_a * matrix_b)

    # 1.20
    # Create matrix
    matrix = np.array([
        [1, 4],
        [2, 5]
    ])

    # Calculate inverse of matrix
    print(np.linalg.inv(matrix))

    # Multiply matrix and its inverse
    print(matrix @ np.linalg.inv(matrix))

    # 1.21
    # Set seed
    np.random.seed(0)

    # Generate three random floats between 0.0 and 1.0
    print(np.random.random(3))

    # Generate three random integers between 1 and 10
    print(np.random.randint(0,11,3))

    # Draw three numbers from a normal distribution with mean 0.0
    # and standard deviation of 1.0
    print(np.random.normal(0.0, 1.0, 3))

    # Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
    print(np.random.logistic(0.0, 1.0, 3))

    # Draw three numbers greater than or equal to 1.0 and less tham 2.0
    print(np.random.uniform(1.0, 2.0, 3))