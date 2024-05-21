import numpy as np

# https://docs.nvidia.com/cuda/cusparse/#cusparse-basic-apis
def dense_to_csr(dense_matrix):
    data = []
    indices = []
    indptr = [0]  
    
    for row in dense_matrix:
        for col, value in enumerate(row):
            if value != 0:
                data.append(value)
                indices.append(col)
        indptr.append(len(data))  

    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)

    return data, indices, indptr

dense_matrix = np.array([[1, 0, 2, 0], [0, 0, 3, 0], [0, 0, 0, 0], [4, 5, 0, 0], [0, 6, 7, 8]])
data, indices, indptr = dense_to_csr(dense_matrix)
print(data, indices, indptr)

# 1 0 2 0 0 0 3 0 0 0 0 0 4 5 0 0 0 6 7 8
# row * leading_dimension + column_indices[row_offsets[row] + k]
# row - row in dense matrix
# leading_dimension - stride of each row(number of columns in a row)
# k - The index of the non-zero element within the row in the column_indices array.
print(3 * 4 + indices[indptr[3] + 1])



