import numpy as np
from fractions import Fraction

# Step 1: Row-Echelon Form (REF) Transformation
def row_echelon_form(matrix):
    A = np.array(matrix, dtype=object)  # Ensure matrix uses objects (fractions)
    rows, cols = A.shape
    r = 0

    for c in range(cols):
        if r >= rows:
            break

        if A[r, c] == 0:
            non_zero_row = np.nonzero(A[r+1:, c])[0]
            if len(non_zero_row) == 0:
                continue
            A[[r, r + non_zero_row[0] + 1], :] = A[[r + non_zero_row[0] + 1, r], :]
            print(f"Step 1.1: Row swap to bring non-zero element to pivot position\n{format_matrix(A)}")

        if A[r, c] != 0:
            A[r, :] = [Fraction(A[r, j], A[r, c]) for j in range(cols)]  # Convert to fractions
            print(f"Step 1.2: Normalize row {r} to make pivot 1\n{format_matrix(A)}")

            for i in range(r + 1, rows):
                if A[i, c] != 0:
                    A[i, :] = [A[i, j] - A[i, c] * A[r, j] for j in range(cols)]
                    print(f"Step 1.3: Row operation to eliminate element at position ({i}, {c})\n{format_matrix(A)}")

            r += 1

    return A

# Step 2: Reduced Row-Echelon Form (RREF) Transformation
def reduced_row_echelon_form(matrix):
    A = row_echelon_form(matrix)
    rows, cols = A.shape

    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if A[r, c] == 1:
                for i in range(r - 1, -1, -1):
                    if A[i, c] != 0:
                        A[i, :] = [A[i, j] - A[i, c] * A[r, j] for j in range(cols)]
                        print(f"Step 2.3: Eliminate non-zero entry above pivot in column {c}\n{format_matrix(A)}")
                break
    return A

# Step 3: Calculate the rank of the matrix
def rank(matrix):
    rref_matrix = reduced_row_echelon_form(matrix)
    rank = sum(1 for row in rref_matrix if any(row))  # Count non-zero rows
    return rank

# Step 4: Function to get the user-defined matrix input
def user_input_matrix():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrix = []
    print("Enter the matrix elements row by row:")
    
    for i in range(rows):
        row = list(map(Fraction, input(f"Enter row {i + 1}: ").split()))  # Convert input to Fraction
        if len(row) != cols:
            print(f"Error: Row {i + 1} must have exactly {cols} elements.")
            return None
        matrix.append(row)

    return np.array(matrix, dtype=object)  # Store as object array for Fraction compatibility

# Step 5: Helper function to format matrix output
def format_matrix(matrix):
    return '\n'.join(['\t'.join(map(str, row)) for row in matrix])

# Step 6: Main function to drive the process
def main():
    matrix = user_input_matrix()
    if matrix is None:
        return

    print("\nOriginal Matrix:")
    print(format_matrix(matrix))

    ref_matrix = row_echelon_form(matrix)
    print("\nRow-Echelon Form (REF):")
    print(format_matrix(ref_matrix))

    rref_matrix = reduced_row_echelon_form(matrix)
    print("\nReduced Row-Echelon Form (RREF):")
    print(format_matrix(rref_matrix))

    matrix_rank = rank(matrix)
    print("\nRank of the matrix:", matrix_rank)

# Step 7: Execute the program
if __name__ == "__main__":
    main()
