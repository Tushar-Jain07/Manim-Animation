import numpy as np
from fractions import Fraction

# Helper function to format matrix output as strings of fractions
def format_matrix(matrix, augmented=False):
    """Format matrix output as strings of fractions."""
    n = len(matrix)
    mid = n if not augmented else len(matrix[0]) // 2

    formatted = ""
    for row in matrix:
        formatted += "["
        for j, elem in enumerate(row):
            if j == mid and augmented:
                formatted += " | "
            frac = Fraction(elem).limit_denominator()
            if frac.denominator == 1:
                formatted += f"{frac.numerator:3}  "
            else:
                formatted += f"{frac!s:6}  "
        formatted += "]\n"
    return formatted

# Step 1: Row-Echelon Form (REF) Transformation with Fractional Arithmetic
def row_echelon_form(matrix):
    # Convert the matrix to a Fraction matrix for exact arithmetic
    A = np.array([[Fraction(e) for e in row] for row in matrix], dtype=object)
    rows, cols = A.shape  # Get the number of rows and columns
    r = 0  # Start from the first row

    print("Initial matrix:")
    print(format_matrix(A))

    step = 1
    # Iterate through each column
    for c in range(cols):
        if r >= rows:  # If we've processed all the rows, break the loop
            break

        # Step 1.1: If the pivot (A[r, c]) is zero, swap rows
        if A[r, c] == 0:
            non_zero_row = np.nonzero(A[r+1:, c])[0]  # Find non-zero elements in the column below the pivot
            if len(non_zero_row) == 0:  # No non-zero element found, continue to next column
                continue
            # Swap the current row with the first row that has a non-zero element in column c
            A[[r, r + non_zero_row[0]], :] = A[[r + non_zero_row[0], r], :]
            print(f"\nStep {step}: Swap row {r + 1} with row {r + non_zero_row[0] + 2}")
            print(format_matrix(A))
            step += 1

        # Step 1.2: If the pivot element is non-zero, normalize the row by dividing by the pivot
        if A[r, c] != 0:
            print(f"\nStep {step}: Divide row {r + 1} by {A[r, c]}")
            A[r, :] = [elem / A[r, c] for elem in A[r, :]]  # Normalize pivot row to make the pivot element 1
            print(format_matrix(A))
            step += 1

            # Step 1.3: Eliminate all elements below the pivot (make them zero)
            for i in range(r + 1, rows):
                if A[i, c] != 0:
                    print(f"\nStep {step}: Subtract {A[i, c]} times row {r + 1} from row {i + 1}")
                    A[i, :] = [A[i, j] - A[i, c] * A[r, j] for j in range(cols)]  # Row operation to eliminate the element below the pivot
                    print(format_matrix(A))
                    step += 1

            r += 1  # Move to the next row

    return A  # Return the Row-Echelon form matrix

# Step 2: Reduced Row-Echelon Form (RREF) Transformation with Fractional Arithmetic
def reduced_row_echelon_form(matrix):
    # Step 2.1: First, get the Row-Echelon Form
    A = row_echelon_form(matrix)
    rows, cols = A.shape
    step = 1

    # Step 2.2: Iterate from the bottom row to the top to make each pivot's column zero above it
    for r in range(rows - 1, -1, -1):  # Traverse the rows in reverse order
        for c in range(cols):  # Traverse the columns in normal order
            if A[r, c] == 1:  # If we find a pivot
                # Step 2.3: Eliminate all non-zero entries above the pivot
                for i in range(r - 1, -1, -1):  # Go up from the row above the pivot
                    if A[i, c] != 0:  # If there's a non-zero value above the pivot
                        print(f"\nStep {step}: Subtract {A[i, c]} times row {r + 1} from row {i + 1}")
                        A[i, :] = [A[i, j] - A[i, c] * A[r, j] for j in range(cols)]  # Row operation to eliminate the non-zero entry above the pivot
                        print(format_matrix(A))
                        step += 1
                break  # Move to the next row after processing the pivot

    return A  # Return the Reduced Row-Echelon form matrix

# Step 3: Calculate the rank of the matrix
def rank(matrix):
    # Step 3.1: Get the Reduced Row-Echelon Form (RREF)
    rref_matrix = reduced_row_echelon_form(matrix)

    # Step 3.2: Count the number of non-zero rows in the RREF (the rank)
    rank = 0
    for row in rref_matrix:
        if any(row != 0):  # If any element in the row is non-zero
            rank += 1  # This is a non-zero row, so it contributes to the rank

    return rank  # Return the rank

# Step 4: Function to get the user-defined matrix input
def user_input_matrix():
    # Step 4.1: Get matrix dimensions
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    
    matrix = []  # Initialize an empty list to store the matrix
    print("Enter the matrix elements row by row:")
    
    # Step 4.2: Get each row from the user input and append it to the matrix
    for i in range(rows):
        row = list(map(Fraction, input(f"Enter row {i + 1}: ").split()))  # Convert input into a list of fractions
        if len(row) != cols:
            print(f"Error: Row {i + 1} must have exactly {cols} elements.")
            return None  # Return None if the row does not have the correct number of elements
        matrix.append(row)  # Add the row to the matrix
    
    return np.array(matrix)  # Convert the list of rows into a numpy array for matrix manipulation

# Step 5: Main function to drive the process
def main():
    matrix = user_input_matrix()  # Step 5.1: Get the matrix input from the user
    if matrix is None:
        return  # Exit if there was an error in input

    # Step 5.2: Display the original matrix
    print("\nOriginal Matrix:")
    print(format_matrix(matrix))

    # Step 5.3: Get the Row-Echelon Form (REF) and display it
    ref_matrix = row_echelon_form(matrix)
    print("\nRow-Echelon Form (REF):")
    print(format_matrix(ref_matrix))
    
    # Step 5.4: Get the Reduced Row-Echelon Form (RREF) and display it
    rref_matrix = reduced_row_echelon_form(matrix)
    print("\nReduced Row-Echelon Form (RREF):")
    print(format_matrix(rref_matrix))
    
    # Step 5.5: Calculate and display the rank of the matrix
    matrix_rank = rank(matrix)
    print("\nRank of the matrix:", matrix_rank)

# Step 6: Execute the program
if __name__ == "__main__":
    main()
