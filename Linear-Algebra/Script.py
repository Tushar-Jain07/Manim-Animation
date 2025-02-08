from fractions import Fraction

def print_matrix_frac(matrix, augmented=True):
    """Print matrix with fractions. If augmented, prints with vertical separator."""
    n = len(matrix)
    mid = n if not augmented else len(matrix[0]) // 2

    for row in matrix:
        print('[', end=' ')
        for j, elem in enumerate(row):
            if j == mid and augmented:
                print('|', end=' ')
            frac = Fraction(elem).limit_denominator()
            if frac.denominator == 1:
                print(f"{frac.numerator:3}", end=' ')
            else:
                print(f"{frac!s:8}", end=' ')
        print(']')
    print()

def gauss_jordan_inverse_with_steps(matrix):
    """
    Find the inverse of a square matrix using Gauss-Jordan elimination.
    Shows each step of the transformation.
    """
    n = len(matrix)
    # Create augmented matrix [A|I]
    augmented = [[Fraction(0) for _ in range(2 * n)] for _ in range(n)]

    # Fill the augmented matrix
    for i in range(n):
        for j in range(n):
            augmented[i][j] = Fraction(matrix[i][j])
    for i in range(n):
        augmented[i][n + i] = Fraction(1)

    print("Initial augmented matrix:")
    print_matrix_frac(augmented)

    step = 1
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        pivot = augmented[i][i]
        if pivot == 0:
            # Look for non-zero pivot in rows below
            for j in range(i + 1, n):
                if augmented[j][i] != 0:
                    print(f"\nStep {step}: Swap row {i + 1} with row {j + 1}")
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    print_matrix_frac(augmented)
                    pivot = augmented[i][i]
                    step += 1
                    break
            if pivot == 0:
                return None

        # Normalize row
        if pivot != 1:
            print(f"\nStep {step}: Divide row {i + 1} by {pivot}")
            for j in range(2 * n):
                augmented[i][j] /= pivot  # Fix: Pivot division should be straightforward
            print_matrix_frac(augmented)
            step += 1

        # Eliminate column
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                if factor != 0:
                    print(f"\nStep {step}: Subtract {factor} times row {i + 1} from row {j + 1}")
                    for k in range(2 * n):
                        augmented[j][k] -= factor * augmented[i][k]
                    print_matrix_frac(augmented)
                    step += 1

    # Extract inverse matrix
    inverse = [[augmented[i][j] for j in range(n, 2 * n)] for i in range(n)]
    return inverse

def get_square_matrix():
    """Get a square matrix from user input."""
    matrix = []
    print("Enter a square matrix row by row (comma-separated values). Type 'q' to quit.")

    while True:
        row_input = input(f"Enter row {len(matrix) + 1}: ")
        if row_input.lower() == 'q':
            print("Exiting...")
            return None

        try:
            row = list(map(Fraction, row_input.split(',')))
            if len(matrix) > 0 and len(row) != len(matrix[0]):
                print("All rows must have the same number of columns.")
                continue
            matrix.append(row)

            # Ensure square matrix
            if len(matrix) > 1 and len(matrix) == len(matrix[0]):
                break  # Stop when a valid square matrix is entered

        except ValueError:
            print("Invalid input. Enter numeric values separated by commas.")

    return matrix

def main():
    while True:
        matrix = get_square_matrix()
        if matrix is None:
            break

        print("Input matrix:")
        print_matrix_frac(matrix, augmented=False)
        print("Finding inverse using Gauss-Jordan elimination:")
        inverse = gauss_jordan_inverse_with_steps(matrix)
        if inverse:
            print("\nFinal inverse matrix:")
            print_matrix_frac(inverse, augmented=False)
        else:
            print("Matrix is not invertible")

if __name__ == "__main__":
    main()
