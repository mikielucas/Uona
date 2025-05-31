import java.util.Scanner;

public class SudokuGame {
    private static final int SIZE = 9;
    private static final int EMPTY = 0;

    // The game board - 0 represents empty cells
    private int[][] board = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    // Track which cells are originally filled (cannot be modified)
    private boolean[][] originalCells = new boolean[SIZE][SIZE];
    private Scanner scanner = new Scanner(System.in);

    public SudokuGame() {
        // Mark original filled cells
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                originalCells[row][col] = (board[row][col] != EMPTY);
            }
        }
    }

    public void playGame() {
        System.out.println("Welcome to Sudoku!");
        System.out.println("Enter numbers 1-9 to fill empty cells.");
        System.out.println("Use row and column numbers 1-9.");
        System.out.println("Enter 'quit' to exit the game.\n");

        while (true) {
            displayBoard();

            if (isSolved()) {
                System.out.println("\n🎉 Congratulations! You solved the puzzle! 🎉");
                break;
            }

            System.out.print("\nEnter your move (row col value) or 'quit': ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("quit")) {
                System.out.println("Thanks for playing!");
                break;
            }

            processMove(input);
        }
    }

    private void displayBoard() {
        System.out.println("\n  1 2 3   4 5 6   7 8 9");
        System.out.println(" ┌─────┬─────┬─────┐");

        for (int row = 0; row < SIZE; row++) {
            if (row == 3 || row == 6) {
                System.out.println(" ├─────┼─────┼─────┤");
            }

            System.out.print((row + 1) + "│");

            for (int col = 0; col < SIZE; col++) {
                if (col == 3 || col == 6) {
                    System.out.print("│");
                }

                if (board[row][col] == EMPTY) {
                    System.out.print(" ");
                } else {
                    System.out.print(board[row][col]);
                }

                if (col < SIZE - 1) {
                    System.out.print(" ");
                }
            }
            System.out.println("│");
        }
        System.out.println(" └─────┴─────┴─────┘");
    }

    private void processMove(String input) {
        String[] parts = input.split("\\s+");

        if (parts.length != 3) {
            System.out.println("Invalid input! Please enter: row col value (e.g., '1 2 5')");
            return;
        }

        try {
            int row = Integer.parseInt(parts[0]) - 1; // Convert to 0-based index
            int col = Integer.parseInt(parts[1]) - 1; // Convert to 0-based index
            int value = Integer.parseInt(parts[2]);

            // Validate input ranges
            if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) {
                System.out.println("Row and column must be between 1 and 9!");
                return;
            }

            if (value < 1 || value > 9) {
                System.out.println("Value must be between 1 and 9!");
                return;
            }

            // Check if cell is modifiable
            if (originalCells[row][col]) {
                System.out.println("Cannot modify original puzzle cells!");
                return;
            }

            // Check if move is valid
            if (isValidMove(row, col, value)) {
                board[row][col] = value;
                System.out.println("Move accepted!");
            } else {
                System.out.println("Invalid move! This number violates Sudoku rules.");
                System.out.println("Check row, column, and 3x3 box for duplicates.");
            }

        } catch (NumberFormatException e) {
            System.out.println("Please enter valid numbers!");
        }
    }

    private boolean isValidMove(int row, int col, int value) {
        // Store original value
        int originalValue = board[row][col];
        board[row][col] = value;

        boolean valid = isValidBoard();

        // Restore original value
        board[row][col] = originalValue;

        return valid;
    }

    private boolean isValidBoard() {
        // Check all rows
        for (int row = 0; row < SIZE; row++) {
            if (!isValidUnit(getRow(row))) {
                return false;
            }
        }

        // Check all columns
        for (int col = 0; col < SIZE; col++) {
            if (!isValidUnit(getColumn(col))) {
                return false;
            }
        }

        // Check all 3x3 boxes
        for (int boxRow = 0; boxRow < 3; boxRow++) {
            for (int boxCol = 0; boxCol < 3; boxCol++) {
                if (!isValidUnit(getBox(boxRow, boxCol))) {
                    return false;
                }
            }
        }

        return true;
    }

    private boolean isValidUnit(int[] unit) {
        boolean[] seen = new boolean[SIZE + 1]; // Index 0 unused, 1-9 for values

        for (int value : unit) {
            if (value != EMPTY) {
                if (seen[value]) {
                    return false; // Duplicate found
                }
                seen[value] = true;
            }
        }
        return true;
    }

    private int[] getRow(int row) {
        return board[row].clone();
    }

    private int[] getColumn(int col) {
        int[] column = new int[SIZE];
        for (int row = 0; row < SIZE; row++) {
            column[row] = board[row][col];
        }
        return column;
    }

    private int[] getBox(int boxRow, int boxCol) {
        int[] box = new int[SIZE];
        int index = 0;

        int startRow = boxRow * 3;
        int startCol = boxCol * 3;

        for (int row = startRow; row < startRow + 3; row++) {
            for (int col = startCol; col < startCol + 3; col++) {
                box[index++] = board[row][col];
            }
        }
        return box;
    }

    private boolean isSolved() {
        // Check if board is completely filled
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (board[row][col] == EMPTY) {
                    return false;
                }
            }
        }

        // Check if board is valid
        return isValidBoard();
    }

    public static void main(String[] args) {
        SudokuGame game = new SudokuGame();
        game.playGame();
    }
}
