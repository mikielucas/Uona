import java.util.Scanner;

public class TicTacToe {
    static char[] board = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    static char player = 'X';

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int gameCount = 0;

        while (true) {
            showBoard();
            System.out.print("Player " + player + ", choose position (1-9): ");
            try {
                int pos = input.nextInt() - 1;

            if (pos >= 0 && pos < 9 && board[pos] != 'X' && board[pos] != 'O') {
                board[pos] = player;

                if (checkWin()) {
                    showBoard();
                    System.out.println("Player " + player + " wins!");
                    break;
                }

                if (checkTie()) {
                    showBoard();
                    System.out.println("It's a tie!");
                    break;
                }

                player = (player == 'X') ? 'O' : 'X';
            } catch (Exception e) {
                System.out.println("Please enter a number between 1-9!");
                input.nextLine(); // clear invalid input
            }
            } else {
                System.out.println("Invalid move! Try again.");
            }
        }

        input.close();
    }

    static void showBoard() {
        System.out.println("\n " + board[0] + " | " + board[1] + " | " + board[2]);
        System.out.println("-----------");
        System.out.println(" " + board[3] + " | " + board[4] + " | " + board[5]);
        System.out.println("-----------");
        System.out.println(" " + board[6] + " | " + board[7] + " | " + board[8]);
        System.out.println();
    }

    static boolean checkWin() {
        return (board[0] == board[1] && board[1] == board[2]) ||
               (board[3] == board[4] && board[4] == board[5]) ||
               (board[6] == board[7] && board[7] == board[8]) ||
               (board[0] == board[3] && board[3] == board[6]) ||
               (board[1] == board[4] && board[4] == board[7]) ||
               (board[2] == board[5] && board[5] == board[8]) ||
               (board[0] == board[4] && board[4] == board[8]) ||
               (board[2] == board[4] && board[4] == board[6]);
    }

    static boolean checkTie() {
        for (char c : board) {
            if (c != 'X' && c != 'O') return false;
        }
        return true;
    }
}
