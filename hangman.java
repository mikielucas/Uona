import java.util.Scanner;

public class Hangman {
    static String[] words = {"JAVA", "COMPUTER", "PROGRAMMING", "SCHOOL", "STUDENT", "TEACHER"};
    static String secretWord;
    static char[] guessedWord;
    static int wrongGuesses = 0;
    static final int MAX_WRONG = 6;
    static Scanner input = new Scanner(System.in);

    public static void main(String[] args) {
        System.out.println("=================================");
        System.out.println("    WELCOME TO HANGMAN GAME!    ");
        System.out.println("=================================");
        System.out.println("Instructions:");
        System.out.println("- Guess the secret word letter by letter");
        System.out.println("- You have " + MAX_WRONG + " wrong guesses allowed");
        System.out.println("- Enter one letter at a time");

        setupGame();

        while (true) {
            displayGame();

            if (isWordGuessed()) {
                System.out.println("CONGRATULATIONS! You guessed the word: " + secretWord);
                break;
            }

            if (wrongGuesses >= MAX_WRONG) {
                displayHangman();
                System.out.println("GAME OVER! The word was: " + secretWord);
                break;
            }

            makeGuess();
        }

        System.out.println("Thanks for playing Hangman!");
        input.close();
    }

    static void setupGame() {
        int randomIndex = (int)(Math.random() * words.length);
        secretWord = words[randomIndex];

        guessedWord = new char[secretWord.length()];
        for (int i = 0; i < secretWord.length(); i++) {
            guessedWord[i] = '_';
        }

        System.out.println("Game setup complete! Word selected.");
        System.out.println("Word length: " + secretWord.length() + " letters");
    }

    static void displayGame() {
        System.out.println();
        System.out.println("========================================");

        displayHangman();

        System.out.print("Word: ");
        for (char c : guessedWord) {
            System.out.print(c + " ");
        }
        System.out.println();

        System.out.println("Wrong guesses: " + wrongGuesses + "/" + MAX_WRONG);
        System.out.println("========================================");
    }

    static void displayHangman() {
        System.out.println("  +---+");
        System.out.println("  |   |");

        switch (wrongGuesses) {
            case 0:
                System.out.println("      |");
                System.out.println("      |");
                System.out.println("      |");
                break;
            case 1:
                System.out.println("  O   |");
                System.out.println("      |");
                System.out.println("      |");
                break;
            case 2:
                System.out.println("  O   |");
                System.out.println("  |   |");
                System.out.println("      |");
                break;
            case 3:
                System.out.println("  O   |");
                System.out.println(" /|   |");
                System.out.println("      |");
                break;
            case 4:
                System.out.println("  O   |");
                System.out.println(" /|\\  |");
                System.out.println("      |");
                break;
            case 5:
                System.out.println("  O   |");
                System.out.println(" /|\\  |");
                System.out.println(" /    |");
                break;
            case 6:
                System.out.println("  O   |");
                System.out.println(" /|\\  |");
                System.out.println(" / \\  |");
                break;
        }
        System.out.println("      |");
        System.out.println("=========");
    }

    static void makeGuess() {
        boolean validGuess = false;

        while (!validGuess) {
            System.out.print("Enter your guess (single letter): ");

            try {
                String inputStr = input.nextLine().trim().toUpperCase();

                if (inputStr.length() == 0) {
                    System.out.println("Error: Please enter a letter!");
                    continue;
                }

                if (inputStr.length() > 1) {
                    System.out.println("Error: Please enter only ONE letter!");
                    continue;
                }

                char guess = inputStr.charAt(0);

                if (!Character.isLetter(guess)) {
                    System.out.println("Error: Please enter a LETTER only!");
                    continue;
                }

                if (isAlreadyGuessed(guess)) {
                    System.out.println("Error: You already guessed '" + guess + "'! Try a different letter.");
                    continue;
                }

                processGuess(guess);
                validGuess = true;

            } catch (Exception e) {
                System.out.println("Error: Invalid input! Please try again.");
            }
        }
    }

    static boolean isAlreadyGuessed(char letter) {
        for (char c : guessedWord) {
            if (c == letter) {
                return true;
            }
        }
        return false;
    }

    static void processGuess(char guess) {
        boolean correctGuess = false;

        for (int i = 0; i < secretWord.length(); i++) {
            if (secretWord.charAt(i) == guess) {
                guessedWord[i] = guess;
                correctGuess = true;
            }
        }

        if (correctGuess) {
            System.out.println("Great! '" + guess + "' is in the word!");
        } else {
            wrongGuesses++;
            System.out.println("Sorry! '" + guess + "' is not in the word.");
            System.out.println("Wrong guesses: " + wrongGuesses + "/" + MAX_WRONG);
        }
    }

    static boolean isWordGuessed() {
        for (char c : guessedWord) {
            if (c == '_') {
                return false;
            }
        }
        return true;
    }
}
