/**
 * Main class to demonstrate the simple shape area calculator
 */
public class Main {
    public static void main(String[] args) {
        ShapeCalculator calc = new ShapeCalculator();
        calc.addShape(new Circle(2));
        calc.addShape(new Rectangle(3, 4));
        calc.addShape(new Circle(-1)); // Will trigger exception
        calc.printAreas();
    }
}
