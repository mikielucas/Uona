/**
 * ShapeCalculator manages a list of shapes and prints their areas.
 */
import java.util.ArrayList;
import java.util.List;

public class ShapeCalculator {
    private List<Shape> shapes = new ArrayList<>();

    public void addShape(Shape shape) {
        shapes.add(shape);
    }

    public void printAreas() {
        for (Shape shape : shapes) { // Repetition
            try {
                double area = shape.getArea();
                if (area < 0) throw new Exception("Area cannot be negative!");
                System.out.println(shape.getName() + " area: " + area);
            } catch (Exception e) { // Exception handling
                System.out.println("Error calculating area for " + shape.getName() + ": " + e.getMessage());
            }
        }
    }
}
