/**
 * Simple Clock class to represent time
 */
class Clock {
    protected int hours;
    protected int minutes;
    
    // Constructor
    public Clock(int hours, int minutes) {
        this.hours = hours;
        this.minutes = minutes;
    }
    
    // Display time
    public void showTime() {
        System.out.println(hours + ":" + minutes);
    }
    
    // Get hours
    public int getHours() {
        return hours;
    }
    
    // Get minutes  
    public int getMinutes() {
        return minutes;
    }
}

/**
 * ExtClock class - Inheritance from Clock
 * Adds timezone functionality
 */
class ExtClock extends Clock {
    private String timeZone;
    
    // Constructor
    public ExtClock(int hours, int minutes, String timeZone) {
        super(hours, minutes); // Call parent constructor
        this.timeZone = timeZone;
    }
    
    // Override showTime to include timezone
    @Override
    public void showTime() {
        System.out.println(hours + ":" + minutes + " " + timeZone);
    }
    
    // Get timezone
    public String getTimeZone() {
        return timeZone;
    }
    
    // Set new timezone
    public void setTimeZone(String newZone) {
        this.timeZone = newZone;
    }
}

/**
 * Custom Exception for Exception Handling
 */
class TimeException extends Exception {
    public TimeException(String message) {
        super(message);
    }
}

/**
 * Test program
 */
public class ClockTest {
    public static void main(String[] args) {
        System.out.println("=== Simple Clock Test ===");
        
        // Test basic Clock
        Clock clock = new Clock(2, 30);
        System.out.print("Regular clock: ");
        clock.showTime();
        
        // Test ExtClock (Inheritance)
        ExtClock extClock = new ExtClock(2, 30, "EST");
        System.out.print("Extended clock: ");
        extClock.showTime();
        
        // Repetition: Test multiple timezones
        String[] zones = {"PST", "EST", "UTC"};
        System.out.println("\nTesting multiple timezones (Repetition):");
        for (int i = 0; i < zones.length; i++) {
            extClock.setTimeZone(zones[i]);
            System.out.print("Zone " + zones[i] + ": ");
            extClock.showTime();
        }
        
        // Exception Handling
        System.out.println("\nTesting Exception Handling:");
        try {
            if (extClock.getHours() > 24) {
                throw new TimeException("Invalid hour!");
            }
            System.out.println("Time is valid");
        } catch (TimeException e) {
            System.out.println("Error: " + e.getMessage());
        }
        
        System.out.println("Test complete!");
    }
}
