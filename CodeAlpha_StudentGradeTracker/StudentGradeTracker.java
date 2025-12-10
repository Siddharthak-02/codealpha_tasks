import java.util.ArrayList;
import java.util.Scanner;

public class StudentGradeTracker {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ArrayList<Integer> grades = new ArrayList<>();

        System.out.println("===== Student Grade Tracker =====");

        while (true) {
            System.out.print("Enter student grade (or -1 to finish): ");
            int grade = scanner.nextInt();

            if (grade == -1) {
                break;
            }

            if (grade < 0 || grade > 100) {
                System.out.println("Invalid grade! Please enter a number between 0 and 100.");
            } else {
                grades.add(grade);
            }
        }

        if (grades.isEmpty()) {
            System.out.println("No grades entered.");
            return;
        }

        int sum = 0;
        int highest = grades.get(0);
        int lowest = grades.get(0);

        for (int g : grades) {
            sum += g;
            if (g > highest) highest = g;
            if (g < lowest) lowest = g;
        }

        double average = (double) sum / grades.size();

        System.out.println("\n===== Grade Summary Report =====");
        System.out.println("Total Students: " + grades.size());
        System.out.println("Average Score: " + average);
        System.out.println("Highest Score: " + highest);
        System.out.println("Lowest Score: " + lowest);

        System.out.println("\nAll Grades: " + grades);
    }
}