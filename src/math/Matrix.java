package math;
import java.util.concurrent.*;

public class Matrix {
    private final int rows, columns;
    private final double[][] data;

    public Matrix(double[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Matrix dimensions mismatched.");
        }
        this.rows = data.length;
        this.columns = data[0].length;
        this.data = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            if (data[i].length != columns) {
                throw new IllegalArgumentException("All rows should be in same length");
            }
            System.arraycopy(data[i], 0, this.data[i], 0, columns);
        }
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double[][] getData(){
        double[][] copy = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, columns);
        }
        return copy;
    }

    public Matrix parallelMultiply(Matrix other) {
        if (this.columns != other.rows) {
            throw new IllegalArgumentException("Matrix cannot multiply: Dimensions mismatch.");
        }

        double[][] result = new double[this.rows][other.columns];
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        try {
            for (int i = 0; i < this.rows; i++) {
                final int row = i;
                executor.execute(() -> {
                    for (int j = 0; j < other.columns; j++) {
                        for (int k = 0; k < this.columns; k++) {
                            result[row][j] += this.data[row][k] * other.data[k][j];
                        }
                    }
                });
            }
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)) {
                    System.err.println("Executor did not terminate in the expected time.");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        return new Matrix(result);
    }

    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.printf("%.2f ", data[i][j]);
            }
            System.out.println();
        }
    }
}
