package jcuda.math;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.core.DeviceManager;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import jcuda.core.JCudaManager; // Cihaz yönetimi için varsayılan DeviceManager sınıfını kullanıyoruz.

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Matrix {
    private int rows;
    private int columns;
    private float[] data; // Matris verisi, row-major düzende saklanır.
    private Pointer gpuData;
    private boolean onGPU;

    // GPU üzerinde matris çarpımı için kernel fonksiyonu
    private static CUfunction matMulFunction;

    static {
        if (JCudaManager.isGpuAvailable()) {
            JCudaDriver.setExceptionsEnabled(true);
            JCudaDriver.cuInit(0);

            String kernelSource =
                    "extern \"C\" __global__ void matrixMultiplyKernel(float* A, float* B, float* C, int A_rows, int A_columns, int B_columns) {" +
                            "    int row = blockIdx.y * blockDim.y + threadIdx.y;" +
                            "    int col = blockIdx.x * blockDim.x + threadIdx.x;" +
                            "    if (row < A_rows && col < B_columns) {" +
                            "        float sum = 0.0f;" +
                            "        for (int i = 0; i < A_columns; i++) {" +
                            "            sum += A[row * A_columns + i] * B[i * B_columns + col];" +
                            "        }" +
                            "        C[row * B_columns + col] = sum;" +
                            "    }" +
                            "}";

            CUmodule module = new CUmodule();
            cuModuleLoadData(module, kernelSource);

            matMulFunction = new CUfunction();
            cuModuleGetFunction(matMulFunction, module, "matrixMultiplyKernel");
        }
    }

    // 2D float dizisinden matris oluşturur.
    public Matrix(float[][] array) {
        if (array == null || array.length == 0 || array[0].length == 0) {
            throw new IllegalArgumentException("Invalid matrix dimensions.");
        }
        this.rows = array.length;
        this.columns = array[0].length;
        this.data = new float[rows * columns];
        int index = 0;
        for (int i = 0; i < rows; i++) {
            if (array[i].length != columns) {
                throw new IllegalArgumentException("All rows must have the same number of columns.");
            }
            for (int j = 0; j < columns; j++) {
                data[index++] = array[i][j];
            }
        }
        this.onGPU = false;
    }

    // Flattened data ve satır/sütun bilgisi ile matris oluşturur.
    public Matrix(int rows, int columns, float[] data) {
        if (data.length != rows * columns) {
            throw new IllegalArgumentException("Data length does not match matrix dimensions.");
        }
        this.rows = rows;
        this.columns = columns;
        this.data = data.clone();
        this.onGPU = false;
    }

    // Matrisin satır sayısını döner.
    public int getRows() {
        return rows;
    }

    // Matrisin sütun sayısını döner.
    public int getColumns() {
        return columns;
    }

    // Matris verisini 2D float dizisi olarak döner.
    public float[][] getData() {
        float[][] array = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data, i * columns, array[i], 0, columns);
        }
        return array;
    }

    /**
     * Veriyi GPU'ya veya CPU'ya transfer eder.
     * @param device Hedef cihaz; DeviceManager.Device.GPU veya DeviceManager.Device.CPU
     * @param gpuIndex GPU cihazı indexi
     */
    public void to(DeviceManager.Device device, int gpuIndex) {
        if (device == DeviceManager.Device.GPU && !onGPU) {
            DeviceManager.setDevice(gpuIndex);
            gpuData = new Pointer();
            cudaMalloc(gpuData, (long) data.length * Sizeof.FLOAT);
            cudaMemcpy(gpuData, Pointer.to(data), (long) data.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
            onGPU = true;
        } else if (device == DeviceManager.Device.CPU && onGPU) {
            cudaMemcpy(Pointer.to(data), gpuData, (long) data.length * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(gpuData);
            gpuData = null;
            onGPU = false;
        }
    }

    /**
     * Matris çarpımı işlemi: this * other
     * this.columns'ın other.rows'a eşit olması gerekir.
     * GPU üzerinde her iki matris de mevcutsa kernel ile, aksi halde CPU üzerinde hesaplanır.
     */
    public Matrix multiply(Matrix other) {
        if (this.columns != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }
        int resultRows = this.rows;
        int resultColumns = other.columns;
        float[] resultData = new float[resultRows * resultColumns];
        Matrix result = new Matrix(resultRows, resultColumns, resultData);

        if (this.onGPU && other.onGPU && JCudaManager.isGpuAvailable()) {
            // GPU ile matris çarpımı
            Pointer d_A = this.gpuData;
            Pointer d_B = other.gpuData;
            Pointer d_C = new Pointer();
            cudaMalloc(d_C, (long) resultData.length * Sizeof.FLOAT);

            // Kernel parametreleri: A, B, C, A_rows, A_columns, B_columns
            Pointer kernelParams = Pointer.to(
                    Pointer.to(d_A),
                    Pointer.to(d_B),
                    Pointer.to(d_C),
                    Pointer.to(new int[] { this.rows }),
                    Pointer.to(new int[] { this.columns }),
                    Pointer.to(new int[] { other.columns })
            );

            int blockSizeX = 16;
            int blockSizeY = 16;
            int gridSizeX = (resultColumns + blockSizeX - 1) / blockSizeX;
            int gridSizeY = (resultRows + blockSizeY - 1) / blockSizeY;

            cuLaunchKernel(matMulFunction,
                    gridSizeX, gridSizeY, 1,
                    blockSizeX, blockSizeY, 1,
                    0, null,
                    kernelParams, null);

            cudaMemcpy(Pointer.to(resultData), d_C, (long) resultData.length * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(d_C);
        } else {
            // CPU üzerinden matris çarpımı (dot product hesaplaması)
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < other.columns; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < this.columns; k++) {
                        sum += this.data[i * this.columns + k] * other.data[k * other.columns + j];
                    }
                    resultData[i * resultColumns + j] = sum;
                }
            }
        }
        return result;
    }

    /**
     * Eleman bazlı toplama işlemi.
     * İki matrisin boyutları aynı olmalıdır.
     * Eğer her iki matris GPU'da ise (ve GPU mevcutsa) kernel; aksi halde CPU üzerinden hesaplanır.
     */
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.columns != other.columns) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition.");
        }
        float[] resultData = new float[this.data.length];
        Matrix result = new Matrix(this.rows, this.columns, resultData);

        // Basit CPU toplama işlemi
        for (int i = 0; i < this.data.length; i++) {
            resultData[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    // Matrisin 2D formatta ekrana yazdırılması
    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.printf("%.2f ", data[i * columns + j]);
            }
            System.out.println();
        }
    }
}
