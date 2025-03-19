package jcuda.core;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaMemcpyKind;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.driver.JCudaDriver.*;

public class Tensor {
    private float[] data;
    private Pointer gpuData;
    private int size;
    private boolean onGPU;
    private int[] shape;

    // Kernel functions for element-wise operations on the GPU
    private static CUfunction addFunction;
    private static CUfunction mulFunction;

    static {
        // If GPU is available, compile and load the kernels
        if (JCudaManager.isGpuAvailable()) {
            JCudaDriver.setExceptionsEnabled(true);
            JCudaDriver.cuInit(0);

            String kernelSource =
                    "extern \"C\" __global__ void addKernel(float* A, float* B, float* C, int N) { " +
                            "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                            "    if (i < N) { C[i] = A[i] + B[i]; } " +
                            "}" +
                            "extern \"C\" __global__ void mulKernel(float* A, float* B, float* C, int N) { " +
                            "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                            "    if (i < N) { C[i] = A[i] * B[i]; } " +
                            "}";

            CUmodule module = new CUmodule();
            cuModuleLoadData(module, kernelSource);

            addFunction = new CUfunction();
            cuModuleGetFunction(addFunction, module, "addKernel");

            mulFunction = new CUfunction();
            cuModuleGetFunction(mulFunction, module, "mulKernel");
        }
    }

    // When only 1D data is provided, the default shape is set to [data.length]
    public Tensor(float[] data) {
        this(data, new int[] { data.length });
    }

    // Constructs a Tensor with the provided data and shape.
    // The product of the elements in newShape must match data.length.
    public Tensor(float[] data, int[] shape) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Data array must not be null or empty.");
        }
        int totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }
        if (totalSize != data.length) {
            throw new IllegalArgumentException("Data length does not match the provided shape.");
        }
        this.data = data;
        this.shape = shape.clone();
        this.size = data.length;
        this.onGPU = false;
    }

    /**
     * Transfers the data to the GPU or CPU.
     * @param device The target device; DeviceManager.Device.GPU or DeviceManager.Device.CPU
     * @param gpuIndex The GPU device index
     */
    public void to(DeviceManager.Device device, int gpuIndex) {
        if (device == DeviceManager.Device.GPU && !onGPU) {
            DeviceManager.setDevice(gpuIndex);
            gpuData = new Pointer();
            cudaMalloc(gpuData, (long) size * Sizeof.FLOAT);
            cudaMemcpy(gpuData, Pointer.to(data), (long) size * Sizeof.FLOAT, cudaMemcpyHostToDevice);
            onGPU = true;
        } else if (device == DeviceManager.Device.CPU && onGPU) {
            cudaMemcpy(Pointer.to(data), gpuData, (long) size * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(gpuData);
            gpuData = null;
            onGPU = false;
        }
    }

    /**
     * Performs element-wise addition.
     * Both tensors must have the same number of elements.
     * Uses GPU kernel if both tensors are on the GPU; otherwise, computes on the CPU.
     */
    public Tensor add(Tensor other) {
        if (this.size != other.size) {
            throw new IllegalArgumentException("Tensors must have the same number of elements for addition.");
        }
        float[] resultData = new float[this.size];
        Tensor result = new Tensor(resultData, this.shape);

        if (this.onGPU && other.onGPU && JCudaManager.isGpuAvailable()) {
            Pointer d_A = this.gpuData;
            Pointer d_B = other.gpuData;
            Pointer d_C = new Pointer();
            cudaMalloc(d_C, (long) size * Sizeof.FLOAT);

            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(d_A),
                    Pointer.to(d_B),
                    Pointer.to(d_C),
                    Pointer.to(new int[] { size })
            );

            cuLaunchKernel(addFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);

            cudaMemcpy(Pointer.to(resultData), d_C, (long) size * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(d_C);
        } else {
            for (int i = 0; i < size; i++) {
                resultData[i] = this.data[i] + other.data[i];
            }
        }
        return result;
    }

    /**
     * Performs element-wise multiplication.
     * Both tensors must have the same number of elements.
     * Uses GPU kernel if available; otherwise, computes on the CPU.
     */
    public Tensor multiply(Tensor other) {
        if (this.size != other.size) {
            throw new IllegalArgumentException("Tensors must have the same number of elements for multiplication.");
        }
        float[] resultData = new float[this.size];
        Tensor result = new Tensor(resultData, this.shape);

        if (this.onGPU && other.onGPU && JCudaManager.isGpuAvailable()) {
            Pointer d_A = this.gpuData;
            Pointer d_B = other.gpuData;
            Pointer d_C = new Pointer();
            cudaMalloc(d_C, (long) size * Sizeof.FLOAT);

            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(d_A),
                    Pointer.to(d_B),
                    Pointer.to(d_C),
                    Pointer.to(new int[] { size })
            );

            cuLaunchKernel(mulFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParams, null);

            cudaMemcpy(Pointer.to(resultData), d_C, (long) size * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
            cudaFree(d_C);
        } else {
            for (int i = 0; i < size; i++) {
                resultData[i] = this.data[i] * other.data[i];
            }
        }
        return result;
    }

    /**
     * Reshapes the tensor.
     * The new shape's total number of elements must match the current tensor size.
     * @param newShape The new tensor shape (e.g., {batch, height, width, channels})
     */
    public void reshape(int[] newShape) {
        int newSize = 1;
        for (int dim : newShape) {
            newSize *= dim;
        }
        if (newSize != this.size) {
            throw new IllegalArgumentException("New shape must have the same number of elements as the current tensor.");
        }
        this.shape = newShape.clone();
    }

    /**
     * Returns the tensor's shape.
     */
    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Returns the underlying data array.
     */
    public float[] getData() {
        return data; // You can also return data.clone() to avoid exposing the internal array.
    }

    /**
     * Converts multi-dimensional indices to a 1D index (row-major order).
     */
    private int getFlatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Number of indices must match tensor rank.");
        }
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index out of bounds at dimension " + i);
            }
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

    /**
     * Returns the value at the specified multi-dimensional index.
     */
    public float get(int... indices) {
        return data[getFlatIndex(indices)];
    }

    /**
     * Sets the value at the specified multi-dimensional index.
     */
    public void set(float value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }

    /**
     * Slices the tensor along a specified dimension.
     * For example, for a 4D tensor, slice(dim, index) removes the given dimension at the specified index
     * and returns a new tensor with the remaining dimensions.
     */
    public Tensor slice(int dim, int index) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension for slicing.");
        }
        if (index < 0 || index >= shape[dim]) {
            throw new IllegalArgumentException("Index out of range for the specified dimension.");
        }

        // New shape: remove the specified dimension.
        int[] newShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != dim) {
                newShape[j++] = shape[i];
            }
        }

        int newSize = 1;
        for (int d : newShape) {
            newSize *= d;
        }

        float[] newData = new float[newSize];

        // For each linear index of the new tensor, convert to the original tensor's multi-dimensional index.
        int newRank = newShape.length;
        int[] newStrides = new int[newRank];
        newStrides[newRank - 1] = 1;
        for (int i = newRank - 2; i >= 0; i--) {
            newStrides[i] = newStrides[i + 1] * newShape[i + 1];
        }

        // For every linear index:
        for (int lin = 0; lin < newSize; lin++) {
            int rem = lin;
            int[] newIndices = new int[newRank];
            for (int i = 0; i < newRank; i++) {
                newIndices[i] = rem / newStrides[i];
                rem = rem % newStrides[i];
            }
            // Convert new indices to the original tensor indices:
            int[] origIndices = new int[shape.length];
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i == dim) {
                    origIndices[i] = index;
                } else {
                    origIndices[i] = newIndices[j++];
                }
            }
            int flatIndex = getFlatIndex(origIndices);
            newData[lin] = data[flatIndex];
        }

        return new Tensor(newData, newShape);
    }

    /**
     * Prints the tensor's shape and data to the console.
     */
    public void printTensor() {
        System.out.print("Shape: [");
        for (int dim : shape) {
            System.out.print(dim + " ");
        }
        System.out.println("]");
        System.out.print("Data: [");
        for (float v : data) {
            System.out.print(v + " ");
        }
        System.out.println("]");
    }
}
