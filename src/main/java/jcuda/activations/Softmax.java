package jcuda.activations;

import jcuda.Pointer;
import jcuda.Sizeof;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class Softmax extends BaseActivation {
    private static final String KERNEL_SOURCE =
            "__global__ void softmaxKernel(float *input, float *output, int n) {" +
                    "    extern __shared__ float sharedExp[];" +
                    "    int i = threadIdx.x + blockIdx.x * blockDim.x;" +
                    "    if (i < n) {" +
                    "        sharedExp[threadIdx.x] = expf(input[i]);" +
                    "        __syncthreads();" +
                    "        float sum = 0.0f;" +
                    "        for (int j = 0; j < n; j++) {" +
                    "            sum += sharedExp[j];" +
                    "        }" +
                    "        output[i] = sharedExp[threadIdx.x] / sum;" +
                    "    }" +
                    "}";
    private static final String KERNEL_NAME = "softmaxKernel";

    public Softmax() {
        super(KERNEL_SOURCE, KERNEL_NAME);
    }

    /**
     * GPU için özel uygulama: dinamik paylaşımlı bellek boyutunu kernel launch sırasında belirliyoruz.
     */
    @Override
    protected float[] applyGPU(float[] input) {
        int n = input.length;
        float[] output = new float[n];

        Pointer d_input = new Pointer();
        Pointer d_output = new Pointer();

        cudaMalloc(d_input, (long) n * Sizeof.FLOAT);
        cudaMalloc(d_output, (long) n * Sizeof.FLOAT);
        cudaMemcpy(d_input, Pointer.to(input), (long) n * Sizeof.FLOAT, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Burada her blok için shared memory boyutu belirleniyor
        int sharedMemSize = threadsPerBlock * Sizeof.FLOAT;

        Pointer kernelParams = Pointer.to(
                Pointer.to(d_input),
                Pointer.to(d_output),
                Pointer.to(new int[]{n})
        );

        cuLaunchKernel(kernelFunction,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                sharedMemSize, null,
                kernelParams, null
        );

        cudaMemcpy(Pointer.to(output), d_output, (long) n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return output;
    }

    @Override
    protected float[] applyCPU(float[] input) {
        float sumExp = 0.0f;
        for (float v : input) {
            sumExp += Math.exp(v);
        }
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i]) / sumExp;
        }
        return output;
    }
}
