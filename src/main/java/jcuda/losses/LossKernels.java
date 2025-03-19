package jcuda.losses;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;

public class LossKernels {
    public static CUfunction mseLossForwardFunction;
    public static CUfunction mseLossBackwardFunction;
    public static CUfunction crossEntropyLossForwardFunction;
    public static CUfunction crossEntropyLossBackwardFunction;

    static {
        String kernelSource =
                // MSE forward kernel: for each element, compute (pred - target)^2 and store in out.
                "extern \"C\" __global__ void mseLossForwardKernel(float* pred, float* target, float* out, int N) {" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" +
                        "    if (i < N) {" +
                        "        float diff = pred[i] - target[i];" +
                        "        out[i] = diff * diff;" +
                        "    }" +
                        "}" +
                        // MSE backward kernel: for each element, compute 2*(pred - target)/N.
                        "extern \"C\" __global__ void mseLossBackwardKernel(float* pred, float* target, float* grad, int N) {" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" +
                        "    if (i < N) {" +
                        "        grad[i] = 2.0f * (pred[i] - target[i]) / N;" +
                        "    }" +
                        "}" +
                        // Cross-Entropy forward kernel: for each element, compute -target * log(pred + epsilon)
                        "extern \"C\" __global__ void crossEntropyLossForwardKernel(float* pred, float* target, float* out, int N, float epsilon) {" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" +
                        "    if (i < N) {" +
                        "        out[i] = -target[i] * logf(pred[i] + epsilon);" +
                        "    }" +
                        "}" +
                        // Cross-Entropy backward kernel: for each element, compute -(target/(pred+epsilon))/N.
                        "extern \"C\" __global__ void crossEntropyLossBackwardKernel(float* pred, float* target, float* grad, int N, float epsilon) {" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" +
                        "    if (i < N) {" +
                        "        grad[i] = -target[i] / (pred[i] + epsilon) / N;" +
                        "    }" +
                        "}";

        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoadData(module, kernelSource);

        mseLossForwardFunction = new CUfunction();
        cuModuleGetFunction(mseLossForwardFunction, module, "mseLossForwardKernel");

        mseLossBackwardFunction = new CUfunction();
        cuModuleGetFunction(mseLossBackwardFunction, module, "mseLossBackwardKernel");

        crossEntropyLossForwardFunction = new CUfunction();
        cuModuleGetFunction(crossEntropyLossForwardFunction, module, "crossEntropyLossForwardKernel");

        crossEntropyLossBackwardFunction = new CUfunction();
        cuModuleGetFunction(crossEntropyLossBackwardFunction, module, "crossEntropyLossBackwardKernel");
    }
}
