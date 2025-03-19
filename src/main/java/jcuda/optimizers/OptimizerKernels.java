package jcuda.optimizers;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;

public class OptimizerKernels {
    public static CUfunction sgdUpdateFunction;
    public static CUfunction momentumUpdateFunction;
    public static CUfunction nesterovUpdateFunction;
    public static CUfunction adagradUpdateFunction;
    public static CUfunction rmspropUpdateFunction;
    public static CUfunction adamUpdateFunction;
    public static CUfunction adamWUpdateFunction;

    static {
        // Kernel source code for all optimizers
        String kernelSource =
                // SGD kernel: param[i] -= learningRate * grad[i];
                "extern \"C\" __global__ void sgdUpdateKernel(float* param, float* grad, float learningRate, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         param[i] -= learningRate * grad[i]; " +
                        "    } " +
                        "}" +
                        // Momentum kernel:
                        // velocity[i] = momentum * velocity[i] - learningRate * grad[i];
                        // param[i] += velocity[i];
                        "extern \"C\" __global__ void momentumUpdateKernel(float* param, float* grad, float* velocity, float learningRate, float momentum, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         velocity[i] = momentum * velocity[i] - learningRate * grad[i]; " +
                        "         param[i] += velocity[i]; " +
                        "    } " +
                        "}" +
                        // Nesterov kernel:
                        // float prev = velocity[i];
                        // velocity[i] = momentum * velocity[i] - learningRate * grad[i];
                        // param[i] += -momentum * prev + (1 + momentum) * velocity[i];
                        "extern \"C\" __global__ void nesterovUpdateKernel(float* param, float* grad, float* velocity, float learningRate, float momentum, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         float prev = velocity[i]; " +
                        "         velocity[i] = momentum * velocity[i] - learningRate * grad[i]; " +
                        "         param[i] += -momentum * prev + (1 + momentum) * velocity[i]; " +
                        "    } " +
                        "}" +
                        // Adagrad kernel:
                        // sumSquares[i] += grad[i]^2; param[i] -= learningRate * grad[i] / (sqrt(sumSquares[i]) + epsilon);
                        "extern \"C\" __global__ void adagradUpdateKernel(float* param, float* grad, float* sumSquares, float learningRate, float epsilon, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         sumSquares[i] += grad[i] * grad[i]; " +
                        "         param[i] -= learningRate * grad[i] / (sqrtf(sumSquares[i]) + epsilon); " +
                        "    } " +
                        "}" +
                        // RMSProp kernel:
                        // cache[i] = beta * cache[i] + (1 - beta) * grad[i]^2;
                        // param[i] -= learningRate * grad[i] / (sqrt(cache[i]) + epsilon);
                        "extern \"C\" __global__ void rmspropUpdateKernel(float* param, float* grad, float* cache, float learningRate, float beta, float epsilon, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         cache[i] = beta * cache[i] + (1 - beta) * grad[i] * grad[i]; " +
                        "         param[i] -= learningRate * grad[i] / (sqrtf(cache[i]) + epsilon); " +
                        "    } " +
                        "}" +
                        // Adam kernel:
                        // m[i] = beta1*m[i] + (1-beta1)*grad[i];
                        // v[i] = beta2*v[i] + (1-beta2)*grad[i]^2;
                        // param[i] -= learningRate * (m[i]/(1-pow(beta1,t))) / (sqrt(v[i]/(1-pow(beta2,t))) + epsilon);
                        "extern \"C\" __global__ void adamUpdateKernel(float* param, float* grad, float* m, float* v, float learningRate, float beta1, float beta2, float epsilon, int t, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         m[i] = beta1 * m[i] + (1 - beta1) * grad[i]; " +
                        "         v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]; " +
                        "         float mHat = m[i] / (1 - powf(beta1, t)); " +
                        "         float vHat = v[i] / (1 - powf(beta2, t)); " +
                        "         param[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon); " +
                        "    } " +
                        "}" +
                        // AdamW kernel:
                        // m[i] and v[i] are updated as in Adam.
                        // param[i] -= learningRate * (mHat/(sqrt(vHat)+epsilon) + weightDecay * param[i]);
                        "extern \"C\" __global__ void adamWUpdateKernel(float* param, float* grad, float* m, float* v, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int t, int N) { " +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " +
                        "    if (i < N) { " +
                        "         m[i] = beta1 * m[i] + (1 - beta1) * grad[i]; " +
                        "         v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]; " +
                        "         float mHat = m[i] / (1 - powf(beta1, t)); " +
                        "         float vHat = v[i] / (1 - powf(beta2, t)); " +
                        "         param[i] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[i]); " +
                        "    } " +
                        "}";

        CUmodule module = new CUmodule();
        cuModuleLoadData(module, kernelSource);

        sgdUpdateFunction = new CUfunction();
        cuModuleGetFunction(sgdUpdateFunction, module, "sgdUpdateKernel");

        momentumUpdateFunction = new CUfunction();
        cuModuleGetFunction(momentumUpdateFunction, module, "momentumUpdateKernel");

        nesterovUpdateFunction = new CUfunction();
        cuModuleGetFunction(nesterovUpdateFunction, module, "nesterovUpdateKernel");

        adagradUpdateFunction = new CUfunction();
        cuModuleGetFunction(adagradUpdateFunction, module, "adagradUpdateKernel");

        rmspropUpdateFunction = new CUfunction();
        cuModuleGetFunction(rmspropUpdateFunction, module, "rmspropUpdateKernel");

        adamUpdateFunction = new CUfunction();
        cuModuleGetFunction(adamUpdateFunction, module, "adamUpdateKernel");

        adamWUpdateFunction = new CUfunction();
        cuModuleGetFunction(adamWUpdateFunction, module, "adamWUpdateKernel");
    }
}
