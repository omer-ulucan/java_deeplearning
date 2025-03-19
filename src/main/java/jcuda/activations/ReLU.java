package jcuda.activations;

public class ReLU extends BaseActivation {
    private static final String KERNEL_SOURCE =
            "__global__ void reluKernel(float *input, float *output, int n) {" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" +
                    "    if (i < n) {" +
                    "        output[i] = fmaxf(0.0f, input[i]);" +
                    "    }" +
                    "}";
    private static final String KERNEL_NAME = "reluKernel";

    public ReLU() {
        super(KERNEL_SOURCE, KERNEL_NAME);
    }

    @Override
    protected float[] applyCPU(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
}
