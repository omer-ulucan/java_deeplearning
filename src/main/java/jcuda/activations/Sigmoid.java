package jcuda.activations;

public class Sigmoid extends BaseActivation {
    private static final String KERNEL_SOURCE =
            "__global__ void sigmoidKernel(float *input, float *output, int n) {" +
                    "    int i = threadIdx.x + blockIdx.x * blockDim.x;" +
                    "    if (i < n) {" +
                    "        output[i] = 1.0f / (1.0f + expf(-input[i]));" +
                    "    }" +
                    "}";
    private static final String KERNEL_NAME = "sigmoidKernel";

    public Sigmoid() {
        super(KERNEL_SOURCE, KERNEL_NAME);
    }

    @Override
    protected float[] applyCPU(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
        }
        return output;
    }
}
