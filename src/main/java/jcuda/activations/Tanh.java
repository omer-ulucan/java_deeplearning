package jcuda.activations;

public class Tanh extends BaseActivation {
    private static final String KERNEL_SOURCE =
            "__global__ void tanhKernel(float *input, float *output, int n) {" +
                    "    int i = threadIdx.x + blockIdx.x * blockDim.x;" +
                    "    if (i < n) {" +
                    "        float e_x = expf(input[i]);" +
                    "        float e_neg_x = expf(-input[i]);" +
                    "        output[i] = (e_x - e_neg_x) / (e_x + e_neg_x);" +
                    "    }" +
                    "}";
    private static final String KERNEL_NAME = "tanhKernel";

    public Tanh() {
        super(KERNEL_SOURCE, KERNEL_NAME);
    }

    @Override
    protected float[] applyCPU(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            float e_x = (float) Math.exp(input[i]);
            float e_neg_x = (float) Math.exp(-input[i]);
            output[i] = (e_x - e_neg_x) / (e_x + e_neg_x);
        }
        return output;
    }
}
