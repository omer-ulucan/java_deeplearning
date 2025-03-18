package jcuda.activations;

public class Softmax {
    public static double[] softmax(double[] z) {
        double[] expValues = new double[z.length];
        double sumExp = 0.0;
        for (int i = 0; i < z.length; i++) {
            expValues[i] = Math.exp(z[i]);
            sumExp += expValues[i];
        }
        double[] softmaxValues = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            softmaxValues[i] = expValues[i] / sumExp;
        }

        return softmaxValues;
    }
}
