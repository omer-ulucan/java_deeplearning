package activations;

public class Tanh {
    public static double tanh(double x){
        double expX = Math.exp(x);
        double expNegX = Math.exp(-x);
        return (expX - expNegX)/(expX + expNegX);
    }
}
