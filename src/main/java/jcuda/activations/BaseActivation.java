package jcuda.activations;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.core.JCudaManager;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public abstract class BaseActivation {
    protected CUfunction kernelFunction;

    /**
     * Bu yapıcı, verilen kernel kaynak kodunu yükler ve kernel fonksiyonunu elde eder.
     * @param kernelSource Kernel kaynak kodu
     * @param kernelName Kernel içindeki fonksiyon ismi
     */
    protected BaseActivation(String kernelSource, String kernelName) {
        // JCuda hata ayıklamayı aktif ediyor ve GPU'yu başlatıyoruz
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);

        CUmodule module = new CUmodule();
        cuModuleLoadData(module, kernelSource);

        kernelFunction = new CUfunction();
        cuModuleGetFunction(kernelFunction, module, kernelName);
    }

    /**
     * Giriş verisine bağlı olarak GPU ya da CPU üzerinde hesaplama yapar.
     * @param input Giriş dizisi
     * @return Hesaplama sonucu
     */
    public float[] apply(float[] input) {
        if (JCudaManager.isGpuAvailable()) {
            return applyGPU(input);
        } else {
            return applyCPU(input);
        }
    }

    /**
     * Ortak GPU işlemleri. Eğer özel bir kernel için farklılık gerekiyorsa alt sınıfta override edilebilir.
     */
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

        Pointer kernelParams = Pointer.to(
                Pointer.to(d_input),
                Pointer.to(d_output),
                Pointer.to(new int[]{n})
        );

        cuLaunchKernel(kernelFunction,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                0, null,
                kernelParams, null
        );

        cudaMemcpy(Pointer.to(output), d_output, (long) n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return output;
    }

    /**
     * Her aktivasyonun CPU versiyonunu kendine özgü olarak tanımlaması gerekir.
     */
    protected abstract float[] applyCPU(float[] input);
}
