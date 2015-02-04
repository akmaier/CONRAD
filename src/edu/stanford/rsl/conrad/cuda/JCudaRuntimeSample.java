package edu.stanford.rsl.conrad.cuda;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * DISCLAIMER: THIS SOFTWARE IS PROVIDED WITHOUT WARRANTY OF ANY KIND
 * If you find any bugs or errors, contact me at http://www.jcuda.org
 *
 * LICENSE: THIS SOFTWARE IS FREE FOR NON-COMMERCIAL USE ONLY
 * For non-commercial applications, you may use this software without
 * any restrictions. If you wish to use it for commercial purposes,
 * contact me at http://www.jcuda.org
 */

import java.util.*;

import jcuda.*;
import jcuda.jcublas.*;
import jcuda.jcudpp.*;
import jcuda.jcufft.*;
import jcuda.runtime.*;

/**
 * This is a class that demonstrates the interoperability among
 * JCuda, JCufft, JCublas and JCudpp. It performs several 
 * computations using each library, using always the same
 * "shared" device memory.
 */
public class JCudaRuntimeSample
{
    public static void main(String args[])
    {
        System.out.println("Creating input data");
        
     // the classpath
        System.out.println( System.getProperty( "java.class.path" ) );

        // extension directories whose jars are included on the classpath
        System.out.println( System.getProperty( "java.ext.dirs" ) );

        // low level classpath, includes system jars
        System.out.println( System.getProperty( "java.library.path" ) );

        // character to separate (not terminate!) entries on the classpath, ; for Windows : for unix.
        System.out.println( System.getProperty( "path.separator" ) );


        // Create some input data
        int complexElements = 100;
        int floatElements = complexElements * 2;
        int memorySize = floatElements * Sizeof.FLOAT;
        float hostX[] = createRandomFloatData(floatElements);
        float hostY[] = createRandomFloatData(floatElements);

        
        System.out.println("Initializing device data using JCuda");
        
        // Allocate memory on the device using JCuda
        Pointer deviceX = new Pointer();
        Pointer deviceY = new Pointer();
        JCuda.cudaMalloc(deviceX, memorySize);
        JCuda.cudaMalloc(deviceY, memorySize);
        
        // Copy memory from host to device using JCuda
        JCuda.cudaMemcpy(deviceX, Pointer.to(hostX), memorySize, 
            cudaMemcpyKind.cudaMemcpyHostToDevice);
        JCuda.cudaMemcpy(deviceY, Pointer.to(hostY), memorySize, 
            cudaMemcpyKind.cudaMemcpyHostToDevice);
        

        System.out.println("Performing FFT using JCufft");
        
        // Perform in-place complex-to-complex 1D transforms using JCufft
        cufftHandle plan = new cufftHandle();
        JCufft.cufftPlan1d(plan, complexElements, cufftType.CUFFT_C2C, 1);
        JCufft.cufftExecC2C(plan, deviceX, deviceX, JCufft.CUFFT_FORWARD);
        JCufft.cufftExecC2C(plan, deviceY, deviceY, JCufft.CUFFT_FORWARD);


        System.out.println("Performing caxpy using JCublas");
        
        // Perform a complex y=a*x+y operation (caxpy) using JCublas
        cuComplex alpha = cuComplex.cuCmplx(0.3f, 0.7f);
        JCublas.cublasInit();
        JCublas.cublasCaxpy(complexElements, alpha, deviceX, 1, deviceY, 1);
        

        // This is a sample application, so perform a scan of one 
        // of the complex vectors using JCudpp, although this does 
        // not make any sense...
        
        System.out.println("Performing scan using JCudpp");
        
        // Create a configuration that describes a scan
        CUDPPConfiguration config = new CUDPPConfiguration();
        config.op = CUDPPOperator.CUDPP_ADD;
        config.datatype = CUDPPDatatype.CUDPP_FLOAT;
        config.algorithm = CUDPPAlgorithm.CUDPP_SCAN;
        config.options = CUDPPOption.CUDPP_OPTION_FORWARD;
        
        // Create a CUDPPHandle for the scan operation
        CUDPPHandle handle = new CUDPPHandle();
        JCudpp.cudppPlan(handle, config, complexElements, 1, 0);  

        // Run the scan
        JCudpp.cudppScan(handle, deviceX, deviceY, floatElements);
    
        
        // Copy the result from the device to the host
        JCuda.cudaMemcpy(Pointer.to(hostX), deviceX, memorySize, 
            cudaMemcpyKind.cudaMemcpyDeviceToHost);
        
        System.out.println("Result: "+hostX[hostX.length-1]);
        
        // Clean up
        JCuda.cudaFree(deviceX);
        JCuda.cudaFree(deviceY);
        JCublas.cublasShutdown();
        JCufft.cufftDestroy(plan);
        JCudpp.cudppDestroyPlan(handle);
    }

    /**
     * Creates an array of the specified size, containing some random data
     */
    private static float[] createRandomFloatData(int x)
    {
        Random random = new Random(0);
        float a[] = new float[x];
        for (int i=0; i<x; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }
}

