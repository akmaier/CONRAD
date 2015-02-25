package TruncationPolynomial;

import ij.ImageJ;
import ij.macro.Program;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Queue;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class FirstOpenCL extends Grid2D {

	public FirstOpenCL(int width, int height) {
		super(width, height);
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		//		Phantom phantom = new Phantom(8,8);
		//		
		//		OpenCLGrid2D clGrid = new OpenCLGrid2D(phantom);
		//		
		//		long GPUbefore = System.currentTimeMillis();
		//		OpenCLGrid2D addedGPUclGrid = addOneMillion(clGrid);
		//		long GPUafter = System.currentTimeMillis();
		//		long GPUtime = GPUafter-GPUbefore;
		//		System.out.println(GPUtime);
		//		
		//		Grid2D addedGPU = new Grid2D(addedGPUclGrid);
		//		addedGPU.show();
		//				
		//		long CPUbefore = System.currentTimeMillis();
		//		Phantom addedCPU = addOneMillion(phantom);
		//		long CPUafter = System.currentTimeMillis();
		//		long CPUtime = CPUafter-CPUbefore;
		//		System.out.println(CPUtime);
		//		addedCPU.show();

//		new ImageJ();
		int phantomSize = 256;
		Phantom phantom1 = new Phantom(phantomSize, phantomSize, true);
		//		phantom1.show();
		Phantom phantom2 = new Phantom(phantomSize, phantomSize, false);
		//		phantom2.show();

		//		OpenCLGrid2D additionresult = addPhantoms(phantom1, phantom2);
		//		additionresult.show();

		Phantom sinogram = Phantom.getSinogram(phantom1);
		Phantom ramlacSinogram = BackProjection.ramlac(sinogram);
		long GPUtimes = 0;
		long CPUtimes = 0;

		for (int i = 0; i < 500; i++) {

			System.out.print(i + ": ");
			
			long GPUbefore = System.currentTimeMillis();
			OpenCLGrid2D backprojection = parallelBackprojection(ramlacSinogram);
			long GPUafter = System.currentTimeMillis();
			long GPUtime = GPUafter-GPUbefore;
			System.out.print("GPU: "+ GPUtime + "; ");
			if(i!=0) {
			GPUtimes += GPUtime;
			//		backprojection.show();
			}
			long CPUbefore = System.currentTimeMillis();
			Phantom reconstructedCPU = BackProjection.backproject(ramlacSinogram);
			long CPUafter = System.currentTimeMillis();
			long CPUtime = CPUafter-CPUbefore;
			System.out.println("CPU: " + CPUtime);
			if (i != 0) {
			CPUtimes += CPUtime;
			//		reconstructedCPU.show();
		
			}
		}

		GPUtimes /= 499;
		System.out.println("GPUmean: " + GPUtimes);
		CPUtimes /= 499;
		System.out.println("CPUmean: " + CPUtimes);

		//		NumericPointwiseOperators.subtractedBy(reconstructedCPU, backprojection).show();

	}

	private static OpenCLGrid2D addPhantoms(Phantom phantom1, Phantom phantom2){

		OpenCLGrid2D phantomCL1 = new OpenCLGrid2D(phantom1);
		CLBuffer<FloatBuffer> bufferPhantomCL1 = phantomCL1.getDelegate().getCLBuffer();

		OpenCLGrid2D phantomCL2 = new OpenCLGrid2D(phantom2);
		CLBuffer<FloatBuffer> bufferPhantomCL2 = phantomCL2.getDelegate().getCLBuffer();

		phantomCL1.getDelegate().prepareForDeviceOperation();
		phantomCL2.getDelegate().prepareForDeviceOperation();

		// create Context
		CLContext context = OpenCLUtil.getStaticContext();
		// choose fastest device
		CLDevice device = context.getMaxFlopsDevice();

		//		CLContext context = phantomCL1.getDelegate().getCLContext();
		//		
		//		CLDevice device = phantomCL1.getDelegate().getCLDevice();

		CLProgram program = null;
		try {
			program = context.createProgram(FirstOpenCL.class.getResourceAsStream("addition.cl"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		program.build();

		CLKernel kernel = program.createCLKernel("add");
		kernel.putArg(bufferPhantomCL1);
		kernel.putArg(bufferPhantomCL2);
		kernel.putArg(phantom1.getWidth() * phantom1.getHeight());

		int localWorksize = 128;
		long globalWorksize = OpenCLUtil.roundUp(localWorksize, phantom1.getWidth()*phantom1.getHeight());

		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put1DRangeKernel(kernel, 0, globalWorksize, localWorksize).finish();

		phantomCL1.getDelegate().notifyDeviceChange();
		phantomCL2.getDelegate().notifyDeviceChange();

		return phantomCL1;
	}

	private static OpenCLGrid2D addOneMillion(OpenCLGrid2D clGrid) {

		OpenCLGrid2D result = new OpenCLGrid2D(clGrid);

		for (int i = 0; i < 1000000; i++) {
			NumericPointwiseOperators.addBy(result, clGrid);
			//			operator.addBy(result, clGrid);
		}



		return result;
	}

	private static Phantom addOneMillion(Phantom phantom) {
		Phantom result = new Phantom(phantom.getWidth(), phantom.getHeight());
		for (int i = 1; i < 1000000; i++) {
			NumericPointwiseOperators.addBy(result, phantom);
		}
		return result;
	}

	public static OpenCLGrid2D parallelBackprojection(Phantom sinogram) {

		OpenCLGrid2D sinogramCL = new OpenCLGrid2D(sinogram);
		int a = Math.round((float)(sinogram.getWidth()/Math.sqrt(2)));
		Grid2D out = new Grid2D(a,a);
		OpenCLGrid2D outCL = new OpenCLGrid2D(out);


		CLBuffer<FloatBuffer> bufferSinogram = sinogramCL.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferOut = outCL.getDelegate().getCLBuffer();

		sinogramCL.getDelegate().prepareForDeviceOperation();
		outCL.getDelegate().prepareForDeviceOperation();

		// create Context
		CLContext context = OpenCLUtil.getStaticContext();
		// choose fastest device
		CLDevice device = context.getMaxFlopsDevice();

		CLProgram program = null;
		try {
			program = context.createProgram(FirstOpenCL.class.getResourceAsStream("backprojection.cl"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		program.build();

		CLKernel kernel = program.createCLKernel("backproject");
		kernel.putArg(bufferSinogram);
		kernel.putArg(bufferOut);
		kernel.putArg(out.getWidth());
		kernel.putArg(out.getHeight());
		kernel.putArg(sinogram.getWidth());


		int localWorksize = 16;
		long globalWorksizeX = OpenCLUtil.roundUp(localWorksize, out.getWidth());
		long globalWorksizeY = OpenCLUtil.roundUp(localWorksize, out.getHeight());

		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeX, globalWorksizeY, localWorksize, localWorksize).finish();

		sinogramCL.getDelegate().notifyDeviceChange();
		outCL.getDelegate().notifyDeviceChange();


		return outCL;
	}


}
