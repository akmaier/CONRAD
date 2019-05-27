package edu.stanford.rsl.science.iterativedesignstudy;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;

public class SimpleCLGridExample {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();

		new ImageJ();

		Phantom shepp = new SheppLogan(2000);

		shepp.show("CPU Memory");

		OpenCLGrid2D sheppCL = new OpenCLGrid2D(shepp,context,device);
		

		long timeCPU = System.nanoTime();
		for (int i = 0; i < 10; i++){
			NumericPointwiseOperators.addBy(shepp, shepp);
		}
		timeCPU = System.nanoTime() - timeCPU;
		
		long timeGPU = System.nanoTime();
		for (int i = 0; i < 10; i++){
			NumericPointwiseOperators.addBy(sheppCL, sheppCL);
		}
		timeGPU = System.nanoTime() - timeGPU;
		
		System.out.println("GPU :" + timeGPU/1000);
		System.out.println("CPU :" + timeCPU/1000);
		
		sheppCL.show("CL Memory");


	}

}
