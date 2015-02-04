/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import java.io.InputStream;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.opencl.OpenCLGenericGridInterface;
import edu.stanford.rsl.conrad.data.generic.opencl.OpenCLGenericGridOperators;

public class OpenCLComplexGridOperator extends OpenCLGenericGridOperators<Complex> implements ComplexGridOperatorInterface{
	
	static OpenCLComplexGridOperator op = new OpenCLComplexGridOperator();

	public static OpenCLComplexGridOperator getInstance() {
		return op;
	}
	
	@Override
	protected InputStream getProgramFileAsStream() {
		return OpenCLComplexGridOperator.class.getResourceAsStream("PointwiseOperators.cl");
	}

	public void conj(final ComplexGrid grid) {
		if (debug) System.out.println("Bei OpenCL conj");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<Complex> clGrid = (OpenCLGenericGridInterface<Complex>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runUnaryKernelNoReturn("conj", device, clmem, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}

	
	@Override
	public void abs(final ComplexGrid grid) {
		if (debug) System.out.println("Bei OpenCL abs");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<Complex> clGrid = (OpenCLGenericGridInterface<Complex>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runUnaryKernelNoReturn("absolute", device, clmem, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public Complex getSum(CLBuffer<FloatBuffer> clRes) {
		Complex res = new Complex();
		int n = clRes.getCLCapacity()/2;
		for (int i = 0; i < n; i++) {
			res = res.add(new Complex(clRes.getBuffer().get(i*2),clRes.getBuffer().get(i*2+1)));
		}
		return res;
	}

	@Override
	public Complex getMin(CLBuffer<FloatBuffer> clRes) {
		Complex res = new Complex(Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY);
		int n = clRes.getCLCapacity()/2;
		for (int i = 0; i < n; i++) {
			Complex val = new Complex(clRes.getBuffer().get(i*2),clRes.getBuffer().get(i*2+1));
			res = (val.compareTo(res) < 0) ? val : res;
		}
		return res;
	}

	@Override
	public Complex getMax(CLBuffer<FloatBuffer> clRes) {
		Complex res = new Complex(0,0);
		int n = clRes.getCLCapacity()/2;
		for (int i = 0; i < n; i++) {
			Complex val = new Complex(clRes.getBuffer().get(i*2),clRes.getBuffer().get(i*2+1));
			res = (val.compareTo(res) > 0) ? val : res;
		}
		return res;
	}
}
