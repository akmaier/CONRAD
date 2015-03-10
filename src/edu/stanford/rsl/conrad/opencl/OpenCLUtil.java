/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.opencl;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLCommandQueue.Mode;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLMemory.Mem;












import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cone;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Pyramid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLBox;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLCompoundShape;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLCone;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLCylinder;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLEllipsoid;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLPyramid;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLSphere;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLTextureTimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformTextureSurfaceBSpline;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildBox;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildCone;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildCylinder;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildEllipsoid;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildSphere;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.jpop.utils.UserUtil;

public abstract class OpenCLUtil {

	public static CLProgram program;
	public static CLProgram render;
	public static CLProgram simpleObjects;
	public static CLProgram yxdraw;
	public static CLProgram appendBuffer;
	public static CLContext staticContext;
	public static CLCommandQueue staticCommandQueue;
	public static CLProgram filter;
	
	public static boolean isOpenCLConfigured(){
		return Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.OPENCL_DEVICE_SELECTION) != null;
	}
	
	public static void releaseContext (CLContext context){
		program = null;
		render = null;
		simpleObjects = null;
		yxdraw = null;
		appendBuffer = null;
		staticContext = null;
		filter = null;
		context.release();
	}
	
	public synchronized static CLContext getStaticContext (){
		if (staticContext == null) {
			staticContext = createContext();
		}
		return staticContext;
	}
	
	public synchronized static CLCommandQueue getStaticCommandQueue (){
		if(staticContext == null)
			staticContext = createContext();
		if(staticCommandQueue == null || staticCommandQueue.isReleased())
			staticCommandQueue = staticContext.getDevices()[0].createCommandQueue();
		return staticCommandQueue;
	}
	
	public synchronized static void releaseStaticContext (){
		if (staticCommandQueue != null && !staticCommandQueue.isReleased()) {
			staticCommandQueue.release();
		}
		if (staticContext != null && !staticContext.isReleased()) {
			staticContext.release();
		}
		staticContext = null;
		staticCommandQueue = null;
	}
	
	public static CLBuffer<FloatBuffer> generateSamplingPoints(int elementCountU, int elementCountV, CLContext context, CLDevice device){
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*2, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
				samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}
	
	//public static CLContext context;


	//public static CLContext getContext(){
	//	return context;
	//}
	/**
	 * Creates the CLContext for the device that is preconfigured in CONRAD.
	 * @return the CLContext
	 * @see edu.stanford.rsl.conrad.utils.RegKeys#OPENCL_DEVICE_SELECTION
	 */
	public synchronized static CLContext createContext(){
		CLContext context = null;
		if( Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}

		String devString = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.OPENCL_DEVICE_SELECTION);
		if (devString != null){
			CLPlatform[] platforms = CLPlatform.listCLPlatforms();
			CLDevice[] deviceList = platforms[0].listCLDevices();
			for (int i = 1; i < platforms.length; i++) {
				CLDevice[] temp = platforms[i].listCLDevices();
				int N = deviceList.length; 
				deviceList = Arrays.copyOf(deviceList, N + temp.length);
				for (int j = 0; j < temp.length; j++)
					deviceList[N+j] = temp[j];
			}
			CLDevice device = null;
			for (CLDevice selection: deviceList){
				String test = selection.getName()+" "+selection.getVendor();
				if (test.equals(devString)){
					device = selection;
					break;
				}
			}
			context = CLContext.create(device);				
		} else {

			/*try {
					context = CLContext.create();
				} catch (CLException e) {*/
			CLPlatform[] platforms = CLPlatform.listCLPlatforms();
			CLDevice[] deviceList = platforms[0].listCLDevices();
			for (int i = 1; i < platforms.length; i++) {
				CLDevice[] temp = platforms[i].listCLDevices();
				int N = deviceList.length; 
				deviceList = Arrays.copyOf(deviceList, N + temp.length);
				for (int j = 0; j < temp.length; j++)
					deviceList[N+j] = temp[j];
			}

			CLDevice device = deviceList[0];
			try {
				device = (CLDevice) UserUtil.chooseObject("Choose OpenCL device", "OpenCL device selection", deviceList, deviceList[0]);
				Configuration.getGlobalConfiguration().getRegistry().put(RegKeys.OPENCL_DEVICE_SELECTION, device.getName()+" "+device.getVendor());
				Configuration.saveConfiguration();
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			context = CLContext.create(device);
			//}
		}
		return context;
	}

	public synchronized static void initYXDraw(CLContext context){
		if (yxdraw==null){
			try {
				yxdraw = context.createProgram(TestOpenCL.class.getResourceAsStream("yxdraw.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!yxdraw.getContext().equals(context)){
				try {
					yxdraw = context.createProgram(TestOpenCL.class.getResourceAsStream("yxdraw.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	
	public synchronized static void initFilter(CLContext context){
		if (filter==null){
			try {
				filter = context.createProgram(TestOpenCL.class.getResourceAsStream("filter.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!filter.getContext().equals(context)){
				try {
					filter = context.createProgram(TestOpenCL.class.getResourceAsStream("filter.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public static CLProgram getYXDrawInstance(){
		return yxdraw;
	}
	
	public static OpenCLEvaluatable getOpenCLEvaluatableSubclass(AbstractShape s, CLDevice device){
		if (s instanceof Cylinder || s instanceof ForbildCylinder)
			return new OpenCLCylinder((Cylinder) s, device);
		else if (s instanceof Box  || s instanceof ForbildBox)
			return new OpenCLBox((Box)s, device);
		else if (s instanceof Sphere || s instanceof ForbildSphere)
			return new OpenCLSphere((Sphere) s, device);
		else if (s instanceof Cone  || s instanceof ForbildCone)
			return new OpenCLCone((Cone)s, device);
		else if (s instanceof Ellipsoid  || s instanceof ForbildEllipsoid)
			return new OpenCLEllipsoid((Ellipsoid)s, device);
		else if (s instanceof Pyramid)
			return new OpenCLPyramid((Pyramid)s, device);
		else if (s instanceof BSpline)
			return new OpenCLUniformBSpline((BSpline) s, device);
		else if (s instanceof SurfaceBSpline)
			return new OpenCLUniformTextureSurfaceBSpline((SurfaceBSpline) s, device);
		else if (s instanceof TimeVariantSurfaceBSpline)
			return new OpenCLTextureTimeVariantSurfaceBSpline((TimeVariantSurfaceBSpline) s, device);
		else if (s instanceof CompoundShape)
			return new OpenCLCompoundShape((CompoundShape) s, device);
		else
			throw new RuntimeException(s.getName() + " --> This shape is not yet implemented.");
	}

	public synchronized static void initRender(CLContext context){
		if (render==null){
			try {
				render = context.createProgram(TestOpenCL.class.getResourceAsStream("projection.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!render.getContext().equals(context)){
				try {
					render = context.createProgram(TestOpenCL.class.getResourceAsStream("projection.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	
	public synchronized static void initSimpleObjectEvaluator(CLContext context){
		if (simpleObjects==null){
			try {
				simpleObjects = context.createProgram(TestOpenCL.class.getResourceAsStream("simpleObjects.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!simpleObjects.getContext().equals(context)){
				try {
					simpleObjects = context.createProgram(TestOpenCL.class.getResourceAsStream("simpleObjects.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public static CLProgram getRenderInstance(){
		return render;
	}

	public synchronized static void initAppendBufferRender(CLContext context){
		if (appendBuffer==null){
			try {
				appendBuffer = context.createProgram(TestOpenCL.class.getResourceAsStream("appendBuffer.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!appendBuffer.getContext().equals(context)){
				try {
					appendBuffer = context.createProgram(TestOpenCL.class.getResourceAsStream("appendBuffer.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public synchronized static void initTriangleAppendBufferRender(CLContext context){
		if (appendBuffer==null){
			try {
				appendBuffer = context.createProgram(TestOpenCL.class.getResourceAsStream("triangleAppendBuffer.cl")).build(); // CLProgram.CompilerOptions.DISABLE_OPT
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!appendBuffer.getContext().equals(context)){
				try {
					appendBuffer = context.createProgram(TestOpenCL.class.getResourceAsStream("triangleAppendBuffer.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	
//	public synchronized static void initSimpleObjectEvaluator(CLContext context){
//		if (simpleObjects==null){
//			try {
//				simpleObjects = context.createProgram(TestOpenCL.class.getResourceAsStream("simpleObjects.cl")).build();
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		} else {
//			if (!simpleObjects.getContext().equals(context)){
//				try {
//					simpleObjects = context.createProgram(TestOpenCL.class.getResourceAsStream("simpleObjects.cl")).build();
//				} catch (IOException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//			}
//		}
//	}


	public static CLProgram getAppendBufferRenderInstance(){
		return appendBuffer;
	}
	
	
	public static CLBuffer<FloatBuffer> copyMatrixToDevice(SimpleMatrix m, CLContext context, CLDevice device){
		CLBuffer<FloatBuffer> pMatrix = context.createFloatBuffer((m.getCols()*m.getRows()), Mem.READ_ONLY);
		pMatrix.getBuffer().clear();
		for (int i = 0; i < m.getRows(); i++){
			for (int j=0; j<m.getCols(); j++){
				pMatrix.getBuffer().put((float)m.getElement(i,j));
			}
		}
		pMatrix.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(pMatrix, false).finish();
		clc.release();
		return pMatrix;
	}
		
	public static void transformPoints(CLBuffer<FloatBuffer> outputBuffer, SimpleMatrix matrix, CLContext context, CLDevice device){
		int elementCount = outputBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLBuffer<FloatBuffer> clmatrix = copyMatrixToDevice(matrix, context, device);
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("applyTransform");
		
		kernel.putArgs(outputBuffer)
		.putArg(clmatrix).putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clmatrix.release();
		kernel.release();
		clc.release();
	}

	public synchronized static void initProgram(CLContext context){
		if (program == null){
			try {
				program = context.createProgram(TestOpenCL.class.getResourceAsStream("bspline.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			if (!program.getContext().equals(context)){
				try {
					program = context.createProgram(TestOpenCL.class.getResourceAsStream("bspline.cl")).build();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	public static CLProgram getProgramInstance(){
		return program;
	}

	/**
	 * rounded up to the nearest multiple of the groupSize
	 * @param groupSize
	 * @param globalSize
	 * @return the rounded value
	 */
	public static int roundUp(int groupSize, int globalSize) {
		int r = globalSize % groupSize;
		if (r == 0) {
			return globalSize;
		} else {
			return globalSize + groupSize - r;
		}
	}


	/**
	 * Integral division, rounding the result to the next highest integer.
	 * 
	 * @param a Dividend
	 * @param b Divisor
	 * @return a/b rounded to the next highest integer.
	 */
	public static long iDivUp(long a, long b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}


	/**
	 * Integral division, rounding the result to the next highest integer.
	 * 
	 * @param a Dividend
	 * @param b Divisor
	 * @return a/b rounded to the next highest integer.
	 */
	public static int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

}
