/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.conrad.geometry.motion;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

/**
 * Parzen-interpolating motion field. Source points are generated from an ArrayList of TimeVariantSurfaceBSplines.
 * @author akmaier
 *
 */
public class TimeVariantSurfaceBSplineListMotionField extends
ParzenWindowMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7397812496219257524L;
	private ArrayList<TimeVariantSurfaceBSpline> variants = null;
	
	public TimeVariantSurfaceBSplineListMotionField(
			ArrayList<TimeVariantSurfaceBSpline> variants, double sigma) {
		super(sigma);
		this.variants = variants;
	}
	
	@Override
	PointND[] getRasterPoints(double time) {
		PointND[] result = timePointMap.get(time);
		if (result == null){
			if (OpenCLUtil.isOpenCLConfigured()){
				CLContext context = OpenCLUtil.getStaticContext();
				CLDevice device = context.getMaxFlopsDevice();
				double sampleTime = getTimeWarper().warpTime(time);
				ArrayList<PointND> pointList = new ArrayList<PointND>();
				for(int i=0; i< variants.size(); i++){
					OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(variants.get(i), device);
					int elementCountU = (int)TessellationUtil.getSamplingU(variants.get(i));
					int elementCountV = (int)TessellationUtil.getSamplingV(variants.get(i));
					CLBuffer<FloatBuffer> inputBuffer = OpenCLParzenWindowMotionField.generateTimeSamplingPoints((float)sampleTime, elementCountU, elementCountV, context, device);
					CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.READ_WRITE);
					os.evaluate(inputBuffer, outputBuffer);
					CLCommandQueue clc = device.createCommandQueue();
					clc.putReadBuffer(outputBuffer, true).finish();
					clc.release();
					for (int j = 0; j < elementCountU*elementCountV; j++){
						PointND point = new PointND(outputBuffer.getBuffer().get(), outputBuffer.getBuffer().get(), outputBuffer.getBuffer().get());
						pointList.add(point);
					}
					outputBuffer.getBuffer().rewind();
					inputBuffer.release();
					outputBuffer.release();
				}
				//OpenCLUtil.releaseContext(context);
				result = new PointND[pointList.size()];
				pointList.toArray(result);
			}	else {
				ArrayList<PointND[]> allPoints= new ArrayList<PointND[]>();
				int sizes = 0;
				for(int i=0; i< variants.size(); i++){
					PointND [] current = variants.get(i).getRasterPoints(TessellationUtil.getSamplingU(variants.get(i)), TessellationUtil.getSamplingV(variants.get(i)), time);
					allPoints.add(current);
					sizes += current.length;
				}
				result = new PointND[sizes];
				sizes = 0;
				for(int i=0; i< allPoints.size(); i++){
					PointND[] current = allPoints.get(i);
					for (int j=0;j<current.length;j++){
						result[j+sizes] = current[j];
					}
					sizes += current.length;
				}

			}
			timePointMap.put(time, result);
		}
		return result;

	}

	/**
	 * @return the variants
	 */
	public ArrayList<TimeVariantSurfaceBSpline> getVariants() {
		return variants;
	}

	/**
	 * @param variants the variants to set
	 */
	public void setVariants(ArrayList<TimeVariantSurfaceBSpline> variants) {
		this.variants = variants;
	}

}
