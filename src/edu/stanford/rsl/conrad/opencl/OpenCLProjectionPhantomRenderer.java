/*
 * Copyright (C) 2010-2014 Peter Fischer, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLCompoundShape;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLTimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.MovingBallPhantom;
import edu.stanford.rsl.conrad.phantom.asmheart.CONRADCardiacModel3D;
import edu.stanford.rsl.conrad.phantom.renderer.StreamingPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.xcat.CombinedBreathingHeartScene;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.RegKeys;

/**
 * Class to enable creation of projections using OpenCL.
 * Interface for Conrad to OpenCLAppendBufferRenderer
 * 
 * @author peterf
 *
 */
public class OpenCLProjectionPhantomRenderer extends StreamingPhantomRenderer {

	protected AnalyticPhantom phantom;
	protected Trajectory trajectory = Configuration.getGlobalConfiguration().getGeometry();

	private CLContext clContext;
	
	protected OpenCLAppendBufferRenderer renderer; // better use parent class OpenCLRenderer?
	protected CLBuffer<FloatBuffer> screenBuffer;
	protected CLBuffer<FloatBuffer> mu;
	protected CLBuffer<IntBuffer> priorities;
	protected ArrayList<OpenCLEvaluatable> clEvaluatables = new ArrayList<OpenCLEvaluatable>();
	protected CLBuffer<FloatBuffer> outputBuffer;
	private int elementCountU = 100;
	private int elementCountV = 100;
	private int elementCountT = 1;

	@Override
	public void createPhantom() {
		// TODO: generalize AbstractShape to allow tessellation of all kinds of objects
		// TODO: tessellate all shapes on GPU
		System.out.println(phantom.getName());
		//PrioritizableScene phantomScene = phantom;

		// Generate sampling points for static splines
		CLBuffer<FloatBuffer> samplingPoints = OpenCLUtil.generateSamplingPoints(elementCountU, elementCountV, renderer.context, renderer.device);
		CLBuffer<FloatBuffer> samplingPointsVariants; // can only be initialized in the loop
		CLBuffer<FloatBuffer> samplingPointsVariantsHeart = null;
		SimpleVector diaphragmMotion = null;
		for (int k =0; k  < dimz; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k].computePMetric();
			renderer.setProjectionMatrix(matrix);
			boolean negativeDepth = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k].negativeDepth();
			double sampleTime = (float)((1.0/ (dimz-1))*k);
			if (phantom instanceof AnalyticPhantom4D) {
				sampleTime = ((AnalyticPhantom4D) phantom).getTimeWarper().warpTime(sampleTime);
			}
			if (phantom instanceof MovingBallPhantom){
				clEvaluatables.clear();
				PrioritizableScene scene = ((MovingBallPhantom) phantom).getScene(sampleTime);
				Translation centerTranslation = phantom.computeCenterTranslation();
				// apply heart translation to heart parts.
				for (PhysicalObject o:scene) {
					applyCenterTranslation(centerTranslation, o);
					OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(o.getShape(), renderer.device);		
					clEvaluatables.add(os);
				}
			}
			if (phantom instanceof CombinedBreathingHeartScene){
				double heartTime = (float) ((CombinedBreathingHeartScene) phantom).getHeart().getTimeWarper().warpTime(sampleTime);
				samplingPointsVariantsHeart = generateTimeSamplingPoints((float) heartTime);
				sampleTime = ((CombinedBreathingHeartScene) phantom).getBreathing().getTimeWarper().warpTime(sampleTime);
				diaphragmMotion =((CombinedBreathingHeartScene) phantom).getBreathing().getDiaphragmMotionVector(0, sampleTime);
			}
			samplingPointsVariants = generateTimeSamplingPoints((float)sampleTime);

			long time = System.nanoTime();
			for (int ID = 0; ID < clEvaluatables.size(); ID++){
			//for (int ID = 4; ID < 5; ID++){
					
			OpenCLEvaluatable os = clEvaluatables.get(ID);
				// create spline points


				// project points
				if (phantom instanceof CombinedBreathingHeartScene){
					CombinedBreathingHeartScene combi = (CombinedBreathingHeartScene) phantom;
					boolean heartSpline = false;
					if (os instanceof OpenCLTimeVariantSurfaceBSpline){
						String title = ((OpenCLTimeVariantSurfaceBSpline)os).getTitle();
						for (TimeVariantSurfaceBSpline tvs:combi.getHeart().getVariants()){
							if (tvs.getTitle().equals(title)){
								heartSpline = true;
								break;
							}
						}
					}
					if (heartSpline){
						os.evaluate(samplingPointsVariantsHeart, outputBuffer);
						renderer.project(outputBuffer, diaphragmMotion);
					} else {
						if (os.isTimeVariant()){
							os.evaluate(samplingPointsVariants, outputBuffer);
						} else {
							os.evaluate(samplingPoints, outputBuffer, elementCountU, elementCountV);
						}
						renderer.project(outputBuffer);
					}
				} else if(phantom instanceof CONRADCardiacModel3D){
					if (outputBuffer != null)
						outputBuffer.release();
					outputBuffer = null;
					OpenCLCompoundShape s = (OpenCLCompoundShape) os;
					outputBuffer = clContext.createFloatBuffer(s.size()*3*3, Mem.READ_WRITE);
					s.evaluate(samplingPoints, outputBuffer);
					//renderer.debugOut(outputBuffer);
					renderer.project(outputBuffer);
				
				}else {
					if (os.isTimeVariant()){
						os.evaluate(samplingPointsVariants, outputBuffer);
					} else {
						os.evaluate(samplingPoints, outputBuffer, elementCountU, elementCountV);
					}
					renderer.project(outputBuffer);	
				}
				//renderer.debugOut(outputBuffer);
				// draw on the screen
				//System.out.println("clockwise: " + os.isClockwise() + " " + negativeDepth);
				boolean clockwise = os.isClockwise();
				if (!negativeDepth) clockwise = !clockwise;
				if(phantom instanceof CONRADCardiacModel3D){
					renderer.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, 1, outputBuffer.getBuffer().capacity()/3, -1);
				}else if (clockwise){
					renderer.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, -1);
				} else {
					renderer.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, 1);
				}
			}
			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection "+ k +" global took: "+ (time/1000000) +"ms ");

			evaluateAbsorptionModel(k);

			renderer.resetBuffers();			// its not the internal buffer, also used by TestOpenCL
			samplingPointsVariants.release();
			if (phantom instanceof CombinedBreathingHeartScene){
				samplingPointsVariantsHeart.release();
			}
		}
		samplingPoints.release();
		release();
	}

	protected void evaluateAbsorptionModel(int k) {
		// evaluate absorption model
		long time = System.nanoTime();
		renderer.drawScreenMonochromatic(screenBuffer, mu, priorities);
		//render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("monochromatic screen buffer drawing took: "+(time/1000000)+"ms ");


		CLCommandQueue clc = renderer.device.createCommandQueue();
		clc.putReadBuffer(screenBuffer, true).finish();
		clc.release();

		float[] array = new float [dimx*dimy]; 
		for (int j = 0; j < dimy; j++){
			for (int i = 0; i < dimx; i++){
				array[(j*dimx)+i] = screenBuffer.getBuffer().get();
			}
		}
		screenBuffer.getBuffer().rewind();

		// Save data to buffer
		Grid2D image = new Grid2D(array, dimx, dimy);
		buffer.add(image, k); // its not the buffer, deleted it

		screenBuffer.getBuffer().clear(); 	// its not the screenbuffer, copied TestOpenCL code
	}

	@Override
	public String toString() {
		return "OpenCL Projection Phantom";
	}

	public void release(){
		if (screenBuffer != null) screenBuffer.release();
		screenBuffer = null;
		if (outputBuffer != null) outputBuffer.release();
		outputBuffer = null;
		if (priorities != null) priorities.release();
		priorities = null;
		if (mu != null) mu.release();
		mu = null;
		if (renderer != null) renderer.release();
		renderer = null;
	}

	private void applyCenterTranslation(Translation centerTranslation, PhysicalObject o){
		o.applyTransform(centerTranslation);
		String translationString = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING);
		if (translationString != null){
			// Center b/w RKJC & LKJC: -292.6426  211.7856  440.7783 (subj 5, static60),-401.1700  165.9885  478.5600 (subj 2, static60)
			// XCAT Center by min & max: -177.73999504606988, 179.8512744259873, 312.19713254613583
			// translationVector = (XCAT Center by min & max) - (Center b/w RKJC & LKJC)=>
			// 114.9026, -31.9343, -128.5811 (subj5),  120, 3, -110(subj2) Try 114.0568    2.4778 -106.2550
			String [] values = translationString.split(", ");
			SimpleVector translationVector = new SimpleVector(Double.parseDouble(values[0]), Double.parseDouble(values[1]), Double.parseDouble(values[2]));
			Translation translationToRotationCenter = new Translation(translationVector);
			o.getShape().applyTransform(translationToRotationCenter);
		}		
	}
	


	protected void configure(AnalyticPhantom phantom, CLContext context, CLDevice device){
		this.phantom = phantom;
		this.clContext = context;
		
		buffer = new ImageGridBuffer();
		projectionNumber = -1;
		dimx = Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth();
		dimy = Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight();
		dimz = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize() * Configuration.getGlobalConfiguration().getNumSweeps();
		//originIndexX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		//originIndexY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		//originIndexZ = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();

		// OpenCL specific configurations
		// TODO: include anti-aliasing

		renderer = new OpenCLAppendBufferRenderer(device);
		renderer.init(dimx, dimy);
		screenBuffer = renderer.generateFloatBuffer(dimx, dimy, Mem.READ_WRITE);
		outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);

		// priorities
		priorities = context.createIntBuffer(phantom.size() +1, Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		for (PhysicalObject o: phantom){
			if(phantom instanceof CONRADCardiacModel3D){
				int prio = priorities.getBuffer().position();
				priorities.getBuffer().put(prio);
				//System.out.println("Priority is: " + prio);
			}else{
				priorities.getBuffer().put(phantom.getPriority(o));
			}
		}
		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		String disableAutoCenterBoolean = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.DISABLE_CENTERING_4DPHANTOM_PROJECTION_RENDERING);
		boolean disableAutoCenter = false;
		if (disableAutoCenterBoolean != null){
			disableAutoCenter = Boolean.parseBoolean(disableAutoCenterBoolean);
		}
		
		Translation centerTranslation = new Translation(new SimpleVector(0,0,0));
		if (!disableAutoCenter){
			SimpleVector center = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);
			centerTranslation = new Translation(center.negated());
		}
		
		for (PhysicalObject o: phantom) {
			applyCenterTranslation(centerTranslation, o);
			AbstractShape s = o.getShape();
			if (phantom instanceof CombinedBreathingHeartScene) {
				CombinedBreathingHeartScene combi = (CombinedBreathingHeartScene) phantom;
				if (combi.getHeart().getVariants().contains(s)) {
					// apply heart translation to heart parts.
					s.applyTransform(combi.getHeartTranslation());
					OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(s, renderer.device);		
					clEvaluatables.add(os);
				} else {
					OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(s, renderer.device);		
					clEvaluatables.add(os);
				}
			} else {
				OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(s, renderer.device);
				clEvaluatables.add(os);				
			}
		}



	}

	protected void generateMuMap(CLContext context, CLDevice device) {
		// absorption model
		mu = context.createFloatBuffer(phantom.size() +1, Mem.READ_ONLY); 
		mu.getBuffer().put((float) phantom.getBackgroundMaterial().getDensity());
		for (PhysicalObject o: phantom){
			float density = (float) o.getMaterial().getDensity();
			mu.getBuffer().put(density);
			//System.out.println(o.getMaterial().getName() + " density " + o.getMaterial().getDensity());
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();
	}

	@Override
	public void configure() throws Exception{
		AnalyticPhantom phantom;
		if(this.phantom != null){
			phantom = this.phantom;
		}else{
			phantom = AnalyticPhantom.getCurrentPhantom();
		}

		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();

		configure(phantom, context, device);

		generateMuMap(context, device);

		configured = true;
	}

	public CLBuffer<FloatBuffer> generateTimeSamplingPoints(float tIndex){
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = renderer.context.createFloatBuffer(elementCountU * elementCountV*3, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
				samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
				samplingPoints.getBuffer().put(tIndex);
				//System.out.println(i*(1.0f / elementCountU) + " " +j*(1.0f / elementCountV)+ " "+t*(1.0f / elementCountT) );
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = renderer.device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@Article{Maier12-FSO,\n" +
				"  author = {Maier, A. and Hofmann, H.G. and Schwemmer, C. and Hornegger, J. and Keil, A. and Fahrig, R.},\n" +
				"  title = {Fast simulation of x-ray projections of spline-based surfaces using an append buffer},\n" +
				"  journal = {Physics in Medicine and Biology},\n" +
				"  volume = {57},\n" +
				"  number = {19},\n" +
				"  pages = {6193-6210},\n" +
				"  year = {2012}\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		String medline = "Maier A, Hofmann HG, Schwemmer C, Hornegger J, Keil A, Fahrig R. Fast simulation of x-ray projections of spline-based surfaces using an append buffer. Phys Med Biol. 57(19):6193-210. 2012";
		return medline;
	}

	/**
	 * @return the phantom
	 */
	public AnalyticPhantom getPhantom() {
		return phantom;
	}

	/**
	 * @param phantom the phantom to set
	 */
	public void setPhantom(AnalyticPhantom phantom) {
		this.phantom = phantom;
	}
}