package edu.stanford.rsl.conrad.opencl;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.geometry.splines.SplineTests;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.UniformCubicBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLCylinder;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLSphere;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLTextureTimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLTimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformTextureBSpline;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformTextureSurfaceBSpline;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.MTFBeadPhantom;
import edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantomProjectorWorker;
import edu.stanford.rsl.conrad.phantom.xcat.CombinedBreathingHeartScene;
import edu.stanford.rsl.conrad.phantom.xcat.HeartScene;
import edu.stanford.rsl.conrad.phantom.xcat.XCatMaterialGenerator;
import edu.stanford.rsl.conrad.phantom.xcat.XCatScene;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class TestOpenCL {



	public double computeMeanError(CLBuffer<FloatBuffer> outputBuffer, double []x, double []y, double []z, int length, boolean out){
		double distanceSum = 0;
		for(int i = 0; i < length; i++) {
			float read = outputBuffer.getBuffer().get();
			if (out) System.out.println("x " +read + " " + x[i]);
			double componentwiseDistance = read - x[i];
			double euclideanDistance = componentwiseDistance * componentwiseDistance;
			read = outputBuffer.getBuffer().get();
			if (out)System.out.println("y " + read + " " + y[i]);
			componentwiseDistance = read - y[i];
			euclideanDistance += componentwiseDistance * componentwiseDistance;
			read = outputBuffer.getBuffer().get();
			if (out)System.out.println("z " + read + " " + z[i]);
			componentwiseDistance = read - z[i];
			euclideanDistance += componentwiseDistance * componentwiseDistance;
			//if (Math.sqrt(euclideanDistance) > 1) {
				//System.out.println(Math.sqrt(euclideanDistance) + " " + i);
			//}
			if (out) System.out.println("error " + Math.sqrt(euclideanDistance));
			distanceSum += Math.sqrt(euclideanDistance);
		}
		return (distanceSum)/length;
	}

	public FloatProcessor evaluateBSplineCPU(int elementCountT, int elementCountV, int elementCountU, int width, int height, TimeVariantSurfaceBSpline cSpline, double [] x, double y[], double z[], SimpleMatrix proj, boolean show){
		long time = System.nanoTime();
		int index = 0;
		FloatProcessor test = new FloatProcessor(width,height);

		for (int t = 0; t < elementCountT; t++){
			for (int j = 0; j < elementCountV; j++){
				for (int i = 0; i < elementCountU; i++){
					SimpleVector point = cSpline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV), t*(1.0f / elementCountT)).getAbstractVector();
					//SimpleVector point = cSpline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV), 0.3).getAbstractVector();
					SimpleVector homPoint = new SimpleVector(point.getElement(0), point.getElement(1), point.getElement(2), 1);
					SimpleVector p = SimpleOperators.multiply(proj, homPoint);
					x[index] = p.getElement(0)/p.getElement(2);
					y[index] = p.getElement(1)/p.getElement(2);
					z[index] = p.getElement(2);
					if (show)test.putPixelValue((int)x[index], (int)y[index], 500); // multiple time points are not visualized
					index++;
				}
			}
			if(show) {
				VisualizationUtil.showImageProcessor(test, "CPU Result").show();
			}
		}
		time = System.nanoTime() - time;

		System.out.println("CPU computation + projection using cubic splines took: "+(time/1000000)+"ms");
		return test;
	}

	public CLBuffer<FloatBuffer> generateSamplingPoints(CLContext context, CLDevice device, float tIndex, int elementCountV, int elementCountU){
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*3, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
				samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
				samplingPoints.getBuffer().put(tIndex);
				//System.out.println(i*(1.0f / elementCountU) + " " +j*(1.0f / elementCountV)+ " "+t*(1.0f / elementCountT) );
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}

	public CLBuffer<FloatBuffer> generateSamplingPoints(CLContext context, CLDevice device, int elementCountT, int elementCountV, int elementCountU){
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*3*elementCountT, Mem.READ_ONLY);
		for (int t = 0; t < elementCountT; t++){
			for (int j = 0; j < elementCountV; j++){
				for (int i = 0; i < elementCountU; i++){
					samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
					samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
					samplingPoints.getBuffer().put(t*(1.0f / elementCountT));
					//System.out.println(i*(1.0f / elementCountU) + " " +j*(1.0f / elementCountV)+ " "+t*(1.0f / elementCountT) );
				}
			}
		}
		samplingPoints.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(samplingPoints, true).finish();
		return samplingPoints;
	}

	public CLBuffer<FloatBuffer> generateSamplingPoints(CLContext context, CLDevice device, int elementCountV, int elementCountU){
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*2, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
				samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
			}
		}
		samplingPoints.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(samplingPoints, true).finish();
		return samplingPoints;
	}
	
	public CLBuffer<FloatBuffer> generateScreenBuffer(CLContext context, CLDevice device, int width, int height){
		CLBuffer<FloatBuffer> screenBuffer = context.createFloatBuffer(width*height, Mem.READ_WRITE);

		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				screenBuffer.getBuffer().put(0.f);
			}
		}
		screenBuffer.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(screenBuffer, true).finish();
		clc.release();
		return screenBuffer;
	}

	public void readAndShowBuffer(int width, int height, CLBuffer<FloatBuffer> screenBuffer, String title){
		float [] array = new float [width*height]; 
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				array[(j*width)+i] = screenBuffer.getBuffer().get();
			}
		}
		screenBuffer.getBuffer().rewind();
		FloatProcessor test = new FloatProcessor(width, height, array, null);
		VisualizationUtil.showImageProcessor(test, title).show();
	}

	@Test
	public void TestOpenCLSurfaceBSplineRenderingYBuffer(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 1;
		boolean show = true;
		boolean out = false;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		heartScene.createArteryTree(heartScene);
		//heartScene.createLesions(heartScene);
		SimpleVector center = SimpleOperators.add(heartScene.getMax().getAbstractVector(), heartScene.getMin().getAbstractVector()).dividedBy(2);
		Translation centerTranslation = new Translation(center.negated());
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context,device,0.0f, elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		int width = 640;
		int height = 480;
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);

		HashMap<String, Integer> priorityMap = XCatScene.getSplinePriorityLUT();
		CLBuffer<IntBuffer> priorities = context.createIntBuffer(heartScene.getVariants().size() +1, Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		for (int i = 0; i < heartScene.getVariants().size();i++){
			String name = heartScene.getVariants().get(i).getTitle();
			priorities.getBuffer().put(priorityMap.get(name));
		}
		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(heartScene.getVariants().size() +1, Mem.READ_ONLY);
		mu.getBuffer().put((float)MaterialsDB.getMaterial("Air").getDensity());
		for (int i = 0; i < heartScene.getVariants().size();i++){
			String name = heartScene.getVariants().get(i).getTitle();
			//System.out.println(name);
			mu.getBuffer().put((float) XCatMaterialGenerator.generateFromMaterialName(XCatScene.getSplineNameMaterialNameLUT().get(name)).getDensity());
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLYBufferRenderer render = new OpenCLYBufferRenderer(device);
		render.init(width, height);

		OpenCLTextureTimeVariantSurfaceBSpline [] clSplines = new OpenCLTextureTimeVariantSurfaceBSpline[heartScene.getVariants().size()];

		for (int ID = 0; ID < heartScene.getVariants().size(); ID++) {
			TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(ID);
			cSpline.applyTransform(centerTranslation);
			clSplines[ID] = new OpenCLTextureTimeVariantSurfaceBSpline(cSpline.getSplines(), device);
		}

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = 10;
		
		ImageStack stack = new ImageStack(width, height, (int) timePoints);
		
		for (int k =0; k < timePoints; k++) {
			
			samplingPoints = generateSamplingPoints(context, device, (float)((1.0/ timePoints)*k), elementCountV, elementCountU);
			long time = System.nanoTime();

			for (int ID = 0; ID < heartScene.getVariants().size(); ID++) {
				// create spline points
				clSplines[ID].evaluate(samplingPoints, outputBuffer);

				// project points
				render.project(outputBuffer);

				// draw on the screen
				render.drawTriangles(outputBuffer, screenBuffer, ID+1);		

			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation local took: "+(time/1000000)+"ms ");


			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreen(screenBuffer);
			//render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "+(time/1000000)+"ms ");
			
			device.createCommandQueue().putReadBuffer(screenBuffer, true).finish();
			float [] array = new float [width*height]; 
			for (int j = 0; j < height; j++){
				for (int i = 0; i < width; i++){
					array[(j*width)+i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			stack.setPixels(array, k+1);
			
			render.resetBuffers();
			screenBuffer = generateScreenBuffer(context, device, width, height);
		}
		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		screenBuffer = generateScreenBuffer(context, device, width, height);

		long time = System.nanoTime();
		for (int ID = 0; ID < heartScene.getVariants().size(); ID++) {
			// create spline points
			clSplines[ID].evaluate(samplingPoints, outputBuffer);

			// project points
			render.project(outputBuffer);

			render.drawTriangles(outputBuffer, screenBuffer, ID +1);				

		}
		time = System.nanoTime() - time;
		System.out.println("Open CL computation local took: "+(time/1000000)+"ms ");


		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: "+(time/1000000)+"ms ");


		device.createCommandQueue().putReadBuffer(screenBuffer, true).finish();		

		if(show) {
			readAndShowBuffer(width, height, screenBuffer, "GPU result");
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}


	}
	
	@Test
	public void testPlaneModel(){
		float x_1 = 2;
		float y_1 = 1;
		float z_1 = 1;
		//
		float x_2 = 3;
		float y_2 = 3;
		float z_2 = 2;
		// 
		float x_3 = 4;
		float y_3 = 1;
		float z_3 = 4;
		
		float A = (z_2-z_3)/(x_2-x_3);
		float B = (z_1-z_2)/(x_1-x_2);
		float C = (y_2-y_3)/(x_2-x_3);
		float D = (y_1-y_2)/(x_1-x_2);
		
		float n_2 = (A - B) / (C-D);
		float n_1 = B-(n_2*D);
		float n_0 = z_1 - (x_1 * n_1) - (y_1 * n_2);
		Assert.assertEquals(true, (n_0 == -1.7500)&& (n_1 ==  1.5000)&& (n_2 ==  -0.2500));
	}
	
	@Test
	public void TestCPURayTracer(){
		XRayDetector detector = null;
		try {
			detector = (XRayDetector) UserUtil.queryObject("Select Detector:", "Detector Selection", XRayDetector.class);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		int elementCountU = 90;
		int elementCountV = 90;
		boolean show = true;
		boolean out = false;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		heartScene.createArteryTree(heartScene);
		//heartScene.createLesions(heartScene);
		SimpleVector center = SimpleOperators.add(heartScene.getMax().getAbstractVector(), heartScene.getMin().getAbstractVector()).dividedBy(2);
		Translation centerTranslation = new Translation(center.negated());
		
		int antialias = 1;
		int width = 640*antialias;
		int height = 480*antialias;
		double timePoints = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices().length;

		ImageStack stack = new ImageStack(width, height);
		// render the scene
		// the number of steps gives the number of volumes rendered. (k < 40)
		
		long time = System.nanoTime();
		
		for (int k = 0; k < timePoints; k++){
			ImageGridBuffer buffer = new ImageGridBuffer();
			IJ.showStatus("Rendering State " + k);
			IJ.showProgress(((double)k)/timePoints);

			AnalyticPhantomProjectorWorker worker = new AnalyticPhantomProjectorWorker();
			worker.setImageProcessorBuffer(null);
			worker.setLatch(null);
			worker.setShowStatus(false);
			worker.setSliceList(null);
			
			try {
				worker.configure(heartScene, detector);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			// tessellate scene -> convert to triangles -> render the triangles
			PrioritizableScene current = heartScene.tessellateSceneFixedUVSampling(elementCountU, elementCountV, 0);
			for (PhysicalObject s: current){
				s.applyTransform(centerTranslation);
			}
			
			// the volume to the hyperstack
			stack.addSlice("Slice z = " +(k-1) + " t = " + k, worker.raytraceScene(current,  Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k]));
			
		}



		// finalize the hyperstack
		ImagePlus hyper = new ImagePlus();
		Calibration calibration = hyper.getCalibration();
		calibration.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		calibration.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		calibration.zOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
		calibration.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
		calibration.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
		calibration.pixelDepth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
		hyper.setCalibration(calibration);
		hyper.setStack(heartScene.getName(), stack);
		
		IJ.showProgress(1.0);
		
		System.out.println("CPU computation took: "+(time/1000000)+"ms ");

		
		if(show) {
			hyper.show();
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	@Test
	public void OpenCLCreatePhantom(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 1;
		boolean show = true;
		boolean out = true;
		String out_path = "D:/PhD/Data/Conrad_Simulations/testauto/img";
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640*antialias;
		int height = 480*antialias;
		if (antialiasXonly) {
			height = 480;
		}
		double numberOfHeartBeats = 2.0;
		
//		HeartScene scene = new HeartScene();
//		scene.init();
//		scene.createArteryTree(scene);
//		scene.createLesions(scene);
//		scene.createPhysicalObjects();
//		AnalyticPhantom phantom = heartScene;
		
		CombinedBreathingHeartScene scene  = new CombinedBreathingHeartScene();
		try {
			scene.configure();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		AnalyticPhantom phantom = scene;
		
		if (show) new ImageJ();
		Configuration.loadConfiguration();

		SimpleVector center = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);
		Translation centerTranslation = new Translation(center.negated());
		CLBuffer<FloatBuffer> samplingPointsVariants = generateSamplingPoints(context, device, 0.0f, elementCountV, elementCountU);
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);

		// priorities
		CLBuffer<IntBuffer> priorities = context.createIntBuffer(phantom.size() +1, Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		for (PhysicalObject o: phantom){
			priorities.getBuffer().put(phantom.getPriority(o));
		}
		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		// absorption model
		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(phantom.size() +1, Mem.READ_ONLY);
		mu.getBuffer().put((float) phantom.getBackgroundMaterial().getDensity());
		for (PhysicalObject o: phantom){
			mu.getBuffer().put((float) o.getMaterial().getDensity());
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(device);
		render.init(width, height);

		ArrayList<OpenCLEvaluatable> clEvaluatables = new ArrayList<OpenCLEvaluatable>(phantom.size());
		for (PhysicalObject o: phantom) {
			o.applyTransform(centerTranslation);
			AbstractShape s = o.getShape();
			OpenCLEvaluatable os = OpenCLUtil.getOpenCLEvaluatableSubclass(s, device);
			clEvaluatables.add(os);
		}

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices().length;
		TimeWarper warper = new HarmonicTimeWarper(numberOfHeartBeats);
		ImageStack stack;
		if (antialiasXonly){
			stack = new ImageStack(width/antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width/antialias, height/antialias, (int) timePoints);
		}
		//ImageStack stack = new ImageStack(width, height, (int) timePoints);
		
		for (int k =0; k  < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++){
				matrix.setElementValue(0, i, matrix.getElement(0, i)*antialias);
				if (!antialiasXonly) matrix.setElementValue(1, i, matrix.getElement(1, i)*antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPointsVariants.release();
			double sampleTime = (float)((1.0/ timePoints)*k);
			sampleTime = warper.warpTime(sampleTime);
			samplingPointsVariants = generateSamplingPoints(context, device, (float)sampleTime, elementCountV, elementCountU);
			long time = System.nanoTime();

			for (int ID = 0; ID < phantom.size(); ID++) {
				// create spline points
				if (clEvaluatables.get(ID).isTimeVariant()){
					clEvaluatables.get(ID).evaluate(samplingPointsVariants, outputBuffer);
				} else {
					clEvaluatables.get(ID).evaluate(samplingPoints, outputBuffer);
				}

				//// project points
				render.project(outputBuffer);

				// draw on the screen
				if (clEvaluatables.get(ID).isClockwise()){
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, -1);
				} else {
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, 1);
				}
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection "+k+" global took: "+(time/1000000)+"ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			//render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "+(time/1000000)+"ms ");
			
			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();
			
			float []array = new float [width*height]; 
			for (int j = 0; j < height; j++){
				for (int i = 0; i < width; i++){
					array[(j*width)+i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width,height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width/antialias, height);	
				} else {
					fl = fl.resize(width/antialias, height/antialias);
				}				
			}
			stack.setPixels(fl.getPixels(), k+1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(context, device, width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "+(totaltime/1000000)+"ms ");
		
		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: "+(time/1000000)+"ms ");
		
		if (out) {
			for (int i=1; i <= stack.getSize(); i++) {
				ImagePlus img = new ImagePlus("", stack.getProcessor(i));
				FileSaver f = new FileSaver(img);
				f.saveAsPng(out_path + i + ".png");
			}
		}

		if(show) {
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	
	@Test
	public void TestOpenCLSurfaceBSplineRenderingAppendBuffer(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 1;
		boolean show = true;
		boolean out = true;
		String out_path = "D:/PhD/Data/Conrad_Simulations/testauto/img";
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640*antialias;
		int height = 480*antialias;
		double numberOfHeartBeats = 2.0;
		
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		heartScene.createArteryTree(heartScene);
		heartScene.createLesions(heartScene);
		SimpleVector center = SimpleOperators.add(heartScene.getMax().getAbstractVector(), heartScene.getMin().getAbstractVector()).dividedBy(2);
		Translation centerTranslation = new Translation(center.negated());
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, 0.0f, elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);

		HashMap<String, Integer> priorityMap = XCatScene.getSplinePriorityLUT();
		CLBuffer<IntBuffer> priorities = context.createIntBuffer(heartScene.getVariants().size() +1, Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		for (int i = 0; i < heartScene.getVariants().size();i++){
			String name = heartScene.getVariants().get(i).getTitle();
			priorities.getBuffer().put(priorityMap.get(name));
		}
		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(heartScene.getVariants().size() +1, Mem.READ_ONLY);
		mu.getBuffer().put((float)MaterialsDB.getMaterial("Air").getDensity());
		for (int i = 0; i < heartScene.getVariants().size();i++){
			String name = heartScene.getVariants().get(i).getTitle();
			mu.getBuffer().put((float) XCatMaterialGenerator.generateFromMaterialName(XCatScene.getSplineNameMaterialNameLUT().get(name)).getDensity());
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(device);
		render.init(width, height);

		OpenCLEvaluatable [] clSplines = new OpenCLEvaluatable[heartScene.getVariants().size()];

		for (int ID = 0; ID < heartScene.getVariants().size(); ID++) {
			AbstractShape s = heartScene.getVariants().get(ID);
			s.applyTransform(centerTranslation);
			clSplines[ID] = OpenCLUtil.getOpenCLEvaluatableSubclass(s, render.device);
		}

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices().length;
		TimeWarper warper = new HarmonicTimeWarper(numberOfHeartBeats);
		ImageStack stack;
		if (antialiasXonly){
			stack = new ImageStack(width/antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width/antialias, height/antialias, (int) timePoints);
		}
		//ImageStack stack = new ImageStack(width, height, (int) timePoints);
		
		for (int k =0; k  < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++){
				matrix.setElementValue(0, i, matrix.getElement(0, i)*antialias);
				if (!antialiasXonly) matrix.setElementValue(1, i, matrix.getElement(1, i)*antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			double sampleTime = (float)((1.0/ timePoints)*k);
			sampleTime = warper.warpTime(sampleTime);
			samplingPoints = generateSamplingPoints(context, device, (float)sampleTime, elementCountV, elementCountU);
			long time = System.nanoTime();

			for (int ID = 0; ID < heartScene.getVariants().size(); ID++) {
				// create spline points
				clSplines[ID].evaluate(samplingPoints, outputBuffer);

				//// project points
				render.project(outputBuffer);

				// draw on the screen
				if (clSplines[ID].isClockwise()){
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, -1);
				} else {
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, 1);
				}
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection "+k+" global took: "+(time/1000000)+"ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			//render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "+(time/1000000)+"ms ");
			
			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();
			
			float []array = new float [width*height]; 
			for (int j = 0; j < height; j++){
				for (int i = 0; i < width; i++){
					array[(j*width)+i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width,height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width/antialias, height);	
				} else {
					fl = fl.resize(width/antialias, height/antialias);
				}				
			}
			stack.setPixels(fl.getPixels(), k+1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(context, device, width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "+(totaltime/1000000)+"ms ");
		
		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: "+(time/1000000)+"ms ");
		
		if (out) {
			for (int i=1; i <= stack.getSize(); i++) {
				ImagePlus img = new ImagePlus("", stack.getProcessor(i));
				FileSaver f = new FileSaver(img);
				f.saveAsPng(out_path + i + ".png");
			}
		}

		if(show) {
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	@Test
	public void TestOpenCLSimpleObjectRenderingAppendBuffer(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 10;
		boolean show = true;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		
		// Create phantom
		AnalyticPhantom phantom = new MTFBeadPhantom();
		
		ArrayList<OpenCLEvaluatable> objects = new ArrayList<OpenCLEvaluatable>();
		for (PhysicalObject o : phantom){
			AbstractShape s = o.getShape();
			objects.add(OpenCLUtil.getOpenCLEvaluatableSubclass(s, device));
		}
		
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640*antialias;
		int height = 480*antialias;
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);

		CLBuffer<IntBuffer> priorities = context.createIntBuffer(phantom.size()+1, Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		for (PhysicalObject o : phantom){
			priorities.getBuffer().put(phantom.getPriority(o));
		}
		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(phantom.size()+1, Mem.READ_ONLY);
		mu.getBuffer().put((float)MaterialsDB.getMaterial("Air").getDensity());
		for (PhysicalObject o : phantom){
			mu.getBuffer().put((float) o.getMaterial().getDensity());
		}
		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(device);
		render.init(width, height);
		
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.READ_WRITE);
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountV, elementCountU);
		
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices().length;
		ImageStack stack;
		if (antialiasXonly){
			stack = new ImageStack(width/antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width/antialias, height/antialias, (int) timePoints);
		}
		//ImageStack stack = new ImageStack(width, height, (int) timePoints);
		
		for (int k =0; k  < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++){
				matrix.setElementValue(0, i, matrix.getElement(0, i)*antialias);
				if (!antialiasXonly) matrix.setElementValue(1, i, matrix.getElement(1, i)*antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			samplingPoints = generateSamplingPoints(context, device, elementCountV, elementCountU);
			long time = System.nanoTime();

			for (int ID = 0; ID < objects.size(); ID++) {
				// create points
				//float[] test = new Box(50, 50, 50).getRasterPoints(elementCountU, elementCountV);
				//OpenCLCylinder obj = new OpenCLCylinder(50, 50, 100, device);
				//OpenCLSphere obj = new OpenCLSphere(50, new PointND(10,10,10), device);
//				float[] test = obj.getRasterPoints(elementCountU, elementCountV);
				objects.get(ID).evaluate(samplingPoints, outputBuffer, elementCountU, elementCountV);
//				//float[] test = ((SimpleSurface) phantom.getObject(ID).getShape()).getRasterPoints(elementCountU,elementCountV);
//				// copy
//				// prepare sampling points
//				for (int j = 0; j < elementCountV; j++){
//					for (int i = 0; i < elementCountU; i++){
//						// strange way of storing & reading coordinates for triangleAppendBuffer
//						outputBuffer.getBuffer().put(test[3*(j*elementCountU+i)]);
//						outputBuffer.getBuffer().put(test[3*(j*elementCountU+i)+1]);
//						outputBuffer.getBuffer().put(test[3*(j*elementCountU+i)+2]);
//					}
//				}
//				outputBuffer.getBuffer().rewind();
//				CLCommandQueue clc = device.createCommandQueue();
//				clc.putWriteBuffer(outputBuffer, true).finish();
//				clc.release();
				
//				CLCommandQueue clc = device.createCommandQueue();
//				clc.putReadBuffer(outputBuffer, true).finish();
//				clc.release();
//				for (int j = 0; j < elementCountV; j++){
//					for (int i = 0; i < elementCountU; i++){
//						// strange way of storing & reading coordinates for triangleAppendBuffer
//						System.out.println(outputBuffer.getBuffer().get());
//						System.out.println(outputBuffer.getBuffer().get());
//						System.out.println(outputBuffer.getBuffer().get());
//					}
//				}
//				outputBuffer.getBuffer().rewind();

				// project points
				render.project(outputBuffer);

				// draw on the screen
				if (objects.get(ID).isClockwise()){
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, -1);
				} else {
					render.drawTrianglesGlobal(outputBuffer, screenBuffer, ID+1, elementCountU, elementCountV, 1);
				}
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection "+k+" global took: "+(time/1000000)+"ms ");


			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			//render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "+(time/1000000)+"ms ");
			
			
			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();
			
			float []array = new float [width*height]; 
			for (int j = 0; j < height; j++){
				for (int i = 0; i < width; i++){
					array[(j*width)+i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width,height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width/antialias, height);	
				} else {
					fl = fl.resize(width/antialias, height/antialias);
				}
				
			}
			stack.setPixels(fl.getPixels(), k+1);
			//stack.getProcessor(k+1).filter(ImageProcessor.BLUR_MORE);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(context, device, width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "+(totaltime/1000000)+"ms ");
		
		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		if(show) {
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	

	@Test
	public void TestOpenCLSurfaceBSplineRenderingRayCast(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 1000;
		int elementCountV = 1000;
		int elementCountT = 1;
		boolean show = false;
		boolean out = false;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(0);
		OpenCLTimeVariantSurfaceBSpline clSpline = new OpenCLTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountT, elementCountV, elementCountU);

		int width = 640;
		int height = 480;
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);
		CLBuffer<FloatBuffer> ranges = context.createFloatBuffer(elementCountU*4, Mem.READ_WRITE);

		for (int j = 0; j < elementCountU*4; j++){
			ranges.getBuffer().put(0.f);
		}
		ranges.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(ranges, true);

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took (with data transfer): "+(time/1000000)+"ms");


		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLRenderer render = new OpenCLRenderer(device);
		render.init(width, height);

		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];

		evaluateBSplineCPU(elementCountT, elementCountV, elementCountU, width, height, cSpline, x, y, z, proj, show);

		render.setProjectionMatrix(proj);
		time = System.nanoTime();
		render.project(outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("GPU projection took: "+(time/1000000)+"ms");
		// fetch data
		device.createCommandQueue().putReadBuffer(outputBuffer, true).finish();

		time = System.nanoTime();
		render.computeMinMaxValues(outputBuffer, ranges);
		time = System.nanoTime() - time;
		System.out.println("GPU min max took: "+(time/1000000)+"ms");
		device.createCommandQueue().putReadBuffer(ranges, true);

		time = System.nanoTime();
		render.drawTrianglesRayCastRanges(outputBuffer, ranges, screenBuffer, length, 20);
		time = System.nanoTime() - time;
		System.out.println("triangle drawing took: "+(time/1000000)+"ms");
		device.createCommandQueue().putReadBuffer(screenBuffer, true);

		time = System.nanoTime();
		render.drawTrianglesRayCast(outputBuffer, screenBuffer, length, 20); // TODO fails
		time = System.nanoTime() - time;
		System.out.println("triangle w/o clipping drawing took: "+(time/1000000)+"ms");
		device.createCommandQueue().putReadBuffer(screenBuffer, true);

		if(show) {
			readAndShowBuffer(width, height, screenBuffer, "GPU result");
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		double meanError = computeMeanError(outputBuffer, x, y, z, length, out);

		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.5E-2);

	}

	@Test
	public void TestOpenCLSurfaceBSplineRenderingZBuffer(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 1000;
		int elementCountV = 1000;
		int elementCountT = 10;
		boolean show = true;
		boolean out = false;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(6);
		OpenCLTimeVariantSurfaceBSpline clSpline = new OpenCLTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountT, elementCountV, elementCountU);

		int width = 640;
		int height = 480;
		CLBuffer<FloatBuffer> screenBuffer = this.generateScreenBuffer(context, device, width, height);

		CLBuffer<IntBuffer> zBuffer = context.createIntBuffer(width*height, Mem.READ_WRITE);

		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				zBuffer.getBuffer().put(0);
			}
		}
		zBuffer.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(zBuffer, true);

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);

		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took (with data transfer): "+(time/1000000)+"ms");

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLRenderer render = new OpenCLRenderer(device);
		render.init(width, height);

		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		this.evaluateBSplineCPU(elementCountT, elementCountV, elementCountU, width, height, cSpline, x, y, z, proj, show);

		render.setProjectionMatrix(proj);
		time = System.nanoTime();
		render.project(outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("GPU projection took: "+(time/1000000)+"ms");
		// fetch data
		device.createCommandQueue().putReadBuffer(outputBuffer, true);

		time = System.nanoTime();
		render.drawTrianglesZBuffer(outputBuffer, screenBuffer, zBuffer, 20);
		time = System.nanoTime() - time;
		System.out.println("triangle drawing took: "+(time/1000000)+"ms");
		device.createCommandQueue().putReadBuffer(screenBuffer, true);
		device.createCommandQueue().putReadBuffer(zBuffer, true);

		//System.out.println("Access: " + access.getBuffer().get() + " " + 640*480);

		if(show) {
			this.readAndShowBuffer(width, height, screenBuffer, "GPU result");
			float [] array2 = new float [width*height]; 
			for (int j = 0; j < height; j++){
				for (int i = 0; i < width; i++){
					array2[(j*width)+i] = zBuffer.getBuffer().get();
				}
			}
			FloatProcessor test = new FloatProcessor(width, height, array2, null);
			VisualizationUtil.showImageProcessor(test, "GPU zBuffer result").show();
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		double meanError = computeMeanError(outputBuffer, x, y, z, length, out);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.5E-2);

	}

	@Test
	public void TestOpenCLSurfaceBSplineRendering(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 1000;
		int elementCountV = 1000;
		int elementCountT = 1;
		boolean show = false;
		boolean out = false;
		if (show) new ImageJ();
		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(0);
		OpenCLTimeVariantSurfaceBSpline clSpline = new OpenCLTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountT, elementCountV, elementCountU);
		int width = 640;
		int height = 480;
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(context, device, width, height);

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took (with data transfer): "+(time/1000000)+"ms");

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLRenderer render = new OpenCLRenderer(device);
		render.init(width, height);

		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		this.evaluateBSplineCPU(elementCountT, elementCountV, elementCountU, width, height, cSpline, x, y, z, proj, show);

		render.setProjectionMatrix(proj);
		time = System.nanoTime();
		render.project(outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("GPU projection took: "+(time/1000000)+"ms");
		// fetch data
		device.createCommandQueue().putReadBuffer(outputBuffer, true).finish();

		time = System.nanoTime();
		render.drawTriangles(outputBuffer, screenBuffer, 20);
		time = System.nanoTime() - time;
		System.out.println("triangle drawing took: "+(time/1000000)+"ms");
		device.createCommandQueue().putReadBuffer(screenBuffer, true);


		if(show) {
			readAndShowBuffer(width, height, screenBuffer, "GPU Result");
			try {
				Thread.sleep(1000000000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		double meanError = computeMeanError(outputBuffer, x, y, z, length, out);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.5E-2);

	}

	@Test
	public void TestOpenCLSurfaceBSplineProjection(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 100;

		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(0);
		OpenCLTimeVariantSurfaceBSpline clSpline = new OpenCLTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountT, elementCountV, elementCountU);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.READ_WRITE);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took (with data transfer): "+(time/1000000)+"ms");

		SimpleMatrix proj = new SimpleMatrix("[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLRenderer render = new OpenCLRenderer(device);

		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		this.evaluateBSplineCPU(elementCountT, elementCountV, elementCountU, 640, 480, cSpline, x, y, z, proj, false);

		render.setProjectionMatrix(proj);
		time = System.nanoTime();
		render.project(outputBuffer);
		time = System.nanoTime() - time;
		System.out.println("GPU projection took: "+(time/1000000)+"ms");
		CLCommandQueue clc = device.createCommandQueue();
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();

		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.5E-2);

	}

	@Test
	public void TestOpenCLTextureTimeSurfaceBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 100;


		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(0);
		OpenCLTextureTimeVariantSurfaceBSpline clSpline = new OpenCLTextureTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountT, elementCountV, elementCountU);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.WRITE_ONLY);
		
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		CLCommandQueue clc = device.createCommandQueue();
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took (with data transfer): "+(time/1000000)+"ms");

		time = System.nanoTime();
		int number = 100;
		for (int i =0; i < number; i++) {
			clSpline.evaluate(samplingPoints, outputBuffer);
		}
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took ("+ number+" calls without data transfer): "+(time/1000000)/number+"ms per call");

		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		time = System.nanoTime();
		int index = 0;
		for (int t = 0; t < elementCountT; t++){
			for (int j = 0; j < elementCountV; j++){
				for (int i = 0; i < elementCountU; i++){
					PointND p = cSpline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV), t*(1.0f / elementCountT));
					x[index] = p.get(0);
					y[index] = p.get(1);
					z[index] = p.get(2);
					index++;
				}
			}
		}
		time = System.nanoTime() - time;
		System.out.println("CPU computation using cubic splines took: "+(time/1000000)+"ms");


		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.5E-2);

	}

	@Test
	public void TestOpenCLUniformTimeSurfaceBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		int elementCountU = 10;
		int elementCountV = 10;
		int elementCountT = 10;


		Configuration.loadConfiguration();
		HeartScene heartScene = new HeartScene();
		heartScene.init();
		TimeVariantSurfaceBSpline cSpline = heartScene.getVariants().get(0);
		OpenCLTimeVariantSurfaceBSpline clSpline = new OpenCLTimeVariantSurfaceBSpline(cSpline.getSplines(), device);

		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*3*elementCountT, Mem.READ_ONLY);
		for (int t = 0; t < elementCountT; t++){
			for (int j = 0; j < elementCountV; j++){
				for (int i = 0; i < elementCountU; i++){
					samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
					samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
					samplingPoints.getBuffer().put(t*(1.0f / elementCountT));
				}
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*elementCountT*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took: "+(time/1000000)+"ms");
		int length = elementCountU*elementCountV*elementCountT;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		time = System.nanoTime();
		int index = 0;
		for (int t = 0; t < elementCountT; t++){
			for (int j = 0; j < elementCountV; j++){
				for (int i = 0; i < elementCountU; i++){
					PointND p = cSpline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV), t*(1.0f / elementCountT));
					x[index] = p.get(0);
					y[index] = p.get(1);
					z[index] = p.get(2);
					index++;
				}
			}
		}
		time = System.nanoTime() - time;
		System.out.println("CPU computation using cubic splines took: "+(time/1000000)+"ms");
		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.0E-3);

	}


	@Test
	public void TestOpenCLUniformTextureSurfaceBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		// select fastest device

		System.out.println("using "+device);

		int elementCountU = 100;
		int elementCountV = 100;

		SurfaceBSpline spline = SplineTests.createTestSurfaceSpline();

		SurfaceUniformCubicBSpline cspline = new SurfaceUniformCubicBSpline(spline.getName(), spline.getControlPoints(), spline.getUKnots(), spline.getVKnots());
		OpenCLUniformTextureSurfaceBSpline clSpline = new OpenCLUniformTextureSurfaceBSpline(spline.getName(),spline.getControlPoints(), spline.getUKnots(), spline.getVKnots(), device);

		// prepare sampling points
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

		// prepare output buffer
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took: "+(time/1000000)+"ms");
		int length = elementCountU*elementCountV;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		time = System.nanoTime();
		int index = 0;
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				PointND p = cspline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV));
				x[index] = p.get(0);
				y[index] = p.get(1);
				z[index] = p.get(2);
				index++;
			}
		}

		time = System.nanoTime() - time;
		System.out.println("CPU computation using cubic splines took: "+(time/1000000)+"ms");
		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.0E-2);
		//context.release();
	}


	@Test
	public void TestOpenCLUniformSurfaceBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		// select fastest device

		System.out.println("using "+device);

		int elementCountU = 1000;
		int elementCountV = 1000;

		SurfaceBSpline spline = SplineTests.createTestSurfaceSpline();

		SurfaceUniformCubicBSpline cspline = new SurfaceUniformCubicBSpline(spline.getName(), spline.getControlPoints(), spline.getUKnots(), spline.getVKnots());
		OpenCLUniformSurfaceBSpline clSpline = new OpenCLUniformSurfaceBSpline(spline.getName(),spline.getControlPoints(), spline.getUKnots(), spline.getVKnots(), device);

		// prepare sampling points
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
		
		// prepare output buffer
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation using cubic splines took: "+(time/1000000)+"ms");
		int length = elementCountU*elementCountV;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		//time = System.nanoTime();
		//int index = 0;
		//for (int j = 0; j < elementCountV; j++){
		//	for (int i = 0; i < elementCountU; i++){
		//		PointND p = spline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV));
		//		x[index] = p.get(0);
		//		y[index] = p.get(1);
		//		z[index] = p.get(2);
		//		index++;
		//	}
		//}
		//time = System.nanoTime() - time;
		//System.out.println("CPU computation using general Splines took: "+(time/1000000)+"ms");
		time = System.nanoTime();
		int index = 0;
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				PointND p = cspline.evaluate(i*(1.0f / elementCountU), j*(1.0f / elementCountV));
				x[index] = p.get(0);
				y[index] = p.get(1);
				z[index] = p.get(2);
				index++;
			}
		}
		time = System.nanoTime() - time;
		System.out.println("CPU computation using cubic splines took: "+(time/1000000)+"ms");
		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.0E-5);
		//context.release();
	}

	@Test
	public void TestOpenCLUniformTextureBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		System.out.println("using "+device);
		int elementCount = 1000000;
		BSpline spline = SplineTests.createTestSpline();
		UniformCubicBSpline cspline = new UniformCubicBSpline(spline.getControlPoints(), spline.getKnots());
		OpenCLUniformBSpline clSpline = new OpenCLUniformTextureBSpline(spline.getControlPoints(), spline.getKnots(), device);
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCount, Mem.READ_ONLY);
		for (int i = 0; i < elementCount; i++){
			samplingPoints.getBuffer().put(i*(1.0f / elementCount));
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		// prepare output buffer
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCount*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation took: "+(time/1000000)+"ms");
		int length = elementCount;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		time = System.nanoTime();
		for (int i = 0; i< length; i++){
			PointND p = cspline.evaluate(((double) i) / (length));
			x[i] = p.get(0);
			y[i] = p.get(1);
			z[i] = p.get(2);	
		}
		time = System.nanoTime() - time;
		System.out.println("CPU computation took: "+(time/1000000)+"ms");
		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.0E-2);
		//context.release();
	}

	@Test
	public void TestOpenCLUniformBSpline(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		System.out.println("using "+device);
		int elementCount = 1000000;
		BSpline spline = SplineTests.createTestSpline();
		UniformCubicBSpline cspline = new UniformCubicBSpline(spline.getControlPoints(), spline.getKnots());
		OpenCLUniformBSpline clSpline = new OpenCLUniformBSpline(spline.getControlPoints(), spline.getKnots(), device);
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCount, Mem.READ_ONLY);
		for (int i = 0; i < elementCount; i++){
			samplingPoints.getBuffer().put(i*(1.0f / elementCount));
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCount*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		clSpline.evaluate(samplingPoints, outputBuffer);
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation took: "+(time/1000000)+"ms");
		int length = elementCount;
		double [] x = new double[length];
		double [] y = new double[length];
		double [] z = new double[length];
		time = System.nanoTime();
		for (int i = 0; i< length; i++){
			PointND p = cspline.evaluate(((double) i) / (length));
			x[i] = p.get(0);
			y[i] = p.get(1);
			z[i] = p.get(2);	
		}
		time = System.nanoTime() - time;
		System.out.println("CPU computation took: "+(time/1000000)+"ms");
		double meanError = computeMeanError(outputBuffer, x, y, z, length, false);
		System.out.println("Mean error = " + meanError + " floating point precistion = " + CONRAD.FLOAT_EPSILON );
		org.junit.Assert.assertTrue(meanError < 1.0E-6);
	}
	
	@Test
	public void TestOpenCLCylinder(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		System.out.println("using "+device);
		int elementCountU = 100;
		int elementCountV = 100;
	
		Cylinder cyl = new Cylinder(5, 5, 10);
		OpenCLCylinder openclCyl = new OpenCLCylinder(5, 5, 10, device);
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountV, elementCountU);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		openclCyl.evaluate(samplingPoints, outputBuffer, elementCountU, elementCountV);
		CLCommandQueue clc = device.createCommandQueue();
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation took: "+(time/1000000)+"ms");
		
		
		float[] pointsOpenCL = new float[elementCountU*elementCountV*3];
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				pointsOpenCL[3*(j*elementCountU+i)] = outputBuffer.getBuffer().get();
				pointsOpenCL[3*(j*elementCountU+i)+1] = outputBuffer.getBuffer().get();
				pointsOpenCL[3*(j*elementCountU+i)+2] = outputBuffer.getBuffer().get();
			}
		}
		outputBuffer.getBuffer().rewind();
		
		time = System.nanoTime();
		float[] pointsCPU = cyl.getRasterPoints(elementCountU, elementCountV);
		time = System.nanoTime() - time;
		System.out.println("CPU computation took: "+(time/1000000)+"ms");
	}

	@Test
	public void TestOpenCLSphere(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		System.out.println("using "+device);
		int elementCountU = 100;
		int elementCountV = 100;
	
		Sphere theSphere = new Sphere(50, new PointND(10,10,10));
		OpenCLSphere openclTheSphere = new OpenCLSphere(50, new PointND(10,10,10), device);
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(context, device, elementCountV, elementCountU);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(elementCountU*elementCountV*3, Mem.WRITE_ONLY);
		long time = System.nanoTime();
		openclTheSphere.evaluate(samplingPoints, outputBuffer, elementCountU, elementCountV);
		CLCommandQueue clc = device.createCommandQueue();
		clc.putReadBuffer(outputBuffer, true).finish();
		clc.release();
		time = System.nanoTime() - time;
		System.out.println("Open CL computation took: "+(time/1000000)+"ms");
		
		
		float[] pointsOpenCL = new float[elementCountU*elementCountV*3];
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				pointsOpenCL[3*(j*elementCountU+i)] = outputBuffer.getBuffer().get();
				pointsOpenCL[3*(j*elementCountU+i)+1] = outputBuffer.getBuffer().get();
				pointsOpenCL[3*(j*elementCountU+i)+2] = outputBuffer.getBuffer().get();
			}
		}
		outputBuffer.getBuffer().rewind();
		
		time = System.nanoTime();
		float[] pointsCPU = theSphere.getRasterPoints(elementCountU, elementCountV);
		time = System.nanoTime() - time;
		System.out.println("CPU computation took: "+(time/1000000)+"ms");
	}
	


	@Test
	public void TestComputeWeights(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		try {
			// select fastest device
			System.out.println("using "+device);

			// create command queue on device.
			CLCommandQueue queue = device.createCommandQueue();

			int elementCount = 1444477;                                  // Length of arrays to process
			int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
			int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize


			InputStream programFile = TestOpenCL.class.getResourceAsStream("bspline.cl");
			CLProgram program = context.createProgram(programFile).build();


			// A, B are input buffers, C is for the result
			CLBuffer<FloatBuffer> clBufferC = context.createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

			// get a reference to the kernel function with the name 'VectorAdd'
			// and map the buffers to its input parameters.
			CLKernel kernel = program.createCLKernel("ComputeWeights");
			kernel.putArgs(clBufferC).putArg(elementCount);

			// asynchronous write of data to GPU device,
			// followed by blocking read to get the computed results back.
			long time = System.nanoTime();
			queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
			.putReadBuffer(clBufferC, true);
			time = System.nanoTime() - time;

			// print first few elements of the resulting buffer to the console.
			System.out.println("weight results snapshot: ");
			for(int i = 0; i < 10; i++)
				System.out.print(clBufferC.getBuffer().get() + ", ");
			System.out.println("...; " + clBufferC.getBuffer().remaining() + " more");

			System.out.println("computation took: "+(time/1000000)+"ms");


		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	
	
	@Test
	public void testCLVersion(){
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		boolean foundAtomicsLocalBase = false;
		boolean foundAtomicsLocalExt = false;
		boolean foundAtomicsGlobalBase = false;
		boolean foundAtomicsGlobalExt = false;
		System.out.println(device.getName());
		System.out.println("GlobalMem: " + device.getGlobalMemSize());
		System.out.println("LocalMem: " + device.getLocalMemSize());
		System.out.println(device.getVersion() + " with " + device.getMaxClockFrequency() +"Hz");
		System.out.println("Extensions:");
		for (String extension:device.getExtensions()){
			System.out.println(extension);
			if (extension.equals("cl_khr_local_int32_base_atomics")) foundAtomicsLocalBase = true;
			if (extension.equals("cl_khr_global_int32_base_atomics")) foundAtomicsGlobalBase = true;
			if (extension.equals("cl_khr_local_int32_extended_atomics")) foundAtomicsLocalExt = true;
			if (extension.equals("cl_khr_global_int32_extended_atomics")) foundAtomicsGlobalExt = true;
		}
		Assert.assertEquals(true, foundAtomicsGlobalBase && foundAtomicsGlobalExt && foundAtomicsLocalBase && foundAtomicsLocalExt);
	}

	@Test
	public void TestVectorAdd(){
		System.out.println("Starting vector add.");
		CLContext context = OpenCLUtil.createContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		try {
			// select fastest device
			System.out.println("using "+device);

			// create command queue on device.
			CLCommandQueue queue = device.createCommandQueue();

			int elementCount = 1444477;                                  // Length of arrays to process
			int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
			int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize


			InputStream programFile = TestOpenCL.class.getResourceAsStream("VectorAdd.cl");
			CLProgram program = context.createProgram(programFile).build();


			// A, B are input buffers, C is for the result
			CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(globalWorkSize, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> clBufferB = context.createFloatBuffer(globalWorkSize, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> clBufferC = context.createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

			// fill input buffers with random numbers
			// (just to have test data; seed is fixed -> results will not change between runs).
			fillBuffer(clBufferA.getBuffer(), 12345);
			fillBuffer(clBufferB.getBuffer(), 67890);

			// get a reference to the kernel function with the name 'VectorAdd'
			// and map the buffers to its input parameters.
			CLKernel kernel = program.createCLKernel("VectorAdd");
			kernel.putArgs(clBufferA, clBufferB, clBufferC).putArg(elementCount);

			// asynchronous write of data to GPU device,
			// followed by blocking read to get the computed results back.
			long time = System.nanoTime();
			queue.putWriteBuffer(clBufferA, false)
			.putWriteBuffer(clBufferB, false)
			.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
			.putReadBuffer(clBufferC, true);
			time = System.nanoTime() - time;

			// print first few elements of the resulting buffer to the console.
			System.out.println("a+b=c results snapshot: ");
			for(int i = 0; i < 10; i++)
				System.out.print(clBufferC.getBuffer().get() + ", ");
			System.out.println("...; " + clBufferC.getBuffer().remaining() + " more");

			System.out.println("computation took: "+(time/1000000)+"ms");


		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static void fillBuffer(FloatBuffer buffer, int seed) {
		Random rnd = new Random(seed);
		while(buffer.remaining() != 0)
			buffer.put(rnd.nextFloat()*100);
		buffer.rewind();
	}



}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/