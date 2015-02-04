package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.io.VTKMeshReader;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLAppendBufferRenderer;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.opencl.shapes.OpenCLUniformTextureSurfaceBSpline;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * This class provides methods to render projection images of Surface Splines
 * and Time Variant Surface Splines. DISCLAIMER: Splines created by
 * EstimateBSplineSurface may have the wrong orientation set. If output appears
 * to be wrong, check ((OpenCLUniformTextureSurfaceBSpline)
 * clSpline).setClockwise(false); , and change to true/false
 * 
 * @author Marco Boegel
 * 
 */
public class OpenCLSplineRenderer {
	CLContext context = OpenCLUtil.createContext();
	CLDevice device = context.getMaxFlopsDevice();

	
	/**
	 * This method renders a Spline and displays it in an imagej window
	 * 
	 * @param s
	 *            Spline
	 */
	public void renderAppendBuffer(AbstractShape s) {
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640 * antialias;
		int height = 480 * antialias;

		float density = 1.06f;

		Configuration.loadConfiguration();
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(
				elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(width, height);

		CLBuffer<IntBuffer> priorities = context.createIntBuffer(1 + 1,
				Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		priorities.getBuffer().put(10);

		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(1 + 1,
				Mem.READ_ONLY);
		mu.getBuffer().put((float) MaterialsDB.getMaterial("Air").getDensity());
		mu.getBuffer().put((float) density);

		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix(
				"[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(
				device);
		render.init(width, height);

		OpenCLEvaluatable clSpline;

		clSpline = OpenCLUtil.getOpenCLEvaluatableSubclass(s, device);
		((OpenCLUniformTextureSurfaceBSpline) clSpline).setClockwise(false); // TODO:
																				// check
																				// why
																				// this
																				// is
																				// not
																				// set
																				// correctly
																				// automatically

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(
				elementCountU * elementCountV * 3, Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration()
				.getGeometry().getProjectionMatrices().length;
		ImageStack stack;
		if (antialiasXonly) {
			stack = new ImageStack(width / antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width / antialias, height / antialias,
					(int) timePoints);
		}

		for (int k = 0; k < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration()
					.getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++) {
				matrix.setElementValue(0, i, matrix.getElement(0, i)
						* antialias);
				if (!antialiasXonly)
					matrix.setElementValue(1, i, matrix.getElement(1, i)
							* antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			samplingPoints = generateSamplingPoints(elementCountV,
					elementCountU);
			long time = System.nanoTime();

			clSpline.evaluate(samplingPoints, outputBuffer);

			// // project points
			render.project(outputBuffer);

			// draw on the screen
			if (clSpline.isClockwise()) {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, -1);
			} else {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, 1);
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection " + k
					+ " global took: " + (time / 1000000) + "ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "
					+ (time / 1000000) + "ms ");

			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();

			float[] array = new float[width * height];
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					array[(j * width) + i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width, height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width / antialias, height);
				} else {
					fl = fl.resize(width / antialias, height / antialias);
				}
			}
			stack.setPixels(fl.getPixels(), k + 1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "
				+ (totaltime / 1000000) + "ms ");

		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: " + (time / 1000000)
				+ "ms ");

	}

	/**
	 * This method renders a spline, displays it in an imagej window and also
	 * returns the result as a Grid3D
	 * 
	 * @param s
	 *            Spline
	 * @return Grid3D of the resulting projection images
	 */
	public Grid3D renderAppendBufferToGrid(AbstractShape s) {
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640 * antialias;
		int height = 480 * antialias;

		float density = 1.06f;

		Configuration.loadConfiguration();
		// CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(0.0f,
		// elementCountV, elementCountU);
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(
				elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(width, height);

		CLBuffer<IntBuffer> priorities = context.createIntBuffer(1 + 1,
				Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		priorities.getBuffer().put(10);

		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(1 + 1,
				Mem.READ_ONLY);
		mu.getBuffer().put((float) MaterialsDB.getMaterial("Air").getDensity());
		mu.getBuffer().put((float) density);

		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix(
				"[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(
				device);
		render.init(width, height);

		OpenCLEvaluatable clSpline;

		clSpline = OpenCLUtil.getOpenCLEvaluatableSubclass(s, device);
		((OpenCLUniformTextureSurfaceBSpline) clSpline).setClockwise(false); // TODO:
																				// check
																				// why
																				// this
																				// is
																				// not
																				// set
																				// correctly
																				// automatically

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(
				elementCountU * elementCountV * 3, Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration()
				.getGeometry().getProjectionMatrices().length;
		ImageStack stack;
		if (antialiasXonly) {
			stack = new ImageStack(width / antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width / antialias, height / antialias,
					(int) timePoints);
		}

		for (int k = 0; k < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration()
					.getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++) {
				matrix.setElementValue(0, i, matrix.getElement(0, i)
						* antialias);
				if (!antialiasXonly)
					matrix.setElementValue(1, i, matrix.getElement(1, i)
							* antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			samplingPoints = generateSamplingPoints(elementCountV,
					elementCountU);
			long time = System.nanoTime();

			clSpline.evaluate(samplingPoints, outputBuffer);

			// // project points
			render.project(outputBuffer);

			// draw on the screen
			if (clSpline.isClockwise()) {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, -1);
			} else {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, 1);
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection " + k
					+ " global took: " + (time / 1000000) + "ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			// render.drawScreen(screenBuffer);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "
					+ (time / 1000000) + "ms ");

			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();

			float[] array = new float[width * height];
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					array[(j * width) + i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width, height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width / antialias, height);
				} else {
					fl = fl.resize(width / antialias, height / antialias);
				}
			}
			stack.setPixels(fl.getPixels(), k + 1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "
				+ (totaltime / 1000000) + "ms ");

		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: " + (time / 1000000)
				+ "ms ");

		Grid3D projections = ImageUtil.wrapImagePlus(image);

		return projections;
	}

	public CLBuffer<FloatBuffer> generateSamplingPoints(int elementCountV,
			int elementCountU) {
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(
				elementCountU * elementCountV * 2, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++) {
			for (int i = 0; i < elementCountU; i++) {
				samplingPoints.getBuffer().put(i * (1.0f / elementCountU));
				samplingPoints.getBuffer().put(j * (1.0f / elementCountV));
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}

	public CLBuffer<FloatBuffer> generateScreenBuffer(int width, int height) {
		CLBuffer<FloatBuffer> screenBuffer = context.createFloatBuffer(width
				* height, Mem.READ_WRITE);

		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				screenBuffer.getBuffer().put(0.f);
			}
		}
		screenBuffer.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(screenBuffer, true).finish();
		clc.release();
		return screenBuffer;
	}

	public static void main(String args[]) throws Exception {
		int sampling = 1;
		VTKMeshReader vRead = new VTKMeshReader();
		String filename = FileUtil.myFileChoose(".vtk", false);
		vRead.readFile(filename);
		EstimateBSplineSurface estimator = new EstimateBSplineSurface(
				vRead.getPts());
		SurfaceUniformCubicBSpline spline = estimator
				.estimateUniformCubic(sampling);

		Configuration.loadConfiguration();
		Configuration c = Configuration.getGlobalConfiguration();
		Grid3D grid = new Grid3D(c.getGeometry().getReconDimensionX(), c
				.getGeometry().getReconDimensionY(), c.getGeometry()
				.getReconDimensionZ());

		for (int i = 0; i < grid.getSize()[0]; i++) {
			double u = ((double) i) / (grid.getSize()[0]);
			for (int j = 0; j < grid.getSize()[1]; j++) {
				double v = ((double) j) / (grid.getSize()[1]);
				PointND p = spline.evaluate(u, v);
				if (0 <= -((p.get(0) + c.getGeometry().getOriginX()) / c
						.getGeometry().getVoxelSpacingX())
						&& 0 <= -((p.get(1) + c.getGeometry().getOriginY()) / c
								.getGeometry().getVoxelSpacingY())
						&& 0 <= ((p.get(2) - c.getGeometry().getOriginZ()) / c
								.getGeometry().getVoxelSpacingZ())
						&& -((p.get(0) + c.getGeometry().getOriginX()) / c
								.getGeometry().getVoxelSpacingX()) < grid
								.getSize()[0]
						&& -((p.get(1) + c.getGeometry().getOriginY()) / c
								.getGeometry().getVoxelSpacingY()) < grid
								.getSize()[1]
						&& ((p.get(2) - c.getGeometry().getOriginZ()) / c
								.getGeometry().getVoxelSpacingZ()) < grid
								.getSize()[2])
					grid.setAtIndex(
							(int) -((p.get(0) + c.getGeometry().getOriginX()) / c
									.getGeometry().getVoxelSpacingX()),
							(int) -((p.get(1) + c.getGeometry().getOriginY()) / c
									.getGeometry().getVoxelSpacingY()),
							(int) ((p.get(2) - c.getGeometry().getOriginZ()) / c
									.getGeometry().getVoxelSpacingZ()), 100);
			}
		}
		grid.show();

		OpenCLSplineRenderer o = new OpenCLSplineRenderer();
		o.renderAppendBuffer(spline);

	}

	/**
	 * This method renders a time variant spline and displays the projections in
	 * an imagej window
	 * 
	 * @param s
	 *            time variant spline
	 */
	public void SurfaceBSplineRenderingAppendBuffer(AbstractShape s) {
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 1;
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640 * antialias;
		int height = 480 * antialias;
		float density = 1.06f;
		Configuration.loadConfiguration();
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(0.0f,
				elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(width, height);

		CLBuffer<IntBuffer> priorities = context.createIntBuffer(1 + 1,
				Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		priorities.getBuffer().put(10);

		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(1 + 1,
				Mem.READ_ONLY);
		mu.getBuffer().put((float) MaterialsDB.getMaterial("Air").getDensity());
		mu.getBuffer().put((float) density);

		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix(
				"[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(
				device);
		render.init(width, height);

		OpenCLEvaluatable clSpline;

		clSpline = OpenCLUtil.getOpenCLEvaluatableSubclass(s, device);

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(
				elementCountU * elementCountV * elementCountT * 3,
				Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration()
				.getGeometry().getProjectionMatrices().length;
		// TimeWarper warper = new HarmonicTimeWarper(numberOfHeartBeats);
		ImageStack stack;
		if (antialiasXonly) {
			stack = new ImageStack(width / antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width / antialias, height / antialias,
					(int) timePoints);
		}

		for (int k = 0; k < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration()
					.getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++) {
				matrix.setElementValue(0, i, matrix.getElement(0, i)
						* antialias);
				if (!antialiasXonly)
					matrix.setElementValue(1, i, matrix.getElement(1, i)
							* antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			double sampleTime = (float) ((1.0 / timePoints) * k);
			samplingPoints = generateSamplingPoints((float) sampleTime,
					elementCountV, elementCountU);
			long time = System.nanoTime();

			// create spline points
			clSpline.evaluate(samplingPoints, outputBuffer);

			// // project points
			render.project(outputBuffer);

			// draw on the screen
			if (clSpline.isClockwise()) {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, -1);
			} else {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, 1);
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection " + k
					+ " global took: " + (time / 1000000) + "ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "
					+ (time / 1000000) + "ms ");

			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();

			float[] array = new float[width * height];
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					array[(j * width) + i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width, height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width / antialias, height);
				} else {
					fl = fl.resize(width / antialias, height / antialias);
				}
			}
			stack.setPixels(fl.getPixels(), k + 1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "
				+ (totaltime / 1000000) + "ms ");

		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: " + (time / 1000000)
				+ "ms ");

	}

	/**
	 * This method renders a timevariant spline, displays it in an imagej window
	 * and returns the result as a Grid3D
	 * 
	 * @param s
	 *            timevariant spline
	 * @return Grid3D of the resulting projection images
	 */
	public Grid3D SurfaceBSplineRenderingAppendBufferToGrid(AbstractShape s) {
		long totaltime = System.nanoTime();
		int elementCountU = 100;
		int elementCountV = 100;
		int elementCountT = 1;
		int antialias = 1;
		boolean antialiasXonly = false;
		int width = 640 * antialias;
		int height = 480 * antialias;
		float density = 1.06f;
		Configuration.loadConfiguration();
		CLBuffer<FloatBuffer> samplingPoints = generateSamplingPoints(0.0f,
				elementCountV, elementCountU);
		System.out.println("LocalMemSize " + device.getLocalMemSize());
		if (antialiasXonly) {
			height = 480;
		}
		CLBuffer<FloatBuffer> screenBuffer = generateScreenBuffer(width, height);

		CLBuffer<IntBuffer> priorities = context.createIntBuffer(1 + 1,
				Mem.READ_ONLY);
		priorities.getBuffer().put(0);
		priorities.getBuffer().put(10);

		priorities.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(priorities, false).finish();

		CLBuffer<FloatBuffer> mu = context.createFloatBuffer(1 + 1,
				Mem.READ_ONLY);
		mu.getBuffer().put((float) MaterialsDB.getMaterial("Air").getDensity());
		mu.getBuffer().put((float) density);

		mu.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(mu, false).finish();

		SimpleMatrix proj = new SimpleMatrix(
				"[[-446.54410228325054 -511.663331416319 -3.105016244120407E-13 -224000.0]; [-233.8488155484563 53.98825304252766 599.0000000000002 -168000.00000000003]; [-0.9743700647852351 0.2249510543438652 0.0 -700.0]]");
		OpenCLAppendBufferRenderer render = new OpenCLAppendBufferRenderer(
				device);
		render.init(width, height);

		OpenCLEvaluatable clSpline;

		clSpline = OpenCLUtil.getOpenCLEvaluatableSubclass(s, device);

		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(
				elementCountU * elementCountV * elementCountT * 3,
				Mem.READ_WRITE);
		render.setProjectionMatrix(proj);
		double timePoints = Configuration.getGlobalConfiguration()
				.getGeometry().getProjectionMatrices().length;
		ImageStack stack;
		if (antialiasXonly) {
			stack = new ImageStack(width / antialias, height, (int) timePoints);
		} else {
			stack = new ImageStack(width / antialias, height / antialias,
					(int) timePoints);
		}

		for (int k = 0; k < timePoints; k++) {
			SimpleMatrix matrix = Configuration.getGlobalConfiguration()
					.getGeometry().getProjectionMatrices()[k].computeP();
			for (int i = 0; i < 4; i++) {
				matrix.setElementValue(0, i, matrix.getElement(0, i)
						* antialias);
				if (!antialiasXonly)
					matrix.setElementValue(1, i, matrix.getElement(1, i)
							* antialias);
			}
			render.setProjectionMatrix(matrix);
			samplingPoints.release();
			double sampleTime = (float) ((1.0 / timePoints) * k);
			samplingPoints = generateSamplingPoints((float) sampleTime,
					elementCountV, elementCountU);
			long time = System.nanoTime();

			// create spline points
			clSpline.evaluate(samplingPoints, outputBuffer);

			// // project points
			render.project(outputBuffer);

			// draw on the screen
			if (clSpline.isClockwise()) {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, -1);
			} else {
				render.drawTrianglesGlobal(outputBuffer, screenBuffer, 1,
						elementCountU, elementCountV, 1);
			}

			time = System.nanoTime() - time;
			System.out.println("Open CL computation for projection " + k
					+ " global took: " + (time / 1000000) + "ms ");

			// evaluate absorption model
			time = System.nanoTime();
			render.drawScreenMonochromatic(screenBuffer, mu, priorities);
			time = System.nanoTime() - time;
			System.out.println("monochromatic screen buffer drawing took: "
					+ (time / 1000000) + "ms ");

			CLCommandQueue clc = device.createCommandQueue();
			clc.putReadBuffer(screenBuffer, true).finish();
			clc.release();

			float[] array = new float[width * height];
			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					array[(j * width) + i] = screenBuffer.getBuffer().get();
				}
			}
			screenBuffer.getBuffer().rewind();
			ImageProcessor fl = new FloatProcessor(width, height);
			fl.setPixels(array);
			if (antialias > 1) {
				fl.setInterpolationMethod(ImageProcessor.BILINEAR);
				if (antialiasXonly) {
					fl = fl.resize(width / antialias, height);
				} else {
					fl = fl.resize(width / antialias, height / antialias);
				}
			}
			stack.setPixels(fl.getPixels(), k + 1);
			render.resetBuffers();
			screenBuffer.release();
			screenBuffer = generateScreenBuffer(width, height);
		}
		totaltime = System.nanoTime() - totaltime;
		System.out.println("Open CL computation for all projections took: "
				+ (totaltime / 1000000) + "ms ");

		ImagePlus image = new ImagePlus("GPU Projections", stack);
		image.show();

		render.resetBuffers();
		long time = System.nanoTime();

		time = System.nanoTime();
		render.drawScreen(screenBuffer);
		time = System.nanoTime() - time;
		System.out.println("screen buffer drawing took: " + (time / 1000000)
				+ "ms ");

		Grid3D projections = ImageUtil.wrapImagePlus(image);

		return projections;

	}

	public CLBuffer<FloatBuffer> generateSamplingPoints(float tIndex,
			int elementCountV, int elementCountU) {
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(
				elementCountU * elementCountV * 3, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++) {
			for (int i = 0; i < elementCountU; i++) {
				samplingPoints.getBuffer().put(i * (1.0f / elementCountU));
				samplingPoints.getBuffer().put(j * (1.0f / elementCountV));
				samplingPoints.getBuffer().put(tIndex);
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}
}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/