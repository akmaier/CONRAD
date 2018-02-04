package edu.stanford.rsl.tutorial.physics;

import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;

import java.awt.Color;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * A raytracer for X-rays using the PriorityRaytracer for intersection calculation. It considers only the Compton and the Photoelectric Effect, disregarding other effects which are insignificant for X-rays.
 * 
 * @author Fabian Rückert
 * 
 */
public class XRayTracer {

	// settings
	public static final boolean INFINITE_SIMULATION = true;
	private static final int numRays = 10000;
	private static final int startEnergyEV = 100000;
	private static final int numThreads = Runtime.getRuntime().availableProcessors();

	//enable to use the Geant4 code for sampling of the Compton Scattering Angle/Energy
	public static boolean SAMPLE_GEANT4 = false;

	//enable to just send the rays in x-direction starting from the origin, without a detector
	public static final boolean TEST_MODE = false;

	// variables
	private Thread[] threads = new Thread[numThreads];
	private XRayWorker[] rayTraceThreads = new XRayWorker[numThreads];
	private XRayDetector detector = new XRayDetector();

	/**
	 * Saves the raytracing results for each worker thread.
	 */
	public class RaytraceResult {
		ArrayList<PointND> points = new ArrayList<PointND>();
		ArrayList<Color> colors = new ArrayList<Color>();
		ArrayList<Edge> edges = new ArrayList<Edge>();
		double pathlength = 0;
		double pathlength2 = 0;
		double x = 0;
		double x2 = 0;
		double y = 0;
		double y2 = 0;
		double z = 0;
		double z2 = 0;
		int count = 0;

		public void addResults(RaytraceResult res) {
			this.points.addAll(res.points);
			this.colors.addAll(res.colors);
			this.edges.addAll(res.edges);

			this.count += res.count;
			this.pathlength += res.pathlength;
			this.pathlength2 += res.pathlength2;
			this.x += res.x;
			this.x2 += res.x2;
			this.y += res.y;
			this.y2 += res.y2;
			this.z += res.z;
			this.z2 += res.z2;
		}
	}

	public XRayTracer() {

	}

	/**
	 * Constructs a simple test scene for the raytracer
	 * @return A Test Scene
	 */
	private static PrioritizableScene constructTestScene() {
		PrioritizableScene scene = new PrioritizableScene();
		scene.setName("Test Scene");

		PhysicalObject water = new PhysicalObject();
		water.setMaterial(MaterialsDB.getMaterial("water"));
		water.setNameString("water");
		float boxsizeMM = 30000.f;
		AbstractShape shape = new Box(boxsizeMM, boxsizeMM, boxsizeMM);
		shape.applyTransform(new Translation(-boxsizeMM / 2, -boxsizeMM / 2, -boxsizeMM / 2));

		water.setShape(shape);
		scene.add(water, 0);

		// put a smaller box inside the big box
		PhysicalObject object2 = new PhysicalObject();
		object2.setMaterial(MaterialsDB.getMaterial("air"));
		object2.setNameString("air");
		float boxsize2 = 1000.f;
		AbstractShape shape2 = new Box(boxsize2, boxsize2, boxsize2);
		shape2.applyTransform(new Translation(-boxsize2 / 2, -boxsize2 / 2, -boxsize2 / 2));
		object2.setShape(shape2);
		scene.add(object2, 1);

		//		// add water boxes inside each other
		//		int numboxes = 10;
		//		for (int i = numboxes; i >= 1; --i) {
		//			PhysicalObject object2 = new PhysicalObject();
		//			object2.setMaterial(MaterialsDB.getMaterial("water"));
		//			object2.setNameString("water");
		//			float boxsize2 = i * 100.f;
		//			AbstractShape shape2 = new Box(boxsize2, boxsize2, boxsize2);
		//			shape2.applyTransform(new Translation(-boxsize2 / 2, -boxsize2 / 2, -boxsize2 / 2));
		//			object2.setShape(shape2);
		//			scene.add(object2, numboxes + 1 - i);
		//		}

		//add a piece of lead
		//		 PhysicalObject lead = new PhysicalObject();
		//		 lead.setMaterial(MaterialsDB.getMaterial("lead"));
		//		 lead.setNameString("air");
		//		 float boxsize2 = 200.f;
		//		 AbstractShape shape2 = new Box(boxsize2, boxsize2, boxsize2);
		//		 shape2.applyTransform(new Translation(100, -boxsize2 / 2, -boxsize2 / 2));
		//		 lead.setShape(shape2);
		//		 scene.add(lead, 1);

		return scene;
	}

	public static double computeMean(ArrayList<Double> array) {
		double mean = 0;
		for (int i = 0; i < array.size(); i++) {
			mean += array.get(i);
		}
		return mean / array.size();
	}

	public static double computeStddev(ArrayList<Double> array, double mean) {
		double stddev = 0;
		for (int i = 0; i < array.size(); i++) {
			stddev += Math.pow(array.get(i) - mean, 2);
		}
		return Math.sqrt(stddev / array.size());
	}

	private void showCurrentGrid(Trajectory traj, ImagePlus imp, Grid2D grid, String sceneName, long startTime) {
		if (TEST_MODE)
			return;

		long numberOfRays = totalRayCount();
		long detectorHits = totalDetectorHits();
		int pixels = traj.getDetectorWidth() * traj.getDetectorHeight();
		String title = sceneName + ", Energy: " + startEnergyEV / 1000.f + " keV." + ", Rays: " + numberOfRays
				+ ", Hits/Pixel: " + (float) (detectorHits * 100 / pixels) / 100 + ", Rays hit detector: "
				+ (double) (detectorHits * 100 / numberOfRays) + "%, Time: "
				+ ((System.currentTimeMillis() - startTime) / (1000 * 60)) + "m "
				+ ((System.currentTimeMillis() - startTime) / 1000) % 60 + "s";

		ImageProcessor image = ImageUtil.wrapGrid2D(grid);

		ImageStack stack = new ImageStack(image.getWidth(), image.getHeight());
		stack.addSlice(title, image);
		imp.setStack(title, stack);
		imp.show();

	}

	private long totalRayCount() {
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getRayCount();
		}
		return sum;
	}

	private long totalDetectorHits() {
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getDetectorHits();
		}
		return sum;
	}

	public void simulateScene(int projectionIndex, PrioritizableScene scene) {
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();

		int width = traj.getDetectorWidth();
		int height = traj.getDetectorHeight();

		Grid2D grid = detector.createDetectorGrid(width, height, traj.getProjectionMatrix(projectionIndex));

		if (!TEST_MODE) {
			//add the detector to the scene
			detector.setMaterial(MaterialsDB.getMaterial("lead"));
			detector.generateDetectorShape(traj.getProjectionMatrix(projectionIndex), 200);
			detector.setNameString("detector");
			scene.add(detector, 100000);
		}

		final PriorityRayTracer raytracer = new PriorityRayTracer();
		raytracer.setScene(scene);

		final StraightLine ray;

		if (TEST_MODE) {
			ray = new StraightLine(new PointND(0, 0, 0), new SimpleVector(1, 0, 0));
		} else {
			ray = new StraightLine(new PointND(traj.getProjectionMatrix(projectionIndex).computeCameraCenter()), traj
					.getProjectionMatrix(projectionIndex).computeRayDirection(new SimpleVector(width / 2, height / 2)));
		}

		System.out.println("Ray direction: " + ray);
		System.out.println("Raytracing scene \"" + scene.getName() + "\" with " + numRays + " rays using " + numThreads
				+ " threads. Energy: " + startEnergyEV / 1000000.f + " MeV.");

		long startTime = System.currentTimeMillis();

		boolean writeAdditionalData;
		if (numRays < 50000) {
			writeAdditionalData = true;
		} else {
			writeAdditionalData = false;
			System.err.println("Warning: Not saving additional data at interaction points because of high ray count.");
		}

		// start worker threads
		for (int i = 0; i < numThreads; ++i) {
			int numberofrays = numRays / numThreads;
			if (i == numThreads - 1)
				numberofrays += (numRays % numThreads);
			rayTraceThreads[i] = new XRayWorker(ray, raytracer, new RaytraceResult(), numberofrays, startEnergyEV, grid,
					detector, traj.getProjectionMatrix(projectionIndex), INFINITE_SIMULATION, writeAdditionalData);
			threads[i] = new Thread(rayTraceThreads[i]);
			threads[i].start();
		}

		RaytraceResult combinedResult = new RaytraceResult();

		ImagePlus imp = new ImagePlus();

		// wait for threads to finish and combine the results
		for (int i = 0; i < numThreads; ++i) {
			try {
				while (threads[i].isAlive()) {
					threads[i].join(1000);
					showCurrentGrid(traj, imp, grid, scene.getName(), startTime);
				}

				rayTraceThreads[i].addResults(combinedResult);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		showCurrentGrid(traj, imp, grid, scene.getName(), startTime);

		System.out.println("Raytracing finished in " + (System.currentTimeMillis() - startTime) / 1000.f + "s");
		System.out.println("Number of rays: " + totalRayCount());

		// print results

		double mean = combinedResult.pathlength / combinedResult.count;
		double mean2 = combinedResult.pathlength2 / combinedResult.count;
		double stddev = Math.sqrt(Math.abs(mean2 - mean * mean));
		System.out.println("pathlength = " + mean + " mm +- " + stddev + " mm");

		mean = combinedResult.x / combinedResult.count;
		mean2 = combinedResult.x2 / combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean * mean));
		System.out.println("x = " + mean + " mm +- " + stddev + " mm");

		mean = combinedResult.y / combinedResult.count;
		mean2 = combinedResult.y2 / combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean * mean));
		System.out.println("y = " + mean + " mm +- " + stddev + " mm");

		mean = combinedResult.z / combinedResult.count;
		mean2 = combinedResult.z2 / combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean * mean));
		System.out.println("z = " + mean + " mm +- " + stddev + " mm");

		// show OpenGL visualization

		//scale visualization according to distance source - detector
		XRayViewer pcv = new XRayViewer("points", combinedResult.points, null, traj.getSourceToDetectorDistance());
		pcv.setColors(combinedResult.colors);

		//		 XRayViewer pcv = new XRayViewer("points",null, combinedResult.edges);

		pcv.setScene(scene);

		if (!TEST_MODE) {
			//draw source as a simple box for now
			double sidelength = 20;
			Box sourceShape = new Box(sidelength, sidelength, sidelength);
			sourceShape.applyTransform(new Translation(-sidelength / 2, -sidelength / 2, -sidelength / 2));

			sourceShape
					.applyTransform(new Translation(traj.getProjectionMatrix(projectionIndex).computeCameraCenter()));
			pcv.setSource(sourceShape);
		}

		//visualize geant4 data with OpenGL
		//		XRayViewer pcv2 =  new XRayViewer("points Geant4","PATH/1000_1MeV_Air.csv", pcv.getMaxPoint());
		//		pcv2.setScene(scene);
	}

	public static void main(String[] args) {
		new ImageJ();

		Configuration.loadConfiguration();

		XRayTracer monteCarlo = new XRayTracer();

		//		monteCarlo.simulateScene(0, XRayTracer.constructTestScene());

		PrioritizableScene scene;
		try {
			scene = AnalyticPhantom.getCurrentPhantom();

			//add a surrounding box
			PhysicalObject surroundingBox = new PhysicalObject();
			surroundingBox.setMaterial(scene.getBackgroundMaterial());
			surroundingBox.setNameString("background material");
			float boxsize2 = 100000.f;
			AbstractShape shape2 = new Box(boxsize2, boxsize2, boxsize2);
			shape2.applyTransform(new Translation(-boxsize2 / 2, -boxsize2 / 2, -boxsize2 / 2));
			surroundingBox.setShape(shape2);
			scene.add(surroundingBox, -100000);

			//start the simulation
			monteCarlo.simulateScene(1, scene);

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
