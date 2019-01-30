/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;

import java.awt.Color;
import java.util.ArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.tutorial.physics.workingClass.*;

/**
 * A raytracer for X-rays using the PriorityRaytracer for intersection calculation. It considers only the Compton and the Photoelectric Effect, disregarding other effects which are insignificant for X-rays.
 *
 * @author Fabian R�ckert, Tobias Miksch
 *
 */
public class XRayTracer {

	//metric prefix
	public static float deci = 0.1f;
	public static float centi = 0.01f;
	public static float milli = 0.001f;
	public static int kilo = 1000;
	public static int mega = 1000000;
	public static int giga = 1000000000;	
	
	// settings
	private static final int numThreads = Runtime.getRuntime().availableProcessors();
	private static final boolean INFINITE_SIMULATION = false;
	
	private static final long numRays = 1 * mega; //(4000000000L) Long.MAX_VALUE;
	private static final int startEnergyEV = 140 * kilo;
	private static int lightRayLength = 5;
	private static int virtualLightsPerThread = 8000;
	
	//Barriers and locks
	private CyclicBarrier barrier;
//	private Object lock = new Object();
	 
	//enable to use the Geant4 code for sampling of the Compton Scattering Angle/Energy
	public static boolean SAMPLE_GEANT4 = false;
	
	//enable to just send the rays in x-direction starting from the origin, without a detector
	public static final boolean TEST_MODE = false;

	// variables
	private Thread[] threads = new Thread[numThreads];
	private AbstractWorker[] rayTraceThreads = new AbstractWorker[numThreads];
//	private XRayDetector detector = new XRayDetector();
	
	/*
	 * GUI guideline
	 */
	private static boolean renderAll = false;
	private static int tracingVersion = -1;
	public static final String PATHTRACING = "pt";
	public static final String BDPT = "bdpt";
	public static final String VPL = "vpl";
	public static final String VRL = "vrl";
	public static final String DEBUG = "debug";
	public static final String ALL = "all";

	private static boolean pickScene = true;
	private static boolean createOutPut = false;
 
	/**
	 * Saves the raytracing results for each worker thread.
	 */
	public class RaytraceResult {
		public ArrayList<PointND> points = new ArrayList<PointND>();
		public ArrayList<Color> colors = new ArrayList<Color>();
		public ArrayList<Edge> edges = new ArrayList<Edge>();
		public double pathlength = 0;
		public double pathlength2 = 0;
		public double x = 0;
		public double x2 = 0;
		public double y = 0;
		public double y2 = 0;
		public double z = 0;
		public double z2 = 0;
		public int count = 0;

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
	private static PrioritizableScene constructTestScene(int projectionIndex) {
		PrioritizableScene scene = new PrioritizableScene();
		
		int type = 0;
		
		if(type == -1) {
			scene.setName("My_Head");
			return scene;
		}else if(type == 0) {
			Cylinder mainCylinder = new Cylinder(50, 50, 200);
			
			mainCylinder.applyTransform(new ScaleRotate(Rotations.createBasicYRotationMatrix( (Math.PI / 2.0) ) ) );
			PhysicalObject cylinder = new PhysicalObject();
			cylinder.setNameString("TestObject0");
			cylinder.setMaterial(MaterialsDB.getMaterial("bone"));

			cylinder.setShape(mainCylinder);
			scene.add(cylinder);
			scene.setName("BoneCylinder");
		} else if(type == 1) {
			PhysicalObject testSubjectGLaDOS = new PhysicalObject();
			testSubjectGLaDOS.setMaterial(MaterialsDB.getMaterial("Al"));
			testSubjectGLaDOS.setNameString("Al");
			float boxsizeMM = 50.f;
			AbstractShape shape = new Box(boxsizeMM, boxsizeMM, boxsizeMM);
			shape.applyTransform(new Translation(-boxsizeMM / 2, -boxsizeMM / 2, -boxsizeMM / 2));
			testSubjectGLaDOS.setShape(shape);
			scene.add(testSubjectGLaDOS);
			scene.setName("GLaDOS");
		} else if (type == 2) {
			Sphere bead1 = new Sphere(100.0);
			bead1.applyTransform(new Translation(new SimpleVector(0, 0, 0)));
			PhysicalObject p0 = new PhysicalObject();
			 p0.setMaterial(MaterialsDB.getMaterial("water"));
			//p0.setMaterial(MaterialsDB.getMaterial("Al"));// D = 2.99
			p0.setShape(bead1);
			scene.add(p0);
			scene.setName("Just_a_Sphere");
		} else {

		//Verschiebung: x,y,z | x in Richtung der roten Achse, y = grüne Achse, z = blaue Achse
		
//		vacuum with the values: 0.0 and 0.0
//		air with the values:    2.185627007968014E-5 and 0.0017384283309110605
//		water with the values:  0.001137209970821031 and 0.1503751480138633
//		bone with the values:   0.03474922640205473  and 0.2558683914744022
//		aluminium with values:  0.02072087999999999  and 0.34862400000000004
				
		Cylinder mainCylinder = new Cylinder(50, 50, 200);
		
		mainCylinder.applyTransform(new ScaleRotate(Rotations.createBasicYRotationMatrix( (Math.PI / 2.0) ) ) );
		//mainCylinder.applyTransform(new Translation(new SimpleVector(0,0,0)));
		
		PhysicalObject cylinder = new PhysicalObject();
		cylinder.setMaterial(MaterialsDB.getMaterial("bone"));

		cylinder.setShape(mainCylinder);
		scene.add(cylinder);
		}
		
		//add a surrounding box
		PhysicalObject surroundingBox = new PhysicalObject();
		surroundingBox.setMaterial(scene.getBackgroundMaterial());
		surroundingBox.setNameString("background material");
		float boxsize2 = 100000.f;
		AbstractShape shape2 = new Box(boxsize2, boxsize2, boxsize2);
		shape2.applyTransform(new Translation(-boxsize2 / 2, -boxsize2 / 2, -boxsize2 / 2));
		surroundingBox.setShape(shape2);
		scene.add(surroundingBox, -100000);
		
		//Erzeuge ein Objekt Camera
		PhysicalObject camera = new PhysicalObject();
		camera.setMaterial(MaterialsDB.getMaterial("vacuum"));
		camera.setNameString("photonEmitter");
		float camSize = 20.f;
//		camera.setTotalAbsorbedEnergy(startEnergyEV);
		AbstractShape camShape = new Box(camSize, camSize, camSize);
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		
		camShape.applyTransform(new Translation(traj.getProjectionMatrix(projectionIndex).computeCameraCenter()));
		camShape.applyTransform(new Translation(-camSize / 2, -camSize / 2, -camSize / 2));
//		camera.getMaterial().setDensity(startEnergyEV);
		camera.setShape(camShape);
		scene.add(camera);
		
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
	
	private void showCurrentGrid(Trajectory traj, ImagePlus imp, Grid2D grid, String sceneName, long startTime, String operation){
		if (TEST_MODE)return;
		
		long numberOfRays = totalRayCount();
		if(numberOfRays <= 0) {
//			System.err.println("Number of rays is: " + numberOfRays);
			numberOfRays = 1; 
		}
		long detectorHits = totalDetectorHits();
		int pixels =  traj.getDetectorWidth()*traj.getDetectorHeight();
		String title = sceneName + ", Version: " + operation
				+ ", Rays: " + numberOfRays
				+ ", Hits/Pixel: " + (float) (detectorHits* 100/pixels)/ 100
				+ ", Rays hit detector: " + (double) (detectorHits* 100/numberOfRays)
				+ "%, Time: " + ((System.currentTimeMillis() - startTime) / (1000 * 60)) +"m " + ((System.currentTimeMillis() - startTime) / 1000) % 60 + "s";
	
		ImageProcessor image = ImageUtil.wrapGrid2D(grid);

		ImageStack stack = new ImageStack(image.getWidth(),image.getHeight());
		stack.addSlice(title, image);
		imp.setStack(title, stack);
		imp.show();
	}
	
	private long totalRayCount(){
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getRayCount();
		}
		return sum;
	}
	
	private long totalSegmentsTraced(){
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getSegmentsTraced();
		}
		return sum;
	}

	private long totalDetectorHits(){
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getDetectorHits();
		}
		return sum;
	}
	
	private long totalLightSourceAttempts(){
		long sum = 0;
		for (int i = 0; i < numThreads; ++i) {
			sum += rayTraceThreads[i].getVLSAttempts();
		}
		return sum;
	}
	
	private int[] totalScatteringNumber() {
		int sum[] = new int[50];
		
		for(int i = 0; i < numThreads; ++i) {
			int ret[] = rayTraceThreads[i].getScatteringNumber();
			
			if(sum.length != ret.length) {
				System.err.println("Length of the data gathering arrays is not equal! " +  i);
				throw new RuntimeException("sum != ret");
			}
				
			for(int j = 0; j < ret.length; ++j) {
				sum[j] += ret[j];
			}
		}
		return sum;
	}
	
	private double[] totalContributionNumber() {
		double sum[] = new double[50];

		for (int i = 0; i < numThreads; ++i) {
			double ret[] = rayTraceThreads[i].getContributionNumber();
			if (sum.length != ret.length) {
				System.err.println("Length of the data gathering arrays is not equal! " + i);
				throw new RuntimeException("sum != ret");
			}

			for (int j = 0; j < ret.length; ++j) {
				sum[j] += ret[j];
			}
		}

		return sum;
	}
	

	public void simulateScene(int projectionIndex, PrioritizableScene scene, String operation) {
		XRayDetector detector = new XRayDetector();
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
       
		int width = traj.getDetectorWidth();
		int height = traj.getDetectorHeight();

		Grid2D grid = detector.createDetectorGrid(width, height, traj.getProjectionMatrix(projectionIndex));
		
		if (!TEST_MODE){
			//add the detector to the scene
			detector.setMaterial(MaterialsDB.getMaterial("lead"));
			detector.generateDetectorShape( traj.getProjectionMatrix(projectionIndex), 200);
			detector.setNameString("detector");
			scene.add(detector, 100000);	
		}
		
		Material background = scene.getBackgroundMaterial();

		final PriorityRayTracer raytracer = new PriorityRayTracer();
		raytracer.setScene(scene);
		
		double sourceDetectorDistance = traj.getSourceToDetectorDistance();
		double pixelDimX = traj.getPixelDimensionX();
		double pixelDimY = traj.getPixelDimensionY();
		
		
		//System.out.println("CamCenter: " + camCenter);
		System.out.println("Raytracing scene \"" + scene.getName() + "\" using " + operation +  " with " + numRays + " rays using " + numThreads + " threads. Energy: " + startEnergyEV / 1000000.f + " MeV.");
//		System.out.println("Detector-ausmaß: " + width + " : " + height);
						
		long startTime = System.currentTimeMillis();
		
		boolean writeAdditionalData = true;
//		if (numRays <= 50000){
//			writeAdditionalData = true;
//		} else {
//			writeAdditionalData = false;
//			System.out.println("Warning: Not saving additional data at interaction points because of high ray count.");
//		}

		//Cyclic Barrier
		barrier = new CyclicBarrier(numThreads);

		if(PATHTRACING.equals(operation)) {
			tracingVersion = 0;
		} else if(BDPT.equals(operation)) {
			tracingVersion = 1;
		} else if(VPL.equals(operation)) {
			tracingVersion = 2;
		} else if(VRL.equals(operation)) {
			tracingVersion = 3;
		} else if(DEBUG.equals(operation)) {
			tracingVersion += 100;
		}
		
		XRayVPL collection[] 	= new XRayVPL[virtualLightsPerThread * numThreads];
		XRayVRay vrlColl[] 		= new XRayVRay[virtualLightsPerThread * numThreads];
		int attempting[] 		= new int[numThreads]; 

		System.out.println();
		// start worker threads
		for (int i = 0; i < numThreads; ++i) {
			long numberofrays = numRays / numThreads;
			if (i == numThreads - 1)
				numberofrays += (numRays % numThreads); 
			
			if (renderAll) {
				if (BDPT.equals(operation)) {
					numberofrays /= 1.56; // 1.8; //1.56 -> avg 1.68
				} else if (VPL.equals(operation)) {
					numberofrays /= 3.23; // 4.2; //3.23 -> 3.7
				} else if (VRL.equals(operation)) {
					numberofrays /= 12.36; // 16.0; //12.36 -> 14.2
				}
			}
			
			if (tracingVersion == 0 || tracingVersion == 1) {
				rayTraceThreads[i] = new rayWorker(raytracer, new RaytraceResult(), numberofrays, startEnergyEV, grid,
						detector, traj.getProjectionMatrix(projectionIndex), INFINITE_SIMULATION, writeAdditionalData,
						sourceDetectorDistance, pixelDimX, pixelDimY, background, tracingVersion, i, null, lightRayLength);
			} else if (tracingVersion == 2) {
				rayTraceThreads[i] = new virtualPointWorker(raytracer, new RaytraceResult(), numberofrays,
						startEnergyEV, grid, detector, traj.getProjectionMatrix(projectionIndex), INFINITE_SIMULATION,
						writeAdditionalData, sourceDetectorDistance, pixelDimX, pixelDimY, background, tracingVersion,
						i, null, lightRayLength, virtualLightsPerThread, collection, attempting, barrier);
			} else if (tracingVersion == 3) {
				rayTraceThreads[i] = new virtualRayWorker(raytracer, new RaytraceResult(), numberofrays,
						startEnergyEV, grid, detector, traj.getProjectionMatrix(projectionIndex), INFINITE_SIMULATION,
						writeAdditionalData, sourceDetectorDistance, pixelDimX, pixelDimY, background, tracingVersion,
						i, null, lightRayLength, virtualLightsPerThread, vrlColl, attempting, barrier);
			} else if(tracingVersion >= 100) {
				rayTraceThreads[i] = new TestControlUnit(raytracer, new RaytraceResult(), numberofrays,
						startEnergyEV, grid, detector, traj.getProjectionMatrix(projectionIndex), INFINITE_SIMULATION,
						writeAdditionalData, sourceDetectorDistance, pixelDimX, pixelDimY, background, tracingVersion,
						i, null, lightRayLength, virtualLightsPerThread, collection, attempting, barrier);
			}
		
			threads[i] = new Thread(rayTraceThreads[i]);
			threads[i].start();
		}
		
		RaytraceResult combinedResult = new RaytraceResult();
		ImagePlus imp = new ImagePlus();
		
		
//		long time = TimeUnit.HOURS.toMillis(4);
//		long time = TimeUnit.MINUTES.toMillis(20);
		
		// wait for threads to finish and combine the results
//		try {
//			threads[0].join(time);
//		} catch (InterruptedException e1) {
//			e1.printStackTrace();
//		}
		
//		for (int i = 0; i < numThreads; ++i) {
//			if (threads[i].isAlive()) threads[i].interrupt();
//			rayTraceThreads[i].addResults(combinedResult);
//			
//			
//			try {
//				while(threads[i].isAlive()){
//					threads[i].join(1000);
////					showCurrentGrid(traj, imp, grid, scene.getName(), startTime, operation);
//				}
//				
//				rayTraceThreads[i].addResults(combinedResult);
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
//		}
		
		for (int i = 0; i < numThreads; ++i) {
			try {
				while(threads[i].isAlive()){
					threads[i].join(1000);
//					showCurrentGrid(traj, imp, grid, scene.getName(), startTime, operation);
				}

				rayTraceThreads[i].addResults(combinedResult);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		long endTime = System.currentTimeMillis();
		
		System.out.println();
		long rayCount = totalRayCount();
		
		double energySum = XRayAnalytics.totalEnergyCount(grid);
		System.out.println("EnergySum is: " + energySum / rayCount);
		
		showCurrentGrid(traj, imp, grid, scene.getName(), startTime, operation);
		
		double effectivity = (double) totalDetectorHits() / (rayCount + totalLightSourceAttempts()) * 100.0;
		double t = (endTime - startTime) / 1000.f;
		long segments = totalSegmentsTraced();
		
		System.out.println("Raytracing finished in " + t + "s");
		System.out.println("Number of rays: " + rayCount); 
		System.out.println("With the total number of detector hits being: " + totalDetectorHits());
		System.out.println("Number of traced segments: " + segments);
		System.out.println("Rays traced for the creation of light sources: " + totalLightSourceAttempts());
		System.out.println("The effectivity of the rays is at: " + effectivity + "%");
		//System.out.println("The detector did absorb: " + detector.getParent().getTotalAbsorbedEnergy());

		System.out.println("Ray factor: " +  (rayCount / t) );
		System.out.println("Segment factor: " + segments / t);
		
		int[] inscatteringNumber = totalScatteringNumber();
		double[] contrib = totalContributionNumber();
		

			System.out.println("\n--------");
			for (int p = 0; p < inscatteringNumber.length; ++p) {
				if(contrib[p] != 0.0 && inscatteringNumber[p] != 0) {
					System.out.println("Energy of scattering " + p + " is: " + String.format("%.2f", contrib[p] / rayCount) + " & " + (double)inscatteringNumber[p] / rayCount);
					continue;
				}
				
				if(contrib[p] != 0.0)
					System.out.println("Energy of version " +  p + " is: " + String.format("%.2f", contrib[p] / rayCount));
				
				if(inscatteringNumber[p] != 0)
					System.out.println("ScatteringNumber of version " +  p + " is: " + inscatteringNumber[p]
							+" percentage: " + (double)inscatteringNumber[p] / rayCount);
			}
			System.out.println("--------\n");
		
		System.out.println("Total number of valid paths: " + (inscatteringNumber[1] + inscatteringNumber[2]) + " result: " + (double)(inscatteringNumber[1] + inscatteringNumber[2]) / rayCount);
		System.out.println("Number of not absorbed paths: " +  inscatteringNumber[2] + " scaled: " + (double)inscatteringNumber[2] / rayCount);
/*		
		// print results
		double mean = combinedResult.pathlength/combinedResult.count;
		double mean2 = combinedResult.pathlength2/combinedResult.count;
		double stddev = Math.sqrt(Math.abs(mean2 - mean*mean));
		System.out.println("pathlength = " + mean + " mm +- " + stddev + " mm");


		mean = combinedResult.x/combinedResult.count;
		mean2 = combinedResult.x2/combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean*mean));
		System.out.println("x = " + mean + " mm +- " + stddev + " mm");

		mean = combinedResult.y/combinedResult.count;
		mean2 = combinedResult.y2/combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean*mean));
		System.out.println("y = " + mean + " mm +- " + stddev + " mm");

		mean = combinedResult.z/combinedResult.count;
		mean2 = combinedResult.z2/combinedResult.count;
		stddev = Math.sqrt(Math.abs(mean2 - mean*mean));
		System.out.println("z = " + mean + " mm +- " + stddev + " mm");
*/

		//scale visualization according to distance source - detector
		XRayViewer pcv = new XRayViewer("points", combinedResult.points, null, traj.getSourceToDetectorDistance());
		pcv.setColors(combinedResult.colors);

//		 XRayViewer pcv = new XRayViewer("points",null, combinedResult.edges);
		PrioritizableScene emptyScene = new PrioritizableScene();
		if (!TEST_MODE){
			//add the detector to the scene
			emptyScene.add(detector, 100000);	
		}
		pcv.setScene(scene); //TODO change here for different vizualization
		//pcv.setScene(emptyScene);
		

		if(createOutPut) {
			System.out.println(System.getProperty("user.dir"));
			
			String title = scene.getName() + "_" + operation + "_rc_" + rayCount + "_dh_" + totalDetectorHits()+ "_spt_" + totalSegmentsTraced() + "_t_" + Math.floor((endTime - startTime) / 1000.f) + "s";
			if(VPL.equals(operation) || VRL.equals(operation)) {
				title += "_" + (virtualLightsPerThread*numThreads);
			}		
			XRayAnalytics.saveFile(grid, title);
		}

		//visualize geant4 data with OpenGL
//		XRayViewer pcv2 =  new XRayViewer("points Geant4","PATH/1000_1MeV_Air.csv", pcv.getMaxPoint());
//		pcv2.setScene(scene);
	}
	
	public static void main(String[] args) {
		
		new ImageJ();
		Configuration.loadConfiguration();
		XRayTracer monteCarlo = new XRayTracer();
		
		PrioritizableScene scene;
		String operation = null;
		
		try {
			int projectionIndex = 1;
			
			// Enable for standart AnalyticPhantoms.
//			pickScene = UserUtil.queryBoolean("Want to use default cylinder scene?");

			{
				String [] operations = {VRL, VPL, BDPT, PATHTRACING, DEBUG};
				operation = (String) UserUtil.chooseObject("Select tracing version: ", "Energy transport", operations, operation);
			}
			if( operation == DEBUG)
				tracingVersion = 1;
			
			createOutPut = UserUtil.queryBoolean("Create output file?"); // false;//
			
			boolean pickFileLocation = true;
			if(createOutPut && pickFileLocation) {
				String path_location = XRayAnalytics.getPathToDir();
				String filePath = UserUtil.queryString("File path", path_location);
			}
			
			
			//start the simulation
			if(pickScene) {
				scene = XRayTracer.constructTestScene(projectionIndex);
			} else {
				scene = AnalyticPhantom.getCurrentPhantom();
			}
			monteCarlo.simulateScene(projectionIndex, scene, operation);

			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
