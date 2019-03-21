/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics.workingClass;

import java.awt.Color;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ThreadLocalRandom;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.bounds.BoundingBox;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.tutorial.physics.XRayHitVertex;
import edu.stanford.rsl.tutorial.physics.XRayTracer;
import edu.stanford.rsl.tutorial.physics.XRayTracerSampling;
import edu.stanford.rsl.tutorial.physics.XRayVPL;
import edu.stanford.rsl.tutorial.physics.nHelperFkt;
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;


/**
 * Different test cases and visualizations of the methods used by the classes of this package
 * @author Tobias Miksch
 */
public class TestControlUnit extends AbstractWorker {

	rayWorker rayTracer;
	virtualPointWorker vplTracer;
	
	private volatile int attempting[];
	private int vplPathNumber = 0;
	
	private final CyclicBarrier barrierVPL;
	private volatile XRayVPL collectionOfAllVPLs[];

	public TestControlUnit(PriorityRayTracer raytracer, RaytraceResult res, long numRays, int startenergyEV, Grid2D grid,
			XRayDetector detector, Projection proj, boolean infinite, boolean writeAdditionalData,
			double sourceDetectorDist, double pixelDimensionX, double pixelDimensionY, Material background,
			int versionNumber, int threadNumber, Random random, int lightRayLength,
			int virtualLightsPerThread, XRayVPL collection[], int attempting[], CyclicBarrier barrier) {
		super(raytracer, res, numRays, startenergyEV, grid, detector, proj, infinite, writeAdditionalData,
				sourceDetectorDist, pixelDimensionX, pixelDimensionY, background, versionNumber, threadNumber, random, lightRayLength);
		// TODO Auto-generated constructor stub
		
		this.virtualLightsPerThread = virtualLightsPerThread;
		this.collectionOfAllVPLs = collection;
		this.attempting = attempting;
		this.barrierVPL = barrier;
	}
	
	
	@Override
	public void run() {
		if (random == null) {
			sampler = new XRayTracerSampling(ThreadLocalRandom.current(), (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		} else {
			sampler = new XRayTracerSampling(random, (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		}
		
		rayTracer = new rayWorker((PriorityRayTracer) raytracer, result, numRays, startEnergyEV, grid, detector, proj, infiniteSimulation, writeAdditionalData, sourceDetectorDistance, pixelDimX, pixelDimY, background, version, threadNumber, random, lightRayLength);
		vplTracer = new virtualPointWorker((PriorityRayTracer) raytracer, result, numRays, startEnergyEV, grid, detector, proj, infiniteSimulation, writeAdditionalData, sourceDetectorDistance, pixelDimX, pixelDimY, background, version, threadNumber, random, lightRayLength, virtualLightsPerThread, collectionOfAllVPLs, attempting, barrierVPL);
		
		double energyEV = startEnergyEV;

//		if(true) {
//			testMaterial();
//			return;
//		}
		if (testMaterial) {

			if (threadNumber == 0)
				System.out.println("Working with debug functions!");

			boolean test_camera_rays = false;
			boolean test_Angle_with_DAT = false;
			boolean test_ScatterAngle = false;
			boolean transmittanceTesting = false;
			boolean test_camRays = false;
			boolean test_photometry = false;
			boolean testFirstPathVertex = false;
			boolean testNumberOfBDPTConnections = false;
			boolean vizShadowRay = false;
			boolean vizDetectorOutline = true;

			double max = 0.0;
			double maxAngle = 0.0;
			double min = 8.0;
			double minAngle = 0.0;
			double angle = 0.0;
			double phaseFkt = 0.0;

			if(test_camera_rays) {
				testCameraRaysL1();
				return;
			}
			
			if (test_Angle_with_DAT) {
				boolean test_Energy_Behaviour = false;
				boolean print_comptonnagleCrossSection = false;
				boolean print_getComptonAngleTheta = false;

				if (test_Energy_Behaviour) {
					for (int i = 0; i < numRays; ++i) {
						// angle between 0 and 360 deg
						// randomAngle = sampler.random() * 2 * Math.PI;
						double randEnergyEV = sampler.random() * startEnergyEV * 10;
						phaseFkt = XRayTracerSampling.comptonAngleCrossSection(randEnergyEV,
								((1.0 - 0.0000001) * Math.PI));
						// if(randomAngle > Math.toRadians(179.99f) || randomAngle <
						// Math.toRadians(0.01f)) {
						// i--;
						// continue;
						// }
						if (phaseFkt > max) {
							max = phaseFkt;
							maxAngle = randEnergyEV; // nHelperFkt.transformToScatterAngle(randomAngle);
						}
						if (phaseFkt < min) {
							min = phaseFkt;
							minAngle = randEnergyEV; // nHelperFkt.transformToScatterAngle(randomAngle);
						}
					}
					System.out.println("\nMax: " + max + " with an energy of:  " + maxAngle + "\nMin: " + min
							+ " with an energy of: " + minAngle);
				}

				//
				if (print_comptonnagleCrossSection) {
					System.out.println("X Y");
					int sampleSize = 50;
					for (int j = 0; j <= sampleSize; ++j) {
						angle = Math.PI * j / sampleSize;
						// double energyDist = startEnergyEV * j / sampleSize;
						phaseFkt = XRayTracerSampling.comptonAngleCrossSection(startEnergyEV, angle);
						// System.out.println("Angle: " +
						// Math.toDegrees(nHelperFkt.transformToScatterAngle(Math.toRadians(angle))) + "
						// has a energy of: " + phaseFkt);
						System.out.printf("  " + angle + "          " + phaseFkt + "\n");
					}
				}

				//
				if (print_getComptonAngleTheta) {
					int numberOfangles[] = new int[180];
					for (int p = 0; p < 180; p++) {
						numberOfangles[p] = 0;
					}
					int count = 5000000;
					for (int i = 0; i < count; ++i) {
						numberOfangles[(int) Math.toDegrees(sampler.getComptonAngleTheta(startEnergyEV))] += 1;
					}
					// count /= 10;
					for (int p = 179; p >= 0; --p) {
						System.out.println(
								" " + Math.toRadians(p) + "        " + ((double) numberOfangles[p] / (double) count));
					}
				}
			}
			//
			if (test_ScatterAngle) {
				testScatteringAngle(null);
			}

			//
			if (transmittanceTesting) {
				testTransmittance();
			}

			if (test_camRays) {
				boolean vizFromAllPoints = false;
				if (vizFromAllPoints) {
					System.out.println("We try to create some Camera Rays starting from the detector into the scene.");
					// visualization of the camRays
					for (simulatedRays = 0; simulatedRays < 1000; ++simulatedRays) {
						// Setup Cam-Ray
						XRayHitVertex camPathVertex = createCamRay(energyEV, 0.4);
						vizualizeRay(result, camPathVertex.getStartPoint(), camPathVertex.getEndPoint(), Color.BLUE, 20);
					}
				} else {
					XRayHitVertex camPathVertex = null;
					int effort = 0;
					int success = 0;
					
					SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
					PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
							sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

					if(threadNumber == 0)
						vizualizeRay(result, new PointND(0.0, 0.0, 0.0), randomDetectorPoint, Color.magenta, 20);
					
					int attempts = 1000;
					for (int i = 0; i < attempts; ++i) {
						StraightLine camRay = new StraightLine(randomDetectorPoint, new PointND(0.0, 0.0, 0.0));
						camRay.normalize();

						SimpleVector ret = sampler.samplingDirection(camRay.getDirection(), 0.1);
						camRay.setDirection(ret);
						camRay.normalize();
						
						camPathVertex = tryToSetUpCamRay(camRay, energyEV, null);
						effort++;

						// try until there is a valid camRay
						if (camPathVertex != null) {
							// System.out.println("Testing CamPath: " +
							// camPathVertex.getCurrentLineSegment().getNameString());
							camPathVertex.setDistanceFromStartingPoint(effort);
							success++;
							vizualizeRay(result, camRay.getPoint(), camPathVertex.getEndPoint(), Color.cyan, 10);
						} else {
							vizualizeRay(result, camRay.getPoint(), camRay.evaluate(400.0), Color.orange, 4);
						}
					}
					System.out.println("Probability to create a valid point is: " + (double) success / effort);
				}
				return;
			}

			// Test to cast a ray from the emitter to a point on the detector
			if (vizDetectorOutline) {
				testVizualisierungDetectorOutline();
			}

			if (vizShadowRay) {
				testShadowRayVizualisierung();
			}

			if (test_photometry) {
				// True == sampling using compton scattering | false == sampling evenly on unit
				// sphere
				boolean sampling = false;
				testPhotometry(sampling);
				return;
			}

			// TODO: Debugging is here
			if (testFirstPathVertex) {
				boolean fixPointOnly = false;
				boolean onlyFirstHit = false;

				if (fixPointOnly) {
					comparePTFirst();
				} else if (onlyFirstHit) {
					// System.out.println("Debugging only one scatter event");
					universalFirstCon();
				} else {
					compareVersions();
				}
			}

			if(testNumberOfBDPTConnections) {
				boolean testVPL = false;
				if(testVPL) {
					bdptASvpl();
				} else {
					testBDPTNumbers();
				}
			}
			
			return;
		}

		// Test XRayHitVertex
		if (testHitVertex) {
			for (simulatedRays = 0; simulatedRays < 50; ++simulatedRays) {
				StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(
						new SimpleVector(sampler.random() * grid.getWidth(), sampler.random() * grid.getHeight())));
				XRayHitVertex current = nextRayPoint(ray, startEnergyEV);
				// String name = current.getCurrentLineSegment().getNameString();

				System.out.println("Combination: " + current.getCurrentLineSegment().getNameString() + " && "
						+ current.getCurrentLineSegment().getMaterial().getName());
				writeStepResults(result, energyEV, 0.1, current.getEndPoint(), null, Color.green);
				// if (name == null)
				{// && !"detector".equals(name) && !"background material".equals(name)) {
					System.out.println(Thread.currentThread().getId() + "\n Casting his first Ray with Start : "
							+ current.getStartPoint() + " End   : " + current.getEndPoint() + "\n RayDir: "
							+ current.getRayDir() + "\n Reached the Material: "
							+ current.getCurrentLineSegment().getMaterial().getName()
							+ "\n And has the Attenuation values " + current.getPhotoAbsorption() + " & "
							+ current.getComptonAbsorption() + "\n With an Energy Value of" + " Energy: "
							+ current.getEnergyEV() + "\n And an Density of: "
							+ current.getCurrentLineSegment().getMaterial().getDensity() + "\n");
				}
			}
			return;
		}

	}
	
	private double testScattering(double energyEV, SimpleVector dir, double theta, double phi){

		double dirx = Math.sin(theta) * Math.cos(phi);
		double diry = Math.sin(theta) * Math.sin(phi);
		double dirz = Math.cos(theta);	

		SimpleVector gamDirection1 = new SimpleVector(dirx, diry, dirz);
		gamDirection1.normalizeL2();
		
		// transform scattered photon direction vector to world coordinate system
		SimpleVector ret = nHelperFkt.transformToWorldCoordinateSystem(gamDirection1, dir);
		
		double testingTheta = nHelperFkt.getAngleInRad(dir, ret);
		if( Math.abs(theta - testingTheta) > 0.01 ) {
			System.out.println("We have different angles in out function: " + theta + " & " + testingTheta);
		}
		
		dir.setSubVecValue(0, ret);
		return XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
	}

	private double debugTransmittance(PointND startPoint, PointND endPoint, double energyEV) {
		StraightLine shadowRay = new StraightLine(startPoint, endPoint);
		PointND offSet = shadowRay.evaluate(-epsilon);
		shadowRay.setPoint(offSet);
		shadowRay.normalize();

		// check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());
		if (!bb.isSatisfiedBy(shadowRay.getPoint())) {
			System.err.println("Bounding Box does not cover point?");
			return -1.0;
		}

		ArrayList<PhysicalObject> physobjs = raytracer.castRay(shadowRay);
		if (physobjs == null) {
			// should not happen because ray is inside of the bounding box
			System.err.println("No background material set?");
			throw new RuntimeException("physobjs == null");
		}

		System.out.print("\n\nWe start with a new Object: ");
		for (int i = 0; i < physobjs.size(); ++i) {
			PhysicalObject currentLineSegment = physobjs.get(i);
			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = e.getPoint();
			PointND end = e.getEnd();

			System.out.print("\nWe iterate: " + currentLineSegment.getNameString());

			if (nHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					startPoint.getAbstractVector())) {
				System.out.print(" --->  with startPoint");
			}

			if (nHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					endPoint.getAbstractVector())) {
				System.out.print(" --->  with endPoint");
			}
		}

		return 0.0;
	}

	private void testSamplingDirections(int numberOfRays) {
		// SimpleVector randomVec = new SimpleVector(sampler.random() * width,
		// sampler.random() * height);
		System.out.println("Testing the sampling direction function!");

		boolean singlePoint = true;
		SimpleVector randomVec = new SimpleVector(0, 0);
		PointND randomDetectorPoint = new PointND();

		if (singlePoint) {
			randomVec = new SimpleVector(0.0 * width, 0.0 * height);
			randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
					sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
		}

		for (simulatedRays = 0; simulatedRays < 10000; ++simulatedRays) {
			double cumulativeEnergy = 0;

			if (!singlePoint) {
				randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
			}

			// Create a random Ray from detector to the emitter
			StraightLine camRay = new StraightLine(randomDetectorPoint, emitterPoint);
			SimpleVector ret = sampler.samplingDirection(camRay.getDirection().clone(), 0.4);

			StraightLine eval = new StraightLine(randomDetectorPoint, ret);
			eval.normalize();

			writeStepResults(result, 0.0, 0.0, eval.evaluate(700.0), null, Color.MAGENTA);
			// vizualizeRay(result, randomDetectorPoint, eval.evaluate(700.0), Color.CYAN,
			// 5);

			synchronized (grid) {
				detector.absorbPhoton(grid, randomDetectorPoint, cumulativeEnergy);
			}
			detectorHits++;
		}
	}

	private XRayHitVertex testMaterial() {
		// check if ray is inside the scene limits
		double energyEV = startEnergyEV;

		HashSet<Integer> rayHash = new HashSet<Integer>();
		Material arrayOfMat[] = new Material[200];
		int count = 0;
		SimpleVector maxX = new SimpleVector(0.0, 0.0);
		SimpleVector maxY = new SimpleVector(0.0, 0.0);

		SimpleVector vec00 = new SimpleVector(0, 0);
		PointND point00 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), vec00, sourceDetectorDistance,
				pixelDimX, pixelDimY, width, height);
		SimpleVector vec01 = new SimpleVector(width, 0);
		PointND point01 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), vec01, sourceDetectorDistance,
				pixelDimX, pixelDimY, width, height);
		SimpleVector vec10 = new SimpleVector(0, height);
		PointND point10 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), vec10, sourceDetectorDistance,
				pixelDimX, pixelDimY, width, height);

		System.out.println("Points: " + point00 + " and " + point01 + " and " + point10);

		SimpleVector simple01 = SimpleOperators.subtract(point01.getAbstractVector(), point00.getAbstractVector());
		SimpleVector simple10 = SimpleOperators.subtract(point10.getAbstractVector(), point00.getAbstractVector());

		System.out.println("Vectors: " + simple01 + " and " + simple10);

		SimpleVector normal = nHelperFkt.crossProduct3D(simple01, simple10);

		System.out.println("CrossProduct: " + normal);

		// for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
		// int sampleSize = 100;
		// for (int j = 1; j < sampleSize; ++j) {
		// boolean enought = true;
		// double dist = -1.0 / j;
		//
		// for (int p = 0; p < (numRays * 1.0 / sampleSize); ++p) {
		for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
			// SimpleVector randomVector = new SimpleVector(sampler.random() * width,
			// sampler.random() * height);
			SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);

			if (randomVector.getElement(0) <= 0.025 || randomVector.getElement(1) <= 0.025) {
				simulatedRays--;
				continue;
			}

			// StraightLine ray = new StraightLine(camCenter,
			// proj.computeRayDirection(randomVector));
			PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVector,
					sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
			StraightLine ray = new StraightLine(randomDetectorPoint, emitterPoint);

			ray.normalize();
			PointND offSet = ray.evaluate(-100.0);

			// System.out.println("Distance is: "+ dist);

			ray.setPoint(offSet);
			ray.normalize();

			BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());
			if (!bb.isSatisfiedBy(ray.getPoint())) {
				System.err.println("BoundingBox");
				return null;
			}

			ArrayList<PhysicalObject> physobjs = raytracer.castRay(ray);
			if (physobjs == null) {
				// should not happen because ray is inside of the bounding box
				System.err.println("No background material set?");
				throw new RuntimeException("physobjs == null");
			}

			// check direction of the line segments
			boolean reversed = false;
			boolean reversedTesting = false;

			// Edge e1 = (Edge) physobjs.get(0).getShape();
			// SimpleVector diff = e1.getEnd().getAbstractVector().clone();
			// diff.subtract(e1.getPoint().getAbstractVector());
			// diff.normalizeL2();
			// diff.subtract(ray.getDirection());
			//
			// if (diff.normL2() > CONRAD.SMALL_VALUE) {
			// reversedTesting = true;
			// }

//			double totalDistUntilNextInteraction = 0;
			Material currentMaterial = null;
//			PointND rayPoint = ray.getPoint().clone();
//
//			// pass through materials until an interaction is happening
//			PointND start = null;
//			PointND end = null;
//			double photoAbsorption = 0;
//			double comptonAbsorption = 0;
//			boolean foundStartSegment = false;

			PhysicalObject currentLineSegment = null;
			// the line segments are sorted, find the first line segment containing the
			// point
			System.out.println("");

			boolean first = true;
			for (int i = 0; i < physobjs.size(); ++i) {
				currentLineSegment = physobjs.get(i);

				currentMaterial = currentLineSegment.getMaterial();
				
				double photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.PHOTOELECTRIC_ABSORPTION);
				double comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.INCOHERENT_ATTENUATION);
				
				System.out.println("CurrentLineSegment: " + currentLineSegment.getNameString() + " with material: " + currentMaterial.getName() + "\n and has p: " + photoAbsorption + " & c: " + comptonAbsorption);

				if (first && "detector".equals(currentLineSegment.getNameString())) {
					System.out.println("The direction is not working right: " + !(reversedTesting));
				}

				if (first && "photonEmitter".equals(currentLineSegment.getNameString())) {
					first = false;
				}

				// if (first && "photonEmitter".equals(currentLineSegment.getNameString())) {
				// System.out.println("The direction is not working right: " + randomVector);
				////
				//// for (int k = 0; k < physobjs.size(); ++k) {
				//// System.out.println("CurrentLineSegment: " +
				// physobjs.get(k).getNameString());
				//// }
				// if(randomVector.getElement(0) < 0.1 && randomVector.getElement(0) >=
				// maxX.getElement(0)) {
				// maxX = new SimpleVector(randomVector);
				// }
				// if(randomVector.getElement(1) < 0.1 && randomVector.getElement(1) >=
				// maxY.getElement(0)) {
				// maxY = new SimpleVector(randomVector);
				// }
				//
				//// System.out.println("");
				// vizualizeRay(result, offSet, randomDetectorPoint, Color.blue, 20);
				// //enought = false;
				// break;
				// }
				//
				// if (first && "detector".equals(currentLineSegment.getNameString())) {
				// first = false;
				// }
				// if(i == physobjs.size()-1) {
				// writeStepResults(result, 0.0, 0.0, randomDetectorPoint, null, Color.red);
				// }
				//
				// if("glass".equals(currentMaterial.getName())) {
				// Edge e = (Edge) currentLineSegment.getShape();
				// start = reversed ? e.getEnd() : e.getPoint();
				// end = reversed ? e.getPoint() : e.getEnd();
				//
				// vizualizeRay(result, start, end, Color.BLUE, 5);
				// }
				// if("aluminium".equals(currentMaterial.getName())) {
				// Edge e = (Edge) currentLineSegment.getShape();
				// start = reversed ? e.getEnd() : e.getPoint();
				// end = reversed ? e.getPoint() : e.getEnd();
				//
				// vizualizeRay(result, start, end, Color.RED, 5);
				// }

				// photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
				// AttenuationType.PHOTOELECTRIC_ABSORPTION);
				// comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
				// AttenuationType.INCOHERENT_ATTENUATION);
				// currentMaterial.getDensity();

				// int hashNumber = currentLineSegment.getMaterial().hashCode();
				// if(!rayHash.contains(hashNumber)) {
				// rayHash.add(hashNumber);
				// arrayOfMat[count] = currentLineSegment.getMaterial();
				// count++;
				// } else {
				// //Mat allready known
				// continue;
				// }

			}
		}

		// System.out.println("\n------------------------------------\nThe Sceen contain
		// the following Materials: \n");
		// for(int k = 0; k < count; ++k) {
		// Material currentMaterial = arrayOfMat[k];
		//
		// double photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
		// AttenuationType.PHOTOELECTRIC_ABSORPTION);
		// double comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
		// AttenuationType.INCOHERENT_ATTENUATION);
		//
		// System.out.println(currentMaterial.getName() + " with the values: " +
		// photoAbsorption + " and " + comptonAbsorption);
		// }
		// if (!enought)
		// System.out.println("The distance was not enought: " + dist);
		// }

		System.out.println("\n\n The maximum was: " + maxX + "\n and: " + maxY);
		return null;
	}

	public void testTransmittance() {

		double energyEV = startEnergyEV;

		boolean test_Throught = false;
		boolean test_Transmittance = false;
		boolean test_partial_Transmittance = true;

		if (test_Throught) {
			Material currentMaterial = MaterialsDB.getMaterial("bone");

			double photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
					AttenuationType.PHOTOELECTRIC_ABSORPTION);
			double comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
					AttenuationType.INCOHERENT_ATTENUATION);
			double distToNextMaterial = 0;

			int sampleSize = 1000000;
			int count;
			int times = 10;
			for (int i = 0; i < times + 1; ++i) {
				count = 0;
				distToNextMaterial = 100.0f * i / times;

				for (int p = 0; p < sampleSize; ++p) {
					double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption,
							comptonAbsorption);
					if (distToNextInteraction > distToNextMaterial) {
						count++;
					}
				}
				double trans = XRayTracerSampling.getTransmittanceOfMaterial(photoAbsorption, comptonAbsorption,
						distToNextMaterial);
				double troughtput = Math.exp(-trans);
				// System.out.println("Lenght of Material is: " + distToNextMaterial +
				// "\nCompare: " + String.format("%.6f", ((double)count / (double)sampleSize)) +
				// " with a: " + String.format("%.6f", troughtput));
				System.out.println("Compare: " + String.format("%.8f", ((double) count / (double) sampleSize))
						+ "\nwith a:  " + String.format("%.8f", troughtput) + "\n");
			}
		}

		if (test_Transmittance) {
			for (int a = 0; a < 10; ++a) {
				// SimpleVector randomVec = new SimpleVector(0.5 * width, 0.5 * height);
				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine ray = new StraightLine(emitterPoint, randomDetectorPoint);

				// System.out.println("Length: " + ray.getDirection().normL2());
				// PointND direction01 = ray.evaluate(1.0);
				// ray.normalize();
				// PointND direction10 = ray.evaluate(-100.0);
				// System.out.println("Length: " + ray.getDirection().normL2());
				//
				// vizualizeRay(result, ray.getPoint(), direction01, Color.magenta, 100);
				// vizualizeRay(result, ray.getPoint(), direction10, Color.blue, 10);

				int sampleSize = 200000;
				int count = 0;
				XRayHitVertex vertexHit;

				for (int p = 0; p < sampleSize; ++p) {

					if (p == 10000 && (count == 0 || count == p)) {
						randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
						randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
								sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
						ray = new StraightLine(emitterPoint, randomDetectorPoint);
						count = 0;
						p = 0;
					}

					vertexHit = nextRayPoint((StraightLine) ray.clone(), energyEV);
					// System.out.println("VertexHit: " +
					// vertexHit.getCurrentLineSegment().getNameString() );
					if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
						count++;
					}

				}

				double transmittance = calculateTransmittance(emitterPoint, randomDetectorPoint, energyEV);
				double transmittance2 = calculateTransmittance(randomDetectorPoint, emitterPoint, energyEV);

				System.out.println("\nNew result for: " + randomVec + "\nCompare: "
						+ String.format("%.8f", ((double) count / (double) sampleSize)) + "\n" + "with a:  "
						+ transmittance + "\n" + "with a:  " + transmittance2 + "\n");
			}
		}

		if (test_partial_Transmittance) {
			SimpleVector randomVec = new SimpleVector(0.5 * width, 0.5 * height);
			// SimpleVector randomVec = new SimpleVector(sampler.random() * width,
			// sampler.random() * height);
			PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
					sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
			StraightLine ray = new StraightLine(emitterPoint, randomDetectorPoint);
			// ray.normalize();
			// ray.evaluate(500.0);
			// Find "Virtual light point"
			// XRayHitVertex startVertex = nextRayPoint(ray, startEnergyEV);
			// while ("detector".equals(startVertex.getCurrentLineSegment().getNameString())
			// || "background
			// material".equals(startVertex.getCurrentLineSegment().getNameString())) {
			// startVertex = nextRayPoint(ray, startEnergyEV);
			// }

			System.out.println("\nWith end-pont: " + randomVec);
			
			XRayHitVertex startVertex = new XRayHitVertex();
			int times = 10;

			for (int j = 0; j <= times; ++j) {
				double u = 200.0f * j / times;
				// double u = 100.0;
				PointND var = new PointND((u - 100.0), 0.0, 0.0);
				startVertex.setEndPoint(var);

				// Set up random point on the detector
				// randomVec = new SimpleVector(sampler.random() * width, sampler.random() *
				// height);
				randomVec = new SimpleVector(0.5 * width, 0.5 * height);
				randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				ray = new StraightLine(startVertex.getEndPoint(), randomDetectorPoint);
				ray.normalize();

				// System.out.println("Intersting: ");

				int sampleSize = 1000000;
				int count1k = 0;
				int count100k = 0;
				int count1m = 0;

				XRayHitVertex vertexHit;
				// Try 200.000 times to cast a ray from the point to the detector and count the
				// successes
				for (int p = 0; p < sampleSize; ++p) {

					// if(p == 1000 && (count == 0 || count == p)) {
					// randomVec = new SimpleVector(sampler.random() * width, sampler.random() *
					// height);
					// randomDetectorPoint =
					// proj.computeDetectorPoint(camCenter.getAbstractVector(), randomVec,
					// sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
					// ray = new StraightLine(startVertex.getEndPoint(), randomDetectorPoint);
					// ray.normalize();
					// count = 0;
					// p = 0;
					// System.err.println("This should not happen with partial transmittance!");
					// }

					vertexHit = nextRayPoint(ray, energyEV);
					// System.out.println("VertexHit: " +
					// vertexHit.getCurrentLineSegment().getNameString() );
					if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
						if(p < 100)
							count1k++;
						
						if(p < 10000)
							count100k++;
						
						if(p < 1000000)
							count1m++;
					}

				}
				// vizualizeRay(result, startVertex.getEndPoint(), randomDetectorPoint,
				// Color.magenta, 20);
				double transmittance = calculateTransmittance(startVertex.getEndPoint(), randomDetectorPoint, energyEV);
//				double transmittance2 = calculateTransmittance(randomDetectorPoint, startVertex.getEndPoint(), energyEV);

//				System.out.println("\nWe start with our hit-point at: " + startVertex.getEndPoint().getAbstractVector() 
////						+ "\nWith end-pont: " + randomVec
//						+ "\nCompare: " + String.format("%.8f", ((double) count / (double) sampleSize)) + "\n with a: "
//						+ String.format("%.8f", transmittance) + "\n"
////						+ String.format("%.8f", transmittance2) + "\n"
//						);
				System.out.println((var.get(0)+100)+ " & " + ((double) count1k / (double) 100) + " & " + ((double) count100k / (double) 10000) + " & " + ((double) count1m / (double) 1000000) + " & " + String.format("%.8f", transmittance) + " \\");
			}
		}
	}

	public void testScatteringAngle(PointND start) {
		double laenge = 200.0;
		double energyEV = startEnergyEV;
		System.out.println("Start testing the angle!");

		PointND base = new PointND(0.0, 0.0, 0.0);
		if (start != null) {
			base = start;
		}
		StraightLine ray = new StraightLine(emitterPoint, base);
		ray.setPoint(base);
		ray.normalize();

		int version = 11;
		
		if(version == 7) {
			int count = 5000000;
			
			int numberOfangles[] = new int[180];
			System.out.println("\n#x y");
			for (int i = 0; i < count; ++i) {
				SimpleVector dir = ray.getDirection().clone();
				sampler.sampleComptonScattering(energyEV, dir);
				double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir);
				int slot = (int) Math.floor((Math.toDegrees(radians)));
				numberOfangles[slot] += 1;
			}
			for (int i = 0; i < numberOfangles.length; i++) {
				System.out.println(i + " " + (double)numberOfangles[i] * 100.0 / count);
			}
			
		}
		
		if(version == 8) {
			
			//10M = 33.84 minuten
			int count = 10 * XRayTracer.mega;

			int numberOfangles[] = new int[1800];
			int numberOf140[] 	= new int[1800];
			int numberOf20[] 	= new int[1800];
			System.out.println("\n#x y");
			for (int i = 0; i <count; ++i) {
				double rad1M = sampler.getComptonAngleTheta(1 * XRayTracer.mega);
				double rad140k = sampler.getComptonAngleTheta(140 * XRayTracer.kilo);
				double rad20k = sampler.getComptonAngleTheta(20 * XRayTracer.kilo);
//				double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir);
				
				numberOfangles[(int) Math.floor((	Math.toDegrees(rad1M)	* 10.0))] += 1;
				numberOf140[(int) 	Math.floor((	Math.toDegrees(rad140k)	* 10.0))] += 1;
				numberOf20[(int) 	Math.floor((	Math.toDegrees(rad20k)	* 10.0))] += 1;
				
			}
			
			for (int i = 0; i < numberOfangles.length; i++) {
				System.out.println(i + " " + (double)numberOfangles[i] * 100.0 / count  + " " + (double)numberOf140[i] * 100.0 / count + " " + (double)numberOf20[i] * 100.0 / count);
			}
		}
		
		if(version == 9) {
			System.out.println("Output of the kleinNishinaDifferentialCrossSection");
			
			double e0 = 1 * XRayTracer.mega;
			double e1 = 140 * XRayTracer.kilo;
			double e2 = 20 * XRayTracer.kilo;
			
			System.out.println("\nx kndCrossSection0 kndCrossSection1 kndCrossSection2");
			for (double t = 0; t <= (180); t += 1) {

				double theta = Math.toRadians(t);
				double kndCrossSection0 = XRayTracerSampling.comptonAngleCrossSection(e0, theta);
				double kndCrossSection1 = XRayTracerSampling.comptonAngleCrossSection(e1, theta);
				double kndCrossSection2 = XRayTracerSampling.comptonAngleCrossSection(e2, theta);
				
				System.out.println(t + " " + kndCrossSection0 + " " + kndCrossSection1 + " " + kndCrossSection2);
			}			
		}
		
		if(version == 10) {
			System.out.println("Output of the kleinNishinaDifferentialCrossSection");
			
			double e0 = 1 * XRayTracer.mega;
			double e1 = 140 * XRayTracer.kilo;
			double e2 = 20 * XRayTracer.kilo;
			
			System.out.println("\nx kndCrossSection0 kndCrossSection1 kndCrossSection2");
			for (double t = 0; t <= (180); t += 1) {

				double theta = Math.toRadians(t);
				double kndCrossSection0 = XRayTracerSampling.comptonAngleProbability(e0, theta);
				double kndCrossSection1 = XRayTracerSampling.comptonAngleProbability(e1, theta);
				double kndCrossSection2 = XRayTracerSampling.comptonAngleProbability(e2, theta);
				
				System.out.println(t + " " + kndCrossSection0 + " " + kndCrossSection1 + " " + kndCrossSection2);
			}			
		}
		

		if(version == 11) {
			
			//10M = 33.84 minuten
			int count = 10 * XRayTracer.mega;

			int numberOfangles[] = new int[180];
			int numberOf140[] 	= new int[180];
			int numberOf20[] 	= new int[180];
			System.out.println("\n#x y");
			for (int i = 0; i <count; ++i) {
				double rad1M = sampler.getComptonAngleTheta(1 * XRayTracer.mega);
				double rad140k = sampler.getComptonAngleTheta(140 * XRayTracer.kilo);
				double rad20k = sampler.getComptonAngleTheta(20 * XRayTracer.kilo);

				numberOfangles[(int) Math.floor((Math.toDegrees(rad1M)))] += 1;
				numberOf140[(int) Math.floor((Math.toDegrees(rad140k)))] += 1;
				numberOf20[(int) Math.floor((Math.toDegrees(rad20k)))] += 1;
				
			}
			
			for (int i = 0; i < numberOfangles.length; i++) {
				System.out.println(i + " " + (double)numberOfangles[i] * 100.0 / count  + " " + (double)numberOf140[i] * 100.0 / count + " " + (double)numberOf20[i] * 100.0 / count);
			}
		}

		
		if (version == 0) {
			System.out.println("Output of the comptonAngleCrossSection value");

			double e0 = 1 * XRayTracer.mega;
			double e1 = 140 * XRayTracer.kilo;
			double e2 = 20 * XRayTracer.kilo;
			
			System.out.println("\nx " + e0 + " " + e1 + " " + e2 + " ");
			for (double t = 90; t <= (180); t += 0.1) {

				double theta = Math.toRadians(t);
				double phaseFunction0 = XRayTracerSampling.comptonAngleCrossSection(e0, theta);
				double phaseFunction1 = XRayTracerSampling.comptonAngleCrossSection(e1, theta);
				double phaseFunction2 = XRayTracerSampling.comptonAngleCrossSection(e2, theta);
				
				System.out.println(t + " " + phaseFunction0 * 100.0 + " " + phaseFunction1 * 100.0 + " " + phaseFunction2 * 100.0);
			}
		}
		
		if (version == 1) {
			System.out.println("Visualize the comptonAngleCrossSection value");

			for (int t = 0; t <= (180); t += 2) {
				for (int p = 0; p < 360; p += 2) {
					double theta = Math.toRadians(t);
					double phi = Math.toRadians(p);
					SimpleVector dir = ray.getDirection().clone();

					double energy = testScattering(energyEV, dir, theta, phi);

					double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir); // .multipliedBy(-1.0)
					double u = XRayTracerSampling.comptonAngleCrossSection(energy, radians);
					// if(u>max) {
					// max = u;
					// maxAngle = (double)t;
					// }

					// System.out.println("Point: " + t +":" + p + " has an angle of " +
					// Math.toDegrees(radians));

					vizualizeRay(result, base, (new PointND(
							SimpleOperators.add(base.getAbstractVector(), dir.multipliedBy(12.5 * u * laenge)))),
							Color.cyan, 10);
				}
			}
			// System.out.println("We maybe should show max: " + maxAngle);
		}

		if (version == 2) {
			System.out.println("Visualize the probability of scattering in a direction");
			int testingAngle = 90;
			int samplingSize = 2;
			int pickingAngle = (180 * samplingSize);
			int numberOfangles[] = new int[pickingAngle];
			for (int p = 0; p < pickingAngle; p++) {
				numberOfangles[p] = 0;
			}

			int count = 5000000;
			for (int i = 0; i < count; ++i) {
				SimpleVector dir = ray.getDirection().clone();
				sampler.sampleComptonScattering(energyEV, dir);
				double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir);
				int slot = (int) Math.floor((samplingSize * Math.toDegrees(radians)));
				numberOfangles[slot] += 1;
			}

			double scaling = (1.0 * numberOfangles[samplingSize * testingAngle]) / count;
			for (int t = 0; t < pickingAngle; t += 2) {
				double t2 = (double) t / samplingSize;
				double u = (1.0 * numberOfangles[t]) / count;

				// for (int p = 0; p < 360; p += 2) {

				double theta = Math.toRadians(t2);
				double phi = Math.toRadians(0.0);
				SimpleVector dir = ray.getDirection().clone();
				testScattering(energyEV, dir, theta, phi);
				vizualizeRay(result, base, (new PointND(
						SimpleOperators.add(base.getAbstractVector(), dir.multipliedBy(1.0 * u / scaling * laenge)))),
						Color.magenta, 3);
				// }
			}

			for (int i = 0; i < pickingAngle; i += 10) {
				System.out.println("Angle: " + (double) i / samplingSize + " we get= "
						+ ((1.0 * numberOfangles[i]) / count) + "\n compared to: " + XRayTracerSampling
								.comptonAngleCrossSection(energyEV, Math.toRadians((double) i / samplingSize)));
			}
		}

		if (version == 3) {
			System.out.println("Version_3");

			int count = 2000;
			for (int p = 0; p < count; ++p) {

				SimpleVector dir = ray.getDirection().clone();
				sampler.sampleComptonScattering(energyEV, dir);
				writeStepResults(result, energyEV, 0.0,	(new PointND(SimpleOperators.add(base.getAbstractVector(), dir.multipliedBy(laenge)))), null,
						Color.magenta);
				// vizualizeRay(result, base, (new
				// PointND(SimpleOperators.add(base.getAbstractVector(),
				// dir.multipliedBy(laenge)))), Color.magenta, 10);
			}
		}

		if (version == 4) {
			System.out.println("Version_4");

			int samplingSize = 2;
			int pickingAngle = (180 * samplingSize);
			int numberOfangles[] = new int[pickingAngle];
			for (int p = 0; p < pickingAngle; p++) {
				numberOfangles[p] = 0;
			}
			// set up
			int count = 1000000;
			for (int i = 0; i < count; ++i) {
				SimpleVector dir = ray.getDirection().clone();
				sampler.sampleComptonScattering(energyEV, dir);
				double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir);
				int slot = (int) Math.floor((samplingSize * Math.toDegrees(radians)));
				numberOfangles[slot] += 1;
			}

			// for(int p = 0; p < 180; p+=15) {
			// System.out.println("");
			int testingAngle = 10;
			// scattering angle of compton sampling
			double scaling = (1.0 * numberOfangles[samplingSize * testingAngle]) / count;

			for (int t = 0; t < pickingAngle; t++) {
				// for(int m = 0; m < 180; m += 10) {//Enable for 3D
				int m = 0;
				// if(true) {
				// int t = samplingSize * testingAngle;

				double t2 = (double) t / samplingSize;
				double theta = Math.toRadians(t2);
				double phi = Math.toRadians(m);
				SimpleVector dir = ray.getDirection().clone();

				testScattering(energyEV, dir, theta, phi);
				double u = (1.0 * numberOfangles[t]) / count;
				// System.out.println("Array: " + (10.0 * u / scaling * laenge));
				vizualizeRay(result, base, (new PointND(
						SimpleOperators.add(base.getAbstractVector(), dir.multipliedBy(10.0 * u / scaling * laenge)))),
						Color.magenta, 10);
				// }
			}

			// compton angle cross section test
			SimpleVector dir = ray.getDirection().clone();
			double energy = testScattering(energyEV, dir, Math.toRadians(testingAngle), Math.toRadians(0.0));
			scaling = XRayTracerSampling.comptonAngleCrossSection(energy, Math.toRadians(testingAngle));

			for (int t = 0; t < 180; t++) {
				// for(int m = 180; m < 360; m += 10) { //enable for 3D
				int m = 180;
				// if(true) {
				// int t = testingAngle;

				double theta = Math.toRadians(t);
				double phi = Math.toRadians(m);
				dir = ray.getDirection().clone();

				energy = testScattering(energyEV, dir, theta, phi);

				double radians = nHelperFkt.getAngleInRad(ray.getDirection(), dir); // .multipliedBy(-1.0)
				double u = XRayTracerSampling.comptonAngleCrossSection(energy, radians);

				// System.out.println("Point: " + t +":" + p + " has an angle of " +
				// Math.toDegrees(radians));
				// System.out.println("CrossSection: "+ Math.toDegrees(radians) + " compared to
				// " + testingAngle + "\n = " + u + " / " + scaling + " \nresults: " + (10.0 * u
				// / scaling * laenge));
				vizualizeRay(result, base, (new PointND(
						SimpleOperators.add(base.getAbstractVector(), dir.multipliedBy(10.0 * u / scaling * laenge)))),
						Color.cyan, 10);
				// }
			}
		}
		// }

		if (version == 5) {
			System.out.println("Version_5_PATH");
			// PointND base = new PointND(0.0, 0.0, 0.0);
			// StraightLine ray = new StraightLine(camCenter, base);
			// ray.setPoint(base);
			// ray.normalize();
			writeStepResults(result, 0.0, 0.0, base, null, Color.magenta);

			int count = 5000000;
			for (int i = 0; i < count; ++i) {

				SimpleVector dir = ray.getDirection().clone();
				double changedEnergyEV = sampler.sampleComptonScattering(energyEV, dir);
				StraightLine newRay = new StraightLine(base, dir);

				XRayHitVertex vertexHit = nextRayPoint(newRay, changedEnergyEV);

				if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {

					// System.out.println("\nNew Scattering: "
					// + "\n Ray: " + ray.getPoint() + " & " + ray.getDirection()
					// + "\n Transforms: " + newRay.getPoint() + " & " + newRay.getDirection()
					// + "\n Energy: " + energyEV + " => " + changedEnergyEV
					// + "\n VertexHit: " + vertexHit.getCurrentLineSegment().getNameString()
					// + "\n EndPoint: " + vertexHit.getEndPoint()
					// );
					//
					// vizualizeRay(result, base, vertexHit.getEndPoint(), Color.magenta, 10);
					//
					synchronized (grid) {
						detector.absorbPhoton(grid, vertexHit.getEndPoint(), changedEnergyEV);
					}

					detectorHits++;
				}
			}
		}

		if (version == 6) {
			System.out.println("Version_6_VPL");

			XRayVPL vpLight = new XRayVPL(base, energyEV, 1, emitterPoint);

			vizualizeRay(result, base, emitterPoint, Color.cyan, 20);

			int count = 100000;
			for (int i = 0; i < count; ++i) {
				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

				StraightLine camera = new StraightLine(randomDetectorPoint, vpLight.getVplPos());
				camera.normalize();

				double theta = nHelperFkt.getAngleInRad(vpLight.getDirection(), camera.getDirection()); // .multipliedBy(-1.0)
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(vpLight.getEnergyEV(), theta);

				double energy = XRayTracerSampling.getScatteredPhotonEnergy(vpLight.getEnergyEV(), theta);

				double throughput = calculateTransmittance(vpLight.getVplPos(), randomDetectorPoint, energy);

				double lightPower = (energy * throughput * lightPhaseFkt);

				synchronized (grid) {
					detector.absorbPhoton(grid, randomDetectorPoint, lightPower / 1.0);
				}
				detectorHits++;

				// vizualizeRay(result, vpLight.getVplPos(), randomDetectorPoint, Color.magenta,
				// 10);

			}
		}

		return;
	}

	public void testShadowRayVizualisierung() {
		double energyEV = startEnergyEV;
		// Test to show a ShadowRay
		for (simulatedRays = 0; simulatedRays < 5000;) {// ++simulatedRays) {

			// Setup Light-Ray
			XRayHitVertex lightVertex = null;
			while (true) {
				SimpleVector randomVec = new SimpleVector(sampler.random() * grid.getWidth(),
						sampler.random() * grid.getHeight());
				StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVec));

				lightVertex = nextRayPoint(ray, startEnergyEV);

				// Ray scatters out of the scene
				if ("background material".equals(lightVertex.getCurrentLineSegment().getNameString())) {
					continue;
				}
				// if the ray hit the detector box, write the result to the grid
				if ("detector".equals(lightVertex.getCurrentLineSegment().getNameString())) {
					synchronized (grid) {
						detector.absorbPhoton(grid, lightVertex.getEndPoint(), energyEV);
					}
					detectorHits++;
					continue;
				}
				break;
			}

			// Setup Cam-Ray
			XRayHitVertex camPathVertex = null;
			while (true) {
				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine camRay = new StraightLine(randomDetectorPoint, emitterPoint);
				camRay.normalize();

				camPathVertex = tryToSetUpCamRay(camRay, energyEV, null);
				// simulatedRays++;

				if (camPathVertex != null) {
					break;
				}
				// writeStepResults(result, energyEV, 0.0, randomDetectorPoint, null,
				// Color.PINK);

				// Nothing in between emitter and the detector -> direct lighting
				synchronized (grid) {
					detector.absorbPhoton(grid, randomDetectorPoint, startEnergyEV);
				}
				detectorHits++;
			}

			// Calculate the angle between the rays and the shadow ray
			if (showOutputonce) {

				StraightLine shadowRay = new StraightLine(lightVertex.getEndPoint(), camPathVertex.getEndPoint());
				shadowRay.normalize();
				SimpleVector v1 = shadowRay.getDirection();
				SimpleVector v2 = lightVertex.getRayDir().multipliedBy(-1.0);

				double radians = nHelperFkt.getAngleInRad(v1, v2);
				double degree = nHelperFkt.getAngleInDeg(v1, v2);

				double changedEnergy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, radians);
				double divison = changedEnergy / energyEV;

				double scatterRad = nHelperFkt.transformToScatterAngle(radians);
				double scatterDeg = Math.toDegrees(scatterRad);

				double crossRad = XRayTracerSampling.comptonAngleCrossSection(energyEV, scatterRad);
				double crossDeg = XRayTracerSampling.comptonAngleCrossSection(energyEV, scatterDeg);

				if (crossRad > 1.0f) {
					// showOutputonce = false;

					System.out.println("We have a combination of the following values: " + "\n radiant: " + radians
							+ "\n angle  : " + degree + "\n scatterAngle " + scatterRad + "\n scatterDeg " + scatterDeg
							// + "\n probRaD: " + XRayTracerSampling.comptonAngleProbability(energyEV,
							// radians)
							// + "\n ProbAng: " + XRayTracerSampling.comptonAngleProbability(energyEV,
							// degree)
							+ "\n comptonAngleCrossSection with radians: "
							+ XRayTracerSampling.comptonAngleCrossSection(energyEV, radians)
							+ "\n comptonAngleCrossSection with degree: "
							+ XRayTracerSampling.comptonAngleCrossSection(energyEV, degree)
							+ "\n comptonAngleCrossSection with scatter radians: " + crossRad
							+ "\n comptonAngleCrossSection with scatter degree: " + crossDeg + "\n EnergyDivison: "
							+ divison + "\n");
				}

			}

			simulatedRays++;
			vizualizeRay(result, lightVertex.getStartPoint(), lightVertex.getEndPoint(), Color.RED, 20);
			// vizualizeRay(result, camPathVertex.getOrigin(),
			// camPathVertex.getStartPoint(), Color.BLUE, 20);
			vizualizeRay(result, lightVertex.getEndPoint(), camPathVertex.getStartPoint(), Color.GREEN, 20);

		}

	}

	public void testVizualisierungDetectorOutline() {
		double energyEV = startEnergyEV;
		SimpleVector detector00 = new SimpleVector(0.0, 0.0);
		SimpleVector detector01 = new SimpleVector(0.0, height);
		SimpleVector detector10 = new SimpleVector(width, 0.0);
		SimpleVector detector11 = new SimpleVector(width, height);

		// StraightLine ray00 = new StraightLine(camCenter,
		// proj.computeRayDirection(detector00));
		// StraightLine ray01 = new StraightLine(camCenter,
		// proj.computeRayDirection(detector01));
		// StraightLine ray10 = new StraightLine(camCenter,
		// proj.computeRayDirection(detector10));
		// StraightLine ray11 = new StraightLine(camCenter,
		// proj.computeRayDirection(detector11));

		PointND randomDetectorPoint00 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), detector00,
				sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
		PointND randomDetectorPoint01 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), detector01,
				sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
		PointND randomDetectorPoint10 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), detector10,
				sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
		PointND randomDetectorPoint11 = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), detector11,
				sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

		// System.out.println("We have the following points: " + "\n" +
		// randomDetectorPoint00 + "\n" + randomDetectorPoint01 + "\n" +
		// randomDetectorPoint10 + "\n" + randomDetectorPoint11);

		double dist = 1.0;
		writeStepResults(result, energyEV, dist, randomDetectorPoint00, null, Color.CYAN);
		writeStepResults(result, energyEV, dist, randomDetectorPoint01, null, Color.CYAN);
		writeStepResults(result, energyEV, dist, randomDetectorPoint10, null, Color.CYAN);
		writeStepResults(result, energyEV, dist, randomDetectorPoint11, null, Color.CYAN);

//		vizualizeRay(result, randomDetectorPoint00, randomDetectorPoint01, Color.CYAN, 40);
//		vizualizeRay(result, randomDetectorPoint00, randomDetectorPoint10, Color.MAGENTA, 40);
//		vizualizeRay(result, randomDetectorPoint00, randomDetectorPoint11, Color.YELLOW, 40);

		int numberOfPoints = 50;
		vizualizeRay(result, emitterPoint, randomDetectorPoint01, Color.MAGENTA, numberOfPoints);
		vizualizeRay(result, emitterPoint, randomDetectorPoint10, Color.MAGENTA, numberOfPoints);
		vizualizeRay(result, emitterPoint, randomDetectorPoint11, Color.MAGENTA, numberOfPoints);
		vizualizeRay(result, emitterPoint, randomDetectorPoint00, Color.MAGENTA, numberOfPoints);

//		SimpleVector nray01 = SimpleOperators.subtract(randomDetectorPoint10.getAbstractVector(), randomDetectorPoint00.getAbstractVector());
//		SimpleVector nray10 = SimpleOperators.subtract(randomDetectorPoint01.getAbstractVector(), randomDetectorPoint00.getAbstractVector());
//		SimpleVector nray11 = SimpleOperators.subtract(randomDetectorPoint11.getAbstractVector(), randomDetectorPoint00.getAbstractVector());
//
//		SimpleVector middlePoint = new SimpleVector(0.5 * width, 0.5 * height);
//		SimpleVector normDir = proj.computeRayDirection(middlePoint);
//		PointND middle = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), middlePoint, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
//		vizualizeRay(result, middle, emitterPoint, Color.RED, 40);
//
//		SimpleVector normaleVec = nHelperFkt.crossProduct3D(nray01, nray10);

//		normDir.normalizeL2();
//		nray01.normalizeL2();
//		normaleVec.normalizeL2();
//		System.out.println("Check the normals: " + normDir + "\n  & " + normaleVec);
//
//		// double angle10 = nHelperFkt.angleDeg(nray10, normDir);
//		// double angle11 = nHelperFkt.angleDeg(nray11, normDir);
//
//		double normangle01 = nHelperFkt.getAngleInRad(nray01, normaleVec);
//		double normangle10 = nHelperFkt.getAngleInRad(nray10, normaleVec);
//		double normangle11 = nHelperFkt.getAngleInRad(nray11, normaleVec);
//
//		System.out.println("We have the different Angles: " + Math.toDegrees(normangle01) + " & "
//				+ Math.toDegrees(normangle10) + " & " + Math.toDegrees(normangle11));

	}

	public void comparePTFirst() {
		// Create same starting point all the time
		boolean testingPT = true;
		boolean testingBDPT = true;
		int output = 0;
		// SimpleVector cornerVec = new SimpleVector(0.0 * width, 0.0 * height);
		// PointND corner = proj.computeDetectorPoint(emitterPoint.getAbstractVector(),
		// cornerVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
		SimpleVector middleVec = new SimpleVector(0.5 * width, 0.5 * height);

		PointND middle = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), middleVec, sourceDetectorDistance,
				pixelDimX, pixelDimY, width, height);
		PointND var = new PointND(0.0, 0.0, 0.0);
		double energyEV = startEnergyEV;

		double distance = var.euclideanDistance(middle);
		if (threadNumber == 1) {
			System.out.println("Debug:" + "\nEmitter:" + emitterPoint + "\n Point:" + var + "\n Vec:" + middleVec
					+ "\n sourceDetectorDistance:" + sourceDetectorDistance + "\n pixelDims:" + pixelDimX + " | "
					+ pixelDimY + "\n dim:" + width + " | " + height + "\n");

			System.out.println("Middle point: " + middle);
			vizualizeRay(result, var, middle, Color.CYAN, 20);
			System.out.println("Distance of interaction and detector center: " + distance);
			System.out.println("Detector: " + width + "|" + height + " = " + width * height);
			System.out.println("SolidAngle = " + width * height / distance / distance);
			System.out.println("");
		}

		StraightLine ray = new StraightLine(emitterPoint, var);
		if (testingPT) { // Test path tracing with only the last connection
			for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
				// Sample results for PT
				SimpleVector dir = ray.getDirection().clone();

				double changedEnergyEV = sampler.sampleComptonScattering(energyEV, dir);
				// double changedEnergyEV = energyEV;
				// dir = sampler.samplingDirection(dir, 2.0);

				StraightLine newRay = new StraightLine(var, dir);

				if (simulatedRays < 100) //
					vizualizeRay(result, var, newRay.evaluate(200.0), Color.MAGENTA, 10);

				XRayHitVertex vertexHit = nextRayPoint(newRay, changedEnergyEV);

				if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
					// if (output < 1) {
					// output++;
					// System.out.println("\nVertexHit position: " + vertexHit.getEndPoint()+
					// "\nDistance " +
					// vertexHit.getEndPoint().euclideanDistance(newRay.getPoint()));
					// }

					scatteringNumber[0]++;
					contribution[0] += changedEnergyEV;

					synchronized (grid) {
						detector.absorbPhoton(grid, vertexHit.getEndPoint(), energyEV);
					}
					detectorHits++;

				}
			}

		}

		if (testingBDPT) { // Test BPDT with only connecting 1 point to the detector
			for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine camRay = new StraightLine(randomDetectorPoint, var);
				camRay.normalize();

				double theta = nHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0), ray.getDirection());
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);
				// double probability = XRayTracerSampling.angleProbabilityCompton(energyEV,
				// theta);

				// if (lightPhaseFkt - probability > 0.00001) {
				// System.err.println("Dafuq am i doing with my life");
				// System.out.println("Compare: " + lightPhaseFkt + " with " + probability + " =
				// "
				// + (lightPhaseFkt - probability));
				// }

				double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
				double transmittance = calculateTransmittance(randomDetectorPoint, var, energy);
				// if(transmittance != 1.0)
				// System.err.println("What is wrong with the transmittance?" + transmittance);

				double dist = randomDetectorPoint.euclideanDistance(var); // * XRayTracer.milli
				double phi = nHelperFkt.getAngleInRad(camRay.getDirection(), normalOfDetector);
				double dX = Math.cos(phi);
				double dY = Math
						.cos(nHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0), ray.getDirection()));

				StraightLine toMiddle = new StraightLine(middle, var);
				double dZ = Math.cos(nHelperFkt.getAngleInRad(toMiddle.getDirection(), normalOfDetector));

				// double g0erm = (pixelDimX * pixelDimY) * dX / dist / dist;
				// double g1erm = (width * height) * dX / dist / dist ;
				double g2erm = ((width * pixelDimX * height * pixelDimY) * dX) / (dist * dist);
				// double g3erm = ((width*pixelDimX * height*pixelDimY) * 1.0) / (dist * dist);
				// double g4erm = ((width*pixelDimX * height*pixelDimY) * dY * dX)/ (dist *
				// dist);
				// double g5erm = ((width*pixelDimX * height*pixelDimY) * dZ) / (dist * dist);

				if (output < 2) {
					vizualizeRay(result, randomDetectorPoint, var, Color.YELLOW, 20);
					// System.out.println("Pixel Dimensions are: " + pixelDimX + " : " + pixelDimY);
					// System.out.println("Light phase function: " + lightPhaseFkt);
					// //System.out.println("Probability to scatter in this projected solid angle: "
					// + probability);
					// System.out.println("Angle: " + Math.toDegrees(phi) + " results in cos(x)= " +
					// dX);
					// System.out.println("Distance between the points: " + dist);
					// System.out.println("Detector valeus: " + width + "|" + height);
					// System.out.println("Value of the gTerm is: " + gTerm + "\n");
					output++;
				}

				scatteringNumber[1]++;

				// contribution[1] += (energy * transmittance * lightPhaseFkt);
				// contribution[2] += (energy * transmittance * lightPhaseFkt * g0erm);
				// contribution[3] += (energy * transmittance * lightPhaseFkt * g1erm);
				contribution[1] += (energy * transmittance * lightPhaseFkt * g2erm);
				// contribution[2] += (energy * transmittance * lightPhaseFkt * g2erm / 2);
				// contribution[2] += (energy * transmittance * lightPhaseFkt * g3erm);
				// contribution[3] += (energy * transmittance * lightPhaseFkt * g4erm);
				// contribution[4] += (energy * transmittance * lightPhaseFkt * g5erm);

				// //Testing hit probability:
				// contribution[1] += (1.0 * lightPhaseFkt * gTerm);
			}
		}
		return;
	}

	public void universalFirstCon() {
		double energyEV = startEnergyEV;

		for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
			// Find a valid interaction point within the scene
			XRayHitVertex firstScatterPoint = null;
			do {
				SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
				StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));
				firstScatterPoint = nextRayPoint(ray, energyEV);

			} while ("background material".equals(firstScatterPoint.getCurrentLineSegment().getNameString())
					|| "detector".equals(firstScatterPoint.getCurrentLineSegment().getNameString()));

			// Calculate the energy transport for path-tracing
			{
				SimpleVector dir = firstScatterPoint.getRayDir().clone();

				double changedEnergyEV = sampler.sampleComptonScattering(energyEV, dir);
				StraightLine newRay = new StraightLine(firstScatterPoint.getEndPoint(), dir);

				XRayHitVertex vertexHit = nextRayPoint(newRay, changedEnergyEV);

				// if (simulatedRays < 20) //
				// vizualizeRay(result, vertexHit.getEndPoint(),
				// firstScatterPoint.getEndPoint(), Color.MAGENTA, 10);

				if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
					scatteringNumber[0]++;
					contribution[0] += changedEnergyEV;
					synchronized (grid) {
						detector.absorbPhoton(grid, vertexHit.getEndPoint(), energyEV);
					}
					detectorHits++;
				}
			}

			// Calculate the energy transport for bdp-tracing
			{
				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
						sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine camRay = new StraightLine(randomDetectorPoint, firstScatterPoint.getEndPoint());
				camRay.normalize();

				if (simulatedRays < 20) //
					vizualizeRay(result, firstScatterPoint.getEndPoint(), randomDetectorPoint, Color.YELLOW, 10);

				double theta = nHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0),
						firstScatterPoint.getRayDir());
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

				double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
				double transmittance = calculateTransmittance(randomDetectorPoint, firstScatterPoint.getEndPoint(),
						energy);

				double dist = randomDetectorPoint.euclideanDistance(firstScatterPoint.getEndPoint());
				double phi = nHelperFkt.getAngleInRad(camRay.getDirection(), normalOfDetector);
				double dX = Math.cos(phi);
				double gTerm = ((width * pixelDimX * height * pixelDimY) * dX) / (dist * dist);

				contribution[2] += (energy * transmittance * lightPhaseFkt * gTerm);
			}
		}
	}

	public void compareVersions() {
		boolean pathTesting = true;
		boolean vplTesting = false;
		double energyEV = startEnergyEV;

		if (pathTesting) { // Compareing PT and BDPT
			rayTracer.bdpt = false;
			for (int i = 0; i < 2; ++i) {
				for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
					SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
					StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));

					rayTracer.raytrace(ray, energyEV, 0, 0);
				}
				rayTracer.bdpt = true;
			}
			
			double contrib[] = rayTracer.getContributionNumber();
			for (int j = 0; j < contrib.length; j++) {
				if(contrib[j] != 0)
					contribution[j] = contrib[j];
			}
		}

		if (vplTesting) {
			XRayVPL collection[] = new XRayVPL[virtualLightsPerThread];
			simulatedRays = 0;
			// Set up VPLs
			collection = vplTracer.createVPLPoints();

			for (int i = 0; i < virtualLightsPerThread; ++i)
				collectionOfAllVPLs[threadNumber * virtualLightsPerThread + i] = collection[i];

			try {
				barrierVPL.await();
			} catch (InterruptedException e) {
				System.err.println("barrierVPL interrupted!");
				e.printStackTrace();
			} catch (BrokenBarrierException e) {
				System.err.println("barrierVPL broken!");
				e.printStackTrace();
			}
			for (int i = 0; i < attempting.length; i++)
				vplPathNumber += attempting[i];

			vplTracer.collectVPLScattering(numRays, collection, vplTracer.getPercentOfVPLs());
			
			double contrib[] = vplTracer.getContributionNumber();
			for (int j = 0; j < contrib.length; j++) {
				if(contrib[j] != 0)
					contribution[j] = contrib[j];
			}
		}
	}

	public void testBDPTNumbers() {
		if(threadNumber == 0)
			System.out.println("Testing the BPDT with a variable number of connections per hitpoint!");
		
		double energyEV = startEnergyEV;
		for (int i = 0; i < 5; i++) {
			int checks = (int) Math.pow(2.0, i);
			int sum = checks * checks;
			
			if(threadNumber == 0)
				System.out.println("BDPT with " + checks + " connections per dimension or " +  sum + " per pixel!");
			
			for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {

				SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
				StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));
				XRayHitVertex firstScatterPoint = nextRayPoint(ray, energyEV);

				if ("background material".equals(firstScatterPoint.getCurrentLineSegment().getNameString()) || "detector".equals(firstScatterPoint.getCurrentLineSegment().getNameString())) {
					continue;
				}
				if (sampler.random() * (firstScatterPoint.getPhotoAbsorption() + firstScatterPoint.getComptonAbsorption()) <= firstScatterPoint.getPhotoAbsorption()) {
					continue;
				} 
				
				for (int x = 0; x < checks; x++) {
					for (int y = 0; y < checks; y++) {
						SimpleVector randomVec = new SimpleVector(((double) x + sampler.random()) / checks * width, ((double) y + sampler.random()) / checks * height);
						PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
						StraightLine camRay = new StraightLine(randomDetectorPoint, firstScatterPoint.getEndPoint());
						camRay.normalize();

						double theta = nHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0), firstScatterPoint.getRayDir());
						double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

						double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
						double transmittance = calculateTransmittance(randomDetectorPoint, firstScatterPoint.getEndPoint(), energy);
						double g = sampler.gTerm(randomDetectorPoint, firstScatterPoint.getEndPoint(), true);
						double pA = 1.0 / sampler.probabilityDensityOfArea();

						double lightPower = (energy * transmittance * lightPhaseFkt * g * pA);
						contribution[i] += lightPower / sum;
					}
				}
			}
		}
	}
	
	public void testCameraRaysL1() {
		double energyEV = startEnergyEV;
		
		for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
			XRayHitVertex camPathVertex = createCamRay(energyEV, 0.5);
			StraightLine camRay = new StraightLine(camPathVertex.getStartPoint(), camPathVertex.getEndPoint());
			camRay.normalize();

			StraightLine lightRay = new StraightLine(camPathVertex.getEndPoint(), emitterPoint);
			lightRay.normalize();
			
			double theta = nHelperFkt.getAngleInRad(camRay.getDirection(), lightRay.getDirection());
			double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

			double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
			double transmittance = calculateTransmittance(emitterPoint, camPathVertex.getEndPoint(), energy);
			double g = sampler.gTerm(emitterPoint, camPathVertex.getEndPoint(), false);
			double pA = 1.0;

			double lightPower = (energy * transmittance * lightPhaseFkt * g * pA);
			
			contribution[4] += (lightPower / camPathVertex.getDistanceFromStartingPoint());
		}
	}
	
	public void bdptASvpl() {
		int vplCount = 0;
		int vplsPerPoint = (int) (8 * virtualLightsPerThread * vplTracer.getPercentOfVPLs());
		int connectionsPerVPL = (int) Math.floor(numRays * vplsPerPoint / (8 * virtualLightsPerThread));
		double energyEV = startEnergyEV;
		
		if(threadNumber == 0) {
			System.out.println("Using BDPT just like the VPL algorithm!");
			System.out.println("Calculating " + 8 * virtualLightsPerThread + " VPLs in total"
					+ "\nChecking on " + connectionsPerVPL + " points per VPL -> " + vplsPerPoint  + " per point"
					+ "\nResulting in " + virtualLightsPerThread * 8 * connectionsPerVPL + " connections to the scene.");
		}
		
		while(vplCount < virtualLightsPerThread) {
			SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
			StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));
			XRayHitVertex firstScatterPoint = nextRayPoint(ray, energyEV);
			simulatedRays++;
			if ("background material".equals(firstScatterPoint.getCurrentLineSegment().getNameString()) || "detector".equals(firstScatterPoint.getCurrentLineSegment().getNameString())) {
				continue;
			}
			if (sampler.random() * (firstScatterPoint.getPhotoAbsorption() + firstScatterPoint.getComptonAbsorption()) <= firstScatterPoint.getPhotoAbsorption()) {
				continue;
			}
			vplCount++;
			
			for(int i = 0; i < connectionsPerVPL; ++i) {
				SimpleVector randomVec = new SimpleVector(sampler.random()*width, sampler.random()*height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine camRay = new StraightLine(randomDetectorPoint, firstScatterPoint.getEndPoint());
				camRay.normalize();

				double theta = nHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0), firstScatterPoint.getRayDir());
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

				double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
				double transmittance = calculateTransmittance(randomDetectorPoint, firstScatterPoint.getEndPoint(), energy);
				double g = sampler.gTerm(randomDetectorPoint, firstScatterPoint.getEndPoint(), true);
				double pA = 1.0 / sampler.probabilityDensityOfArea();

				double lightPower = (energy * transmittance * lightPhaseFkt * g * pA);
				contribution[2] += lightPower / connectionsPerVPL;
			}
			
		}
	}
	
	public void testPhotometry(boolean sampling) {
		SimpleVector middleVec = new SimpleVector(0.0 * width, 0.0 * height);
		PointND corner = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), middleVec, sourceDetectorDistance,
				pixelDimX, pixelDimY, width, height);

		for (int i = 0; i < 5; ++i) {
			double factor = Math.pow(2.0, i + 1);

			double testingDistance = factor * 300;
			double dist = -1.0;
			PointND resulting = null;
			double total = Double.MAX_VALUE;
			double w = -1.0;

			double offset = -80.0 / Math.pow(2, i);
			for (double k = -1.0; k < 1.0; k += 0.02) {
				PointND changing = new PointND(testingDistance - 599.99 + offset + k, 0.0, 0.0);
				// vizualizeRay(result, emitterPoint, var, Color.CYAN, 20);
				dist = changing.euclideanDistance(corner);

				if (Math.abs((dist - testingDistance)) < total) {
					resulting = changing.clone();
					total = Math.abs((dist - testingDistance));
					w = k;
				}
			}

			dist = resulting.euclideanDistance(corner);

			if (threadNumber == 1) {
				System.out.println("Value: " + i);
				// System.out.println("Point is: " + resulting);
				System.out.println("Distance of interaction and detector: " + dist);
				// System.out.println("Detector: " + width + "|" + height + " = " + width *
				// height);
				System.out.println("SolidAngle = " + width * height / dist / dist);
				System.out.println("");
			}
			vizualizeRay(result, resulting, corner, Color.YELLOW, 20);
			StraightLine ray = new StraightLine(emitterPoint, resulting);

			{ // Test path tracing with only the last connection
				for (simulatedRays = 0; simulatedRays < numRays; ++simulatedRays) {
					// Sample results for PT
					SimpleVector dir = ray.getDirection().clone();
					double changedEnergyEV = 0.0;
					if (sampling) {
						changedEnergyEV = sampler.sampleComptonScattering(startEnergyEV, dir);
					} else {
						changedEnergyEV = startEnergyEV;
						dir = sampler.samplingDirection(dir, 2.0);
					}

					StraightLine newRay = new StraightLine(resulting, dir);

					// if (i == 2 && simulatedRays < 100) //
					// vizualizeRay(result, resulting, newRay.evaluate(200.0), Color.MAGENTA, 10);

					XRayHitVertex vertexHit = nextRayPoint(newRay, changedEnergyEV);

					if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
						scatteringNumber[i]++;
						contribution[i] += changedEnergyEV;

						// This should be equal for all testcases
						double distance = vertexHit.getEndPoint().euclideanDistance(resulting) * XRayTracer.milli;
						double phi = nHelperFkt.getAngleInRad(newRay.getDirection().multipliedBy(-1.0),
								normalOfDetector);
						double dX = Math.cos(phi);
						double gTerm = 1.0 / distance / distance;
						// contribution[i + 5] += (changedEnergyEV / gTerm);
						contribution[i + 5] += (changedEnergyEV / gTerm) * XRayTracer.mega / (width * pixelDimX * height * pixelDimY);
						contribution[i + 10] += dX * (changedEnergyEV / gTerm) * XRayTracer.mega / (width * pixelDimX * height * pixelDimY);
					}
				}
			}
		}
	}
}
			