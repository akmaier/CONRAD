/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics.workingClass;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Random;
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
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.AbstractRayTracer;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.tutorial.physics.XRayHitVertex;
import edu.stanford.rsl.tutorial.physics.XRayTracerSampling;
import edu.stanford.rsl.tutorial.physics.NHelperFkt;
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;

/**
 * Abstract class as a base for the different variations of the light transport algorithm
 * @author Tobias Miksch based on work of Fabian Rückert
 */
public abstract class AbstractWorker implements Runnable {

	// Variables controlling the output
	protected boolean writeAdditionalData = true;
	protected int show = 0;
	protected boolean showOutputonce = false;
	protected boolean allowDirectLighting = true;
	protected boolean infiniteSimulation = false;

	//  distinguishing between versions
	protected int version = -1;
	protected PointND middleOfDetector;

	protected boolean testHitVertex = false;

	protected boolean bdpt = false;
	protected boolean vpl = false;
	protected boolean virtualRayLight = false;
	protected boolean testMaterial = false;

	// Magic values constructed by observation and testing
	protected int lightRayLength;
	// protected double russianRoulette = 0.05;
	protected double epsilon = 0.001;
	protected double coneSample = 0.1;
	
	// Variables filled by constructor
	protected AbstractRayTracer raytracer;
	protected RaytraceResult result;
	protected XRayDetector detector;
	protected Projection proj;
	protected Grid2D grid;
	protected int threadNumber;
	protected PointND emitterPoint;
	protected SimpleVector normalOfDetector;

	XRayTracerSampling sampler;
	
	protected long numRays;
	protected int startEnergyEV;
	protected double sourceDetectorDistance;
	protected double pixelDimX;
	protected double pixelDimY;
	protected int width;
	protected int height;
	protected int virtualLightsPerThread;
	protected Material background;

	// variables for statistics
	protected long simulatedRays = 0;
	protected long segmentsTraced = 0;
	protected long detectorHits = 0;
	protected long virtualLightSourceAttempts = 0;
	protected int scatteringNumber[] = new int[50];
	protected double contribution[] = new double[50];

	protected Random random;
	
	public AbstractWorker(PriorityRayTracer raytracer, RaytraceResult res, long numRays, int startenergyEV, Grid2D grid,
			XRayDetector detector, Projection proj, boolean infinite, boolean writeAdditionalData,
			double sourceDetectorDist, double pixelDimensionX, double pixelDimensionY, Material background,
			int versionNumber, int threadNumber, Random random,  int lightRayLength) {

		this.emitterPoint = new PointND(proj.computeCameraCenter());
		this.raytracer = raytracer;

		this.result = res;
		this.numRays = numRays;
		this.startEnergyEV = startenergyEV;
		this.grid = grid;
		this.detector = detector;
		this.proj = proj;
		this.infiniteSimulation = infinite;
		this.writeAdditionalData = writeAdditionalData;
		this.sourceDetectorDistance = sourceDetectorDist;
		this.pixelDimX = pixelDimensionX;
		this.pixelDimY = pixelDimensionY;
		this.background = background;
		this.lightRayLength = lightRayLength;

		this.width = grid.getWidth();
		this.height = grid.getHeight();

		this.threadNumber = threadNumber;
		this.version = versionNumber;
		this.random = random;
		
		switch (versionNumber) {
		case 0:
			break;
		case 1:
			this.bdpt = true;
			break;
		case 2:
			this.vpl = true;
			break;
		case 3:
			this.virtualRayLight = true;
			break;
		default:
			if (versionNumber > 100 && versionNumber < 107) {
				testMaterial = true;
				version = versionNumber - 100;
			} else {
				System.err.println("Invalid version number!");
				throw new RuntimeException("versionNumber != VALID");
			}
		}
		// Additional variables that are often needed in the calculation
		SimpleVector middle = new SimpleVector(0.5 * width, 0.5 * height);
		middleOfDetector = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), middle, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

		SimpleVector normal = SimpleOperators.subtract(emitterPoint.getAbstractVector(), middleOfDetector.getAbstractVector());
		this.normalOfDetector = normal.normalizedL2();
		
	}
	
	public long getRayCount() {
		// a lock is not needed as reading of a volatile long is atomic
		return simulatedRays;
	}
	
	public long getSegmentsTraced() {
		// a lock is not needed as reading of a volatile long is atomic
		return segmentsTraced;
	}

	public long getDetectorHits() {
		// a lock is not needed as reading of a volatile long is atomic
		return detectorHits;
	}
	
	public long getVLSAttempts() {
		// a lock is not needed as reading of a volatile long is atomic
		return virtualLightSourceAttempts;
	}

	public int[] getScatteringNumber() {
		return scatteringNumber;
	}

	public double[] getContributionNumber() {
		return contribution;
	}
	
	/**
	 * Combine the results of this worker with the final result
	 * 
	 * @param finalResult
	 *            The combined result
	 */
	public void addResults(RaytraceResult finalResult) {
		finalResult.addResults(result);
	}
	
	/**
	 * Trace a single photon until it is absorbed or it left the scene.
	 * 
	 * @param ray
	 *            The start position and direction of the photon
	 * @param energyEV
	 *            The current energy of the photon in eV
	 * @return XRayHitVertex The XRayHitVertex where the results from the current
	 *         step should be saved to
	 */
	protected XRayHitVertex nextRayPoint(StraightLine original, double energyEV) {
		// check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());
		if (!bb.isSatisfiedBy(original.getPoint())) {
			System.err.println("BoundingBox");
			return null;
		}

		StraightLine ray = (StraightLine) original.clone();
		PointND rayPoint = ray.getPoint().clone();
		PointND offSet = ray.evaluate(-epsilon);
		ray.setPoint(offSet);
		ray.normalize();

		ArrayList<PhysicalObject> physobjs = raytracer.castRay(ray);
		if (physobjs == null) {
			// should not happen because ray is inside of the bounding box
			System.err.println("No background material set?");
			throw new RuntimeException("physobjs == null");
		}

		// check direction of the line segments
		boolean reversed = false;

		// Edge e1 = (Edge) physobjs.get(0).getShape();
		// SimpleVector diff = e1.getEnd().getAbstractVector().clone();
		// diff.subtract(e1.getPoint().getAbstractVector());
		// diff.normalizeL2();
		// diff.subtract(ray.getDirection());
		//
		// if (diff.normL2() > CONRAD.SMALL_VALUE) {
		// reversed = true;
		// }

		double distToNextMaterial = Double.NaN;
		double totalDistUntilNextInteraction = 0;
		Material currentMaterial = null;

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		boolean foundStartSegment = false;

		// the line segments are sorted, find the first line segment containing the
		// point
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			// System.out.print("\nCurrentLineSegment: " +
			// currentLineSegment.getNameString());

			if (foundStartSegment || NHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					rayPoint.getAbstractVector())) {
				// get length between the raypoint and the next material in ray direction

				distToNextMaterial = rayPoint.euclideanDistance(end);
				// System.out.println("Pathtracer using: " + rayPoint.getAbstractVector() +
				// distToNextMaterial);
				if (distToNextMaterial < 0.0) {
					System.err.println("Distance is negative");
				}
				foundStartSegment = true;
			} else {
				continue;
			}

			if ("detector".equals(currentLineSegment.getNameString())) {
				return new XRayHitVertex(ray.getPoint(), start.clone(), ray.getDirection(), currentLineSegment, 0.0, 0.0,
						energyEV, totalDistUntilNextInteraction);
			}

			// if the raypoint is very close to the endpoint continue with next line segment
			if (distToNextMaterial > CONRAD.SMALL_VALUE) {
				// get distance until next physics interaction
				currentMaterial = currentLineSegment.getMaterial();

				/*
				 * Calculates the mass attenuation coefficient of a material given its weighted
				 * atomic composition. This method uses an internal static cache to inhibit
				 * excessive file access. However, this requires the method to be synchronized.
				 * This may lead to quite slow performance in parallel use.
				 */
				photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.PHOTOELECTRIC_ABSORPTION);
				comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.INCOHERENT_ATTENUATION);

				double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption,
						comptonAbsorption);

				if (distToNextInteraction < distToNextMaterial) {
					// ray interacts in current line segment, move the photon to the interaction
					// point
					rayPoint = new PointND(SimpleOperators.add(rayPoint.getAbstractVector(), ray.getDirection().multipliedBy(distToNextInteraction)));

					totalDistUntilNextInteraction += distToNextInteraction;
					break;
				} else {
					// ray exceeded boundary of current material, continue with next segment
					rayPoint = end.clone();
					totalDistUntilNextInteraction += distToNextMaterial;
				}

				if (foundStartSegment && "detector".equals(currentLineSegment.getNameString())) {
					break;
				}
			}
			// take the next line segment
		}
		// System.out.println("");
		return new XRayHitVertex(ray.getPoint(), rayPoint, ray.getDirection(), currentLineSegment, photoAbsorption,
				comptonAbsorption, energyEV, totalDistUntilNextInteraction);
	}
	
	/**
	 * Cast Rays from the detector into the scene and collect all the energy the
	 * VPLs contain
	 */
	void collectDirectLighting(long numberOfRays) {
		if (numberOfRays <= 0 || !allowDirectLighting) {
			// System.out.println("Not Calculating any direct illumination!");
			return;
		}

		for (long count = simulatedRays + numberOfRays; simulatedRays < count; ++simulatedRays) {
			SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
			PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
					sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
			double throughput = calculateTransmittance(emitterPoint, randomDetectorPoint, startEnergyEV);

			contribution[0] += throughput * startEnergyEV;
			synchronized (grid) {
				detector.absorbPhoton(grid, randomDetectorPoint, throughput * startEnergyEV);
			}
			detectorHits++;
		}
	}
	
	/**
	 * This function trys to cast rays from the detector into the scene until it
	 * finds a hitpoint
	 * 
	 * @param energyEV
	 *            The simulated energy of the camera ray in eV
	 * @return XRayHitVertex The XRayHitVertex where the interaction should be
	 */
	protected XRayHitVertex createCamRay(double energyEV, double coneSampling) {

		XRayHitVertex camPathVertex = null;
		int effort = 0;
		// Set up a valid camRay by try and error
		if (coneSampling == 0.0) {
			do {
				// try only 10.000 times before aborting the try and error process
				if (effort > 10000) {
					System.err.println("Could not create a usefull camera ray!");
					throw new RuntimeException("camRay == NULL");
				}

				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				StraightLine camRay = new StraightLine(randomDetectorPoint, new PointND(0.0, 0.0, 0.0));
				camRay.normalize();

				camPathVertex = tryToSetUpCamRay(camRay, energyEV, null);
				effort++;

				// try until there is a valid camRay
				if (camPathVertex != null) {
					return camPathVertex;
				}
				// writeStepResults(result, energyEV, 0.0, randomDetectorPoint, null,
				// Color.PINK);
			} while (true);
		} else {
			do {
				// try only 10.000 times before aborting the try and error process
				if (effort > 10000) {
					System.err.println("Could not create a usefull camera ray!");
					throw new RuntimeException("camRay == NULL");
				}

				SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

				int attempts = 100;
				for (int i = 0; i < attempts; ++i) {
					StraightLine camRay = new StraightLine(randomDetectorPoint, new PointND(0.0, 0.0, 0.0));
					camRay.normalize();

					SimpleVector ret = sampler.samplingDirection(camRay.getDirection(), coneSampling);
					camRay.setDirection(ret);

					camPathVertex = tryToSetUpCamRay(camRay, energyEV, null);
					effort++;
					
					// try until there is a valid camRay
					if (camPathVertex != null) {
						// System.out.println("Testing CamPath: " +
						// camPathVertex.getCurrentLineSegment().getNameString());
						camPathVertex.setDistanceFromStartingPoint(effort);
						return camPathVertex;
					}
				}
				// writeStepResults(result, energyEV, 0.0, randomDetectorPoint, null,
				// Color.PINK);
			} while (true);
		}

		// vizualizeRay(result, camPathVertex.getOrigin(),
		// camPathVertex.getStartPoint(), Color.PINK, 5);
		// vizualizeRay(result, camPathVertex.getStartPoint(),
		// camPathVertex.getEndPoint(), Color.CYAN, 10);
		// writeStepResults(result, energyEV, 0.0, camPathVertex.getOrigin(), null,
		// Color.CYAN);
		// writeStepResults(result, energyEV, 0.0, camPathVertex.getStartPoint(), null,
		// Color.MAGENTA);
		// writeStepResults(result, energyEV, 0.0, camPathVertex.getEndPoint(), null,
		// Color.YELLOW);
	}
	
	protected XRayHitVertex tryToSetUpCamRay(StraightLine ray, double energyEV, PointND detectorStart) {

		// check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());
		if (!bb.isSatisfiedBy(ray.getPoint())) {
			System.err.println("BoundingBox ist not satisfied by CamRay?");
			throw new RuntimeException("!bb.isSatisfiedBy");
		}

		PointND rayPoint = ray.getPoint().clone();
		PointND offSet = ray.evaluate(-epsilon);
		ray.setPoint(offSet);
		ray.normalize();

		ArrayList<PhysicalObject> physobjs = raytracer.castRay(ray);
		if (physobjs == null) {
			// should not happen because ray is inside of the bounding box
			System.err.println("No background material set?");
			throw new RuntimeException("physobjs == null");
		}

		// check direction of the line segments
		boolean reversed = false;

		double totalDistUntilNextInteraction = 0;
		Material currentMaterial = null;
		PointND origin;
		if (detectorStart == null) {
			origin = rayPoint.clone();
		} else {
			origin = detectorStart;
		}

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		boolean foundStartSegment = false;
		double distToNextMaterial = Double.NaN;

		// the line segments are sorted, find the first line segment containing the
		// point
		for (int i = 0; i < physobjs.size(); ++i) {

			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			// Test if we are already behind the Detector or still out of the scene
			if (foundStartSegment || NHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					rayPoint.getAbstractVector())) {

				foundStartSegment = true;
				if ("detector".equals(currentLineSegment.getNameString())) {
					rayPoint = new PointND(end);
					continue;
				}
				distToNextMaterial = rayPoint.euclideanDistance(end);
			} else {
				// Case 0
				continue;
			}

			// We went past the emitter and could not find an interaction point!
			if (foundStartSegment && !NHelperFkt.isBetween(ray.getPoint().getAbstractVector(),
					emitterPoint.getAbstractVector(), rayPoint.getAbstractVector())) {
				// Case 2
				return null;
			}

			// if the raypoint is very close to the endpoint continue with next line segment
			if (distToNextMaterial > CONRAD.SMALL_VALUE) { // CONRAD.SMALL_VALUE
				// get distance until next physics interaction
				currentMaterial = currentLineSegment.getMaterial();

				photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.PHOTOELECTRIC_ABSORPTION);
				comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.INCOHERENT_ATTENUATION);

				double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption,
						comptonAbsorption);

				if (distToNextInteraction < distToNextMaterial) {
					// ray interacts in current line segment, move the photon to the interaction
					// point
					rayPoint = new PointND(SimpleOperators.add(rayPoint.getAbstractVector(),
							ray.getDirection().multipliedBy(distToNextInteraction)));
					totalDistUntilNextInteraction += distToNextInteraction;
					break;
				} else {
					// ray exceeded boundary of current material, continue with next segment
					rayPoint = new PointND(end);
					totalDistUntilNextInteraction += distToNextMaterial;
				}
			}

			// if the raypoint is very close to the endpoint continue with next line segment
			// if (distToNextMaterial > epsilon) {//CONRAD.SMALL_VALUE
			//
			// if(currentMaterial == null) {
			// currentMaterial = currentLineSegment.getMaterial();
			//
			// // this lookup slows down the RayTracer because it is synchronized
			// photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
			// AttenuationType.PHOTOELECTRIC_ABSORPTION);
			// comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
			// AttenuationType.INCOHERENT_ATTENUATION);
			// // currentMaterial.getDensity();
			// startingPoint = start.clone();
			//
			// } else if(!currentMaterial.equals(currentLineSegment.getMaterial())) {
			// //System.out.println("Different Materials with the testing sample!\n" +
			// currentMaterial.getName() + "\n" +
			// currentLineSegment.getMaterial().getName());
			// return new XRayHitVertex(startingPoint, rayPoint, origin, ray.getDirection(),
			// currentLineSegment, photoAbsorption, comptonAbsorption, energyEV,
			// totalDistUntilNextInteraction);
			// }
			//
			// No scattering within a camRay
			// double distToNextInteraction = 10 *
			// sampler.getDistanceUntilNextInteractionCm(photoAbsorption,
			// comptonAbsorption);
			// if (distToNextInteraction < distToNextMaterial) {
			// // ray interacts in current line segment, move the photon to the interaction
			// point
			// SimpleVector p = rayPoint.getAbstractVector();
			// p.add(ray.getDirection().multipliedBy(distToNextInteraction));
			//
			// totalDistUntilNextInteraction += distToNextInteraction;
			// break;
			// } else
			// {
			// // ray exceeded boundary of current material, continue with next segment
			// rayPoint = new PointND(end);
			// totalDistUntilNextInteraction += distToNextMaterial;
			// }
			// }
		}

		if ("background material".equals(currentLineSegment.getNameString())) {
			// System.out.println("CamRay set up was not successfull!");
			return null;
		}

		return new XRayHitVertex(origin, rayPoint, ray.getDirection(), currentLineSegment, photoAbsorption,
				comptonAbsorption, energyEV, totalDistUntilNextInteraction);

	}
	
	public double calculateTransmittance(PointND startPoint, PointND endPoint, double energyEV) {

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

		// check if the direction of the line segments is the same direction of they ray
		// or the opposite direction
		boolean reversed = false;

		double throughput = 0.0;
		PhysicalObject currentLineSegment = null;
		Edge e = null;
		Material currentMaterial = null;
		double distToNextMaterial = Double.NaN;
		PointND start = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		boolean foundStartSegment = false;

		// System.out.println("");
		// for (int i = 0; i < physobjs.size(); ++i) {
		// currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);
		// System.out.println(currentLineSegment.getNameString());
		// }

		// Iterate over all physical Objects the Shadowrays hits and sum up the
		// throughput
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);

			e = (Edge) currentLineSegment.getShape();
			start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			// if("bone".equals(currentLineSegment.getNameString())) {
			// vizualizeRay(result, start, startPoint, Color.YELLOW, 20);
			// vizualizeRay(result, end, endPoint, Color.MAGENTA, 20);
			// }

			// if we are not between the two points any longer
			if (foundStartSegment && !NHelperFkt.isBetween(startPoint.getAbstractVector(), endPoint.getAbstractVector(),
					start.getAbstractVector())) {
				// System.out.println(currentLineSegment.getNameString() + " leaving
				// Finished!\n");
				// if(Math.abs(totalDistance - startPoint.euclideanDistance(endPoint)) > 0.01){
				// System.out.println("We have a different distance then the one between the
				// points!" + totalDistance + " & " + startPoint.euclideanDistance(endPoint));
				// }
				return Math.exp(-throughput);
			}

			if (foundStartSegment || NHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					startPoint.getAbstractVector())) {
				// get length between the raypoint and the next material in ray direction
				if (!foundStartSegment) {
					// System.out.println("Starting with: \n "
					// + start.getAbstractVector() + "\n "
					// + end.getAbstractVector() + "\n "
					// + startPoint.getAbstractVector()
					// );
					//
					// System.out.println("Distance: " +
					// shadowRay.getPoint().euclideanDistance(end));

					if (NHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
							endPoint.getAbstractVector())) {
						// System.out.println("We end it here!");
						distToNextMaterial = startPoint.euclideanDistance(endPoint);
						// vizualizeRay(result, startPoint, endPoint, Color.magenta, 10);
					} else {
						// System.out.println("We start with: " + currentLineSegment.getNameString());
						distToNextMaterial = startPoint.euclideanDistance(end);

						// System.out.println("Eingedrungen: " + start.euclideanDistance(startPoint) + "
						// & " + distToNextMaterial);
						// vizualizeRay(result, startPoint, end, Color.blue, 10);
					}

					// System.out.println(currentLineSegment.getNameString() + "-- starting the
					// process!");
				} else {
					if (NHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
							endPoint.getAbstractVector())) {
						// System.out.println(" and add a little bit of: " +
						// currentLineSegment.getNameString());
						distToNextMaterial = start.euclideanDistance(endPoint);
						// vizualizeRay(result, start, endPoint, Color.cyan, 10);
					} else {
						// System.out.println(" and add: " + currentLineSegment.getNameString());
						distToNextMaterial = start.euclideanDistance(end);
						// vizualizeRay(result, start, end, Color.yellow, 10);
					}
					// System.out.println(currentLineSegment.getNameString());
				}
				foundStartSegment = true;
			} else {
				// continue with next line segment
				continue;
			}
			// Ignore the detector since he should absorb 100% of the energy
			if ("detector".equals(currentLineSegment.getNameString())) {
				// totalDistance += distToNextMaterial;
				continue;
			}
			// System.out.println("We have: " + currentLineSegment.getNameString() + " with
			// " + distToNextMaterial);

			currentMaterial = currentLineSegment.getMaterial();
			// this lookup slows down the RayTracer because it is synchronized
			photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
			comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);

			// Siehe Notizblatt += Vaccum Wavelength | numberDensity?
			throughput += XRayTracerSampling.getTransmittanceOfMaterial(photoAbsorption, comptonAbsorption,
					distToNextMaterial);
		}
		// System.out.println("Result: " + throughput + " and " +
		// Math.exp(-throughput));
		// System.out.println(currentLineSegment.getNameString() + " -- Finished!\n");
		// if(Math.abs(totalDistance - startPoint.euclideanDistance(endPoint)) > 0.01){
		// System.out.println("We have a different distance then the one between the
		// points!" + totalDistance + " & " + startPoint.euclideanDistance(endPoint));
		// }
		return Math.exp(-throughput);
	}
	
	
	protected void vizualizeRay(RaytraceResult result, PointND startP, PointND endP, Color farbe, int steps) {

		// writeStepResults(result, startEnergyEV, 0.01, startP, null, Color.RED);
		// writeStepResults(result, startEnergyEV, 0.01, endP, null, Color.CYAN);

		double distance = startP.euclideanDistance(endP);

		StraightLine ray = new StraightLine(startP, endP);
		ray.normalize();
		// System.out.println("The distance to beginning of " +
		// Thread.currentThread().getId() + " is: " + distance);
		// for(double steps = stepSize; steps < distance; steps += stepSize) {
		for (int i = 0; i <= steps; i++) { // Use if start and end are different color for(int i = 1; i < steps; i++) {
			double retreat = (distance / steps * i);
			PointND rayPoint = ray.evaluate(retreat);

			writeStepResults(result, startEnergyEV, (distance / steps), rayPoint, null, farbe);
		}

		return;
	}

	/**
	 * Saves the results for this ray interaction
	 * 
	 * @param result
	 *            The RaytraceResult where the the current result should be saved to
	 * @param energyEV
	 *            The energy in eV
	 * @param totalDistUntilNextInteraction
	 *            The total path length for this step
	 * @param rayPoint
	 *            The point of interaction
	 * @param e
	 *            The edge between the last interaction point and the current one
	 */
	protected void writeStepResults(RaytraceResult result, double energyEV, double totalDistUntilNextInteraction,
			PointND rayPoint, Edge e, Color farbe) {

		float factor = (((float) energyEV / startEnergyEV));
		if (farbe == null) {
			farbe = new Color(factor, 1 - factor, 0);
		}

		result.pathlength += totalDistUntilNextInteraction;
		result.pathlength2 += totalDistUntilNextInteraction * totalDistUntilNextInteraction;

		result.count += 1;
		result.x += rayPoint.get(0);
		result.x2 += rayPoint.get(0) * rayPoint.get(0);
		result.y += rayPoint.get(1);
		result.y2 += rayPoint.get(1) * rayPoint.get(1);
		result.z += rayPoint.get(2);
		result.z2 += rayPoint.get(2) * rayPoint.get(2);

		if (!writeAdditionalData)
			return;

		result.points.add(rayPoint);

		result.colors.add(farbe);

		result.edges.add(e);
	}

}
