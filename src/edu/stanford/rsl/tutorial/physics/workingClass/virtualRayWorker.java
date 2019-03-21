/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics.workingClass;

import java.awt.Color;
import java.util.ArrayList;
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
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.tutorial.physics.XRayHitVertex;
import edu.stanford.rsl.tutorial.physics.XRayTracerSampling;
import edu.stanford.rsl.tutorial.physics.XRayVPL;
import edu.stanford.rsl.tutorial.physics.XRayVRay;
import edu.stanford.rsl.tutorial.physics.nHelperFkt;
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;

/**
 * Implementation of the two step algorithm using virtual point lights for the calculation of indirect illumination
 * @author Tobias Miksch
 */
public class virtualRayWorker extends AbstractWorker {

	// For VRL
	private double vrlCreation = 0.3;
	private double volumeSurfaceContrib = 1.00;
	private double volumeVolumeContrib = 0.00;
	private int vrlSamplingNumber = 3;
	private boolean vrlByNumber = true;
	private double percentOfVRLs = 1.0 / 48.0;
	
	private int vrlAttemptsNumber = 0;
	private int virtualLightsPerThread;

	private volatile int attempting[];
	private final CyclicBarrier barrierVRL;
	private volatile XRayVRay allVirtualRayLights[];
	
	public virtualRayWorker(PriorityRayTracer raytracer, RaytraceResult res, long numRays, int startenergyEV,
			Grid2D grid, XRayDetector detector, Projection proj, boolean infinite, boolean writeAdditionalData,
			double sourceDetectorDist, double pixelDimensionX, double pixelDimensionY, Material background,
			int versionNumber, int threadNumber, Random random, int lightRayLength,
			int virtualLightsPerThread, XRayVRay collection[], int attempting[], CyclicBarrier barrier) {
		
		super(raytracer, res, numRays, startenergyEV, grid, detector, proj, infinite, writeAdditionalData,
				sourceDetectorDist, pixelDimensionX, pixelDimensionY, background, versionNumber, threadNumber, random, lightRayLength);
		// TODO Auto-generated constructor stub

		this.virtualLightsPerThread = virtualLightsPerThread;
		this.allVirtualRayLights = collection;
		this.attempting = attempting;
		this.barrierVRL = barrier;
	}

	@Override
	public void run() {

		if (random == null) {
			this.sampler = new XRayTracerSampling(ThreadLocalRandom.current(), (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		} else {
			this.sampler = new XRayTracerSampling(random, (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		}

		// TODO: Working on VRL
		XRayVRay collection[] = new XRayVRay[virtualLightsPerThread];
		simulatedRays = 0;
		
		// Set up VRLs per Thread
		collection = createVirtualRayLights();

		// Sync Threads with a volatile array -> single write multiple read
		for (int i = 0; i < virtualLightsPerThread; i++) {
			allVirtualRayLights[threadNumber * virtualLightsPerThread + i] = collection[i];
		}

		try {
			barrierVRL.await();
		} catch (InterruptedException e) {
			System.err.println("barrierVPL_2 interrupted!");
			e.printStackTrace();
		} catch (BrokenBarrierException e) {
			System.err.println("barrierVPL_2 broken!");
			e.printStackTrace();
		}
		// Security check for stupidness
		if ((volumeSurfaceContrib + volumeVolumeContrib) > 1.0) {
			System.out.println("Check distribution of numRays!");
			volumeVolumeContrib = Math.max(volumeSurfaceContrib, volumeVolumeContrib) - Math.min(volumeSurfaceContrib, volumeVolumeContrib);
		}

		for (int i = 0; i < attempting.length; i++) {
			vrlAttemptsNumber += attempting[i];
		}

		if(threadNumber == 1) {
			System.out.println("For the creation of " + allVirtualRayLights.length +  " the system had to use " + vrlAttemptsNumber + " attemps.");
		}
		
		// We split the number of simulated rays up into 3 different calculations: direct, surface and volume
		long restCount = numRays; // - simulatedRays
		long distributionSurface = (long) (restCount * volumeSurfaceContrib);
		long distributionVolume = (long) (restCount * volumeVolumeContrib);

		if (threadNumber == 1) {
			System.out.println("Distribtuion: " + (restCount - distributionSurface - distributionVolume) + " " + distributionSurface + " " + distributionVolume);
			System.out.println("Number of VRLs: " + allVirtualRayLights.length + " testing per pixel: "
					+ (allVirtualRayLights.length * percentOfVRLs) + "\nVRL path # total:" + vrlAttemptsNumber
					+ " resulting in = " + (allVirtualRayLights.length / (double) vrlAttemptsNumber) + "\n or scaled "
					+ (allVirtualRayLights.length / (double) vrlAttemptsNumber)
							* allVirtualRayLights[0].getLightScaling());
		}

		collectDirectLighting(restCount - distributionSurface - distributionVolume);
		collectVRLVolumeSurface(distributionSurface, percentOfVRLs);
		if (threadNumber == 1)
			System.out.println("Done with Volume-Surface-Illumination!");

//		collectVolumeToVolume(distributionVolume, percentOfVRLs, vrlCreation);
//		if (threadNumber == 1)
//			System.out.println("Done with Volume-Volume-Illumination! <-");

		return;
	}


	private XRayVRay[] createVirtualRayLights() {

		XRayVRay collection[] = new XRayVRay[virtualLightsPerThread];

		// long startingNumber = simulatedRays;
		int vrlCount = 0;
		int attempts = 0;

		while (true) {
			// prevent infinite loop if there is no participating media
			if (attempts == 100000 && vrlCount == 0) {
				System.err.println("Could not create a single virtual ray light!");
				throw new RuntimeException("vrlNumber == 0");
			}

//			SimpleVector randomVector = new SimpleVector(0.5 * width, 0.5 * height);
			SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
			StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));

			// SimpleVector evalDir = sampler.samplingDirection(proj.computeRayDirection(randomVector), vrlCreation);
			// StraightLine ray = new StraightLine(emitterPoint, evalDir);

			double currentEnergyEV = startEnergyEV;
			for (int i = 0; i < lightRayLength; ++i) {

				XRayVRay vrlHit = calcVRLPoints(ray, currentEnergyEV);
				if(i == 0)
					attempts++;

				if (vrlHit == null || !vrlHit.isValid() || vrlHit.getDistance() <= CONRAD.SMALL_VALUE) {
					break;
				}
				
				{
//					vizualizeRay(result, vrlHit.getVrlStart(), vrlHit.getVrlEnd(), Color.magenta, 3);
					double probNoRR = Math.pow((1.0 - (vrlHit.getPhotoAbsorption() / (vrlHit.getPhotoAbsorption() + vrlHit.getComptonAbsorption()))), 1.0);
//					double probOfAction = 1.0 -  calculateTransmittance(vrlHit.getVrlStart(), vrlHit.getVrlEnd(), vrlHit.getEnergyEV());
					
					vrlHit.setScattercount(i);
					collection[vrlCount] = vrlHit;// calcVRLPoints(ray, currentEnergyEV);
					collection[vrlCount].setLightScaling(probNoRR);
					++vrlCount;
				}
				
				// return asap or else the for could create an index out of bounds exception
				if (vrlCount >= virtualLightsPerThread) {
					virtualLightSourceAttempts = attempts;
					attempting[threadNumber] = attempts;
					return collection;
				}

				if (sampler.random() * (vrlHit.getPhotoAbsorption() + vrlHit.getComptonAbsorption()) <= vrlHit.getPhotoAbsorption()) {
					break;
				} else {
					// Calc next ray direction
					boolean basic = true;
					
					double sampleDistance = sampler.getDistanceUntilNextInteractionCm(vrlHit.getPhotoAbsorption(), vrlHit.getComptonAbsorption());
					if(sampleDistance >= vrlHit.getDistance()) {
						ray = new StraightLine(vrlHit.getVrlEnd(), ray.getDirection().clone());
						continue;
						//break; //This could also continue with a straight ray.
					}
					
					if (basic) {
						SimpleVector dir = ray.getDirection().clone();
						currentEnergyEV = sampler.sampleComptonScattering(currentEnergyEV, dir);
						ray = new StraightLine(vrlHit.getPointByDistance(sampleDistance), dir);
					} else {
						// TODO: sample a random point on the VRL. Maybe even discard if dist > sampler oder anstelle des oberen if
						double dist = 0.0;
						int shots = 0;
						do {
							dist = sampler.getDistanceUntilNextInteractionCm(vrlHit.getPhotoAbsorption(), vrlHit.getComptonAbsorption());
							shots++;
						} while (dist > vrlHit.getDistance() && shots <= 200);
						if (shots == 200) {
							break;
						}
						SimpleVector dir = ray.getDirection().clone();
						currentEnergyEV = sampler.sampleComptonScattering(currentEnergyEV, dir);
						ray = new StraightLine(vrlHit.getVrlStart(), dir);
						ray.setPoint(ray.evaluate(dist));
					}

				}
			}
		}
	}

	private XRayVRay calcVRLPoints(StraightLine original, double energyEV) {
		// check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());

		if (!bb.isSatisfiedBy(original.getPoint())) {
			System.err.println("BoundingBox_VRL");
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

		Material currentMaterial = null;

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		int foundStartSegment = -1;
		boolean foundStartPoint = false;
		PointND startPointOfVRL = null;

		// the line segments are sorted, find the first line segment containing the
		// point
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);
			double distToNextMaterial = Double.NaN;

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			// Search for the segment containing the rayPoint
			if (foundStartSegment != -1 || nHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					rayPoint.getAbstractVector())) {
				if (foundStartSegment == -1) {
					foundStartSegment = i;
				}
			} else {
				continue;
			}

			// Test for materials that are working. Depending it is the start and a VRL over multiple segments
			if (currentMaterial == null) {
				if (!"vacuum".equals(currentLineSegment.getMaterial().getName())) {
					// System.out.println("Setting Material as: " +
					// currentLineSegment.getMaterial());
					currentMaterial = currentLineSegment.getMaterial();
				} else {
					continue;
				}
			} else if (!currentMaterial.equals(currentLineSegment.getMaterial())) {
				break;
			}

			// search for the beging of the VRL-> either at the rayPoint or the start of the following Segment
			if (!foundStartPoint) {
				// System.out.println("Setting the starting point on: " + currentLineSegment.getNameString() + " and " + currentMaterial.getName() + " " + i +"/"+ foundStartSegment);
				if (foundStartSegment != i) {
					startPointOfVRL = start.clone();
				} else {
					startPointOfVRL = rayPoint.clone();
				}
				rayPoint = startPointOfVRL.clone();
				distToNextMaterial = startPointOfVRL.euclideanDistance(end);
				foundStartPoint = true;
			} else {
				distToNextMaterial = start.euclideanDistance(end);
			}

			// TODO: ----- decide if detector or density > x
			// We do not create a VRL on or on top of the detector
			if ("detector".equals(currentLineSegment.getNameString())) {
				foundStartPoint = false;
				break;
			}

			// Do we decide on the length of a VRL based on the scattering or simply trace until we reach an end?
			boolean useDistanceTillInteraction = false;
			if (useDistanceTillInteraction) {
				// if the raypoint is very close to the endpoint continue with next line segment
				if (distToNextMaterial > CONRAD.SMALL_VALUE) {
					// get distance until next physics interaction

					photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
					comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);

					double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption, comptonAbsorption);

					if (distToNextInteraction < distToNextMaterial) {
						// ray interacts in current line segment, move the photon to the interaction
						// point
						rayPoint = new PointND(SimpleOperators.add(rayPoint.getAbstractVector(), ray.getDirection().multipliedBy(distToNextInteraction)));
						// totalDistUntilNextInteraction += distToNextInteraction;
					} else {
						// ray exceeded boundary of current material, continue with next segment
						rayPoint = end.clone();
						// totalDistUntilNextInteraction += distToNextMaterial;
					}
				}
			} else {
				photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
				comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);
				rayPoint = new PointND(end);
			}
			// take the next line segment
		}

		if (currentMaterial == null || !foundStartPoint) {
			return null;
		}

		return new XRayVRay(startPointOfVRL, rayPoint, energyEV, photoAbsorption, comptonAbsorption, foundStartPoint);
	}
	
	private XRayVRay findVirtualRayLights(StraightLine original, double energyEV) {
		// check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());

		if (!bb.isSatisfiedBy(original.getPoint())) {
			System.err.println("BoundingBox_VRL");
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

		Material currentMaterial = null;

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		int foundStartSegment = -1;
		boolean foundStartPoint = false;
		PointND startPointOfVRL = null;

		// the line segments are sorted, find the first line segment containing the
		// point
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);
			double distToNextMaterial = Double.NaN;

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			// Search for the segment containing the rayPoint
			if (foundStartSegment != -1 || nHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(),
					rayPoint.getAbstractVector())) {
				if (foundStartSegment == -1) {
					foundStartSegment = i;
				}
			} else {
				continue;
			}

			// Test for materials that are working. Depending it is the start and a VRL over
			// multiple segments
			if (currentMaterial == null) {
				if (!"vacuum".equals(currentLineSegment.getMaterial().getName())) {
					// System.out.println("Setting Material as: " +
					// currentLineSegment.getMaterial());
					currentMaterial = currentLineSegment.getMaterial();
				} else {
					continue;
				}
			} else if (!currentMaterial.equals(currentLineSegment.getMaterial())) {
				break;
			}

			// search for the beging of the VRL-> either at the rayPoint or the start of the following Segment
			if (!foundStartPoint) {
				// System.out.println("Setting the starting point on: " + currentLineSegment.getNameString() + " and " + currentMaterial.getName() + " " + i +"/"+ foundStartSegment);
				if (foundStartSegment != i) {
					startPointOfVRL = start.clone();
				} else {
					startPointOfVRL = rayPoint.clone();
				}
				rayPoint = startPointOfVRL.clone();
				distToNextMaterial = startPointOfVRL.euclideanDistance(end);
				foundStartPoint = true;
			} else {
				distToNextMaterial = start.euclideanDistance(end);
			}

			// TODO: ----- decide if detector or density > x
			// We do not create a VRL on or on top of the detector
			if ("detector".equals(currentLineSegment.getNameString())) {
				foundStartPoint = false;
				break;
			}
			//TODO: Here we first reached a material worth sampling!
			
			if (distToNextMaterial > CONRAD.SMALL_VALUE) {
					photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
					comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);

					double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption, comptonAbsorption);

					if (distToNextInteraction < distToNextMaterial) {
						rayPoint = new PointND(SimpleOperators.add(rayPoint.getAbstractVector(), ray.getDirection().multipliedBy(distToNextInteraction)));
					} else {
						// ray exceeded boundary of current material, continue with next segment
						rayPoint = end.clone();
						currentMaterial = null;
					}
				}
			}
			// take the next line segment
		
		if (currentMaterial == null || !foundStartPoint) {
			return null;
		}

		return new XRayVRay(startPointOfVRL, rayPoint, energyEV, photoAbsorption, comptonAbsorption, foundStartPoint);
	}
	
	private XRayVRay nextVirtualRayLight(StraightLine original, double energyEV) {
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

		double distToNextMaterial = Double.NaN;
		Material currentMaterial = null;

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		boolean foundStartSegment = false;

		PointND startPointOfVRL = null;
		boolean foundValidPoint = false;
		
		// the line segments are sorted, find the first line segment containing the
		// point
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			if (foundStartSegment || nHelperFkt.isBetween(start.getAbstractVector(), end.getAbstractVector(), rayPoint.getAbstractVector())) {
			
				distToNextMaterial = rayPoint.euclideanDistance(end);
				if (distToNextMaterial < 0.0) {
					System.err.println("Distance is negative");
				}
				foundStartSegment = true;
			} else {
				continue;
			}

			if ("detector".equals(currentLineSegment.getNameString())) {
				return new XRayVRay(startPointOfVRL, rayPoint, energyEV, photoAbsorption, comptonAbsorption, false);
			}

			// if the raypoint is very close to the endpoint continue with next line segment
			if (distToNextMaterial > CONRAD.SMALL_VALUE) {
				// get distance until next physics interaction
				currentMaterial = currentLineSegment.getMaterial();

				photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
				comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,	AttenuationType.INCOHERENT_ATTENUATION);

				double distToNextInteraction = sampler.getDistanceUntilNextInteractionCm(photoAbsorption, comptonAbsorption);

				if (distToNextInteraction < distToNextMaterial) {
					// ray interacts in current line segment, move the photon to the interaction point
					rayPoint 		= end.clone();
					startPointOfVRL = start.clone();
					foundValidPoint = true;
					break;
				} else {
					// ray exceeded boundary of current material, continue with next segment
					rayPoint = end.clone();
				}

				if (foundStartSegment && "detector".equals(currentLineSegment.getNameString())) {
					break;
				}
			}
			// take the next line segment
		}
		
		return new XRayVRay(startPointOfVRL, rayPoint, energyEV, photoAbsorption, comptonAbsorption, foundValidPoint);
	}
	

	/**
	 * Cast Rays from the detector into the scene to multiple points on a VRL to
	 * collect the energy it contains
	 * 
	 * @param numberOfRays
	 *            = how many different points on the detector are picked
	 * 
	 * @param factor
	 *            = what percentage of VRLs are tested for each point -> 1 == check
	 *            every VPL
	 * 
	 *            vrlSamplingNumber = global variable about how many points on each
	 *            VRL are sampled
	 */
	private void collectVRLVolumeSurface(long numberOfRays, double factor) {
		if (numberOfRays <= 0)
			return;

		// System.out.println("Collect Vrl Surface illumination:" + numberOfRays);

		// How many different VRLs are tested for each point
		int virtualLightCount;
		if(vrlByNumber == true) {
			virtualLightCount = 5;
		} else {
			virtualLightCount = (int) (allVirtualRayLights.length * factor);
		}
		
		if(virtualLightCount < 1) {
			System.out.println("The sampling count for the vrls was 0 or less! Changed it to 1.");
			virtualLightCount = 1;
		}
		
		show = 0;
		for (long count = numberOfRays + simulatedRays; simulatedRays < count; ++simulatedRays) {
			// Get random detector point
			SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
			PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
			
			PointND rayLightPoint = new PointND();
			
			double pdf = (double) virtualLightCount * (double) vrlSamplingNumber / (double) allVirtualRayLights.length;
			double lightPower = 0.0;

			for (int i = 0; i < virtualLightCount; ++i) {
				// Get a random VPL
				int x = i;
				if (factor != 1.0) {
					x = sampler.random(allVirtualRayLights.length);
				}
				XRayVRay vRayLight = allVirtualRayLights[x];

				// How many points on each VRL are tested
				for (int p = 0; p < vrlSamplingNumber; ++p) {
//					double fraction = sampler.random();				
//					rayLightPoint = vRayLight.getPointOnStraightLine(fraction);
					double sampleDistance = sampler.getDistanceUntilNextInteractionCm(vRayLight.getPhotoAbsorption(), vRayLight.getComptonAbsorption());
					if(sampleDistance >= vRayLight.getDistance()) {
						//No interaction on the virtual light ray
//						if(show < 1) {
//							System.out.println("Compare: " +  sampleDistance + " with " + vRayLight.getDistance()
//							+ "\n Using: " + vRayLight.getPhotoAbsorption() + " & " + vRayLight.getComptonAbsorption());
//						}
						show++;
						continue;
					}
					segmentsTraced++;
					
					rayLightPoint = vRayLight.getPointByDistance(sampleDistance);

					StraightLine camera = new StraightLine(randomDetectorPoint, rayLightPoint);
					camera.normalize();
					
					double theta = nHelperFkt.getAngleInRad(camera.getDirection().multipliedBy(-1.0), vRayLight.getDirection()); // .multipliedBy(-1.0)
					double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(vRayLight.getEnergyEV(), theta);

					double energy = XRayTracerSampling.getScatteredPhotonEnergy(vRayLight.getEnergyEV(), theta);
					double transmittance = calculateTransmittance(randomDetectorPoint, rayLightPoint, energy);
					double g = sampler.gTerm(randomDetectorPoint, rayLightPoint, true);
					double pA = sampler.probabilityDensityOfArea();

					double scalingVirtualLights = vRayLight.getLightScaling() / ((double) vrlAttemptsNumber) / pdf;
					lightPower += ((energy * scalingVirtualLights) * transmittance * lightPhaseFkt * g / pA);
					contribution[vRayLight.getScattercount()] += ((energy * scalingVirtualLights) * transmittance * lightPhaseFkt * g / pA);
				}
			}
			//contribution[4] += lightPower;
			synchronized (grid) {
				detector.absorbPhoton(grid, randomDetectorPoint, lightPower);
				detectorHits++;
			}
		}
		if (threadNumber == 1) {
			System.out.println("Number of unsuccessfull rays: " +  show + " = " +  (double) show / (simulatedRays*vrlSamplingNumber*virtualLightCount)
					+ "\n resulting in: " + (1-((double) show / (simulatedRays*vrlSamplingNumber*virtualLightCount))) * (allVirtualRayLights.length / (double) vrlAttemptsNumber) * allVirtualRayLights[0].getLightScaling()
				);
		}
	}

	private void collectVolumeToVolume(int numberOfRays, double factor, double coneLotSample) {
		// Savety first
		if (numberOfRays <= 0)
			return;

		// System.out.println("Collect volume to volume illumination:" + numberOfRays);

		// How many different VRLs are tested for each point
		int virtualLightCount = (int) (allVirtualRayLights.length * factor);
		XRayHitVertex camPathVertex = null;

		for (long count = numberOfRays + simulatedRays; simulatedRays < count; ++simulatedRays) {

			// Set up camera ray points
			{
				if (camPathVertex != null) { // && !("background
												// material".equals(camPathVertex.getCurrentLineSegment().getNameString()))
					camPathVertex = tryToSetUpCamRay(
							new StraightLine(camPathVertex.getEndPoint(), camPathVertex.getRayDir()),
							camPathVertex.getEnergyEV(), camPathVertex.getStartPoint());
				}
				long startingNumber = simulatedRays;
				int attempts = 200;
				while (camPathVertex == null) {

					if ((startingNumber + 10000) == simulatedRays) {
						System.err.println("Could not create a usefull camera ray for VRL use!");
						throw new RuntimeException("camRay == NULL");
					}

					SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
					PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
							sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

					for (int i = 0; i < attempts; ++i) {
						StraightLine camRay = new StraightLine(randomDetectorPoint, emitterPoint);
						camRay.normalize();
						if (coneLotSample != 0.0) {
							SimpleVector ret = sampler.samplingDirection(camRay.getDirection(), coneLotSample);
							camRay.setDirection(ret);
						}

						camPathVertex = tryToSetUpCamRay(camRay, startEnergyEV, null);
						simulatedRays++;

						// try until there is a valid camRay
						if (camPathVertex != null) {
							// System.out.println("Attempts:" + i);
							break;
						}
					}

					// System.out.println("Could not find a point in time");
					// writeStepResults(result, energyEV, 0.0, randomDetectorPoint, null,
					// Color.PINK);
				}
			}
			// System.out.println("CamPathVertex Test: " +
			// camPathVertex.getCurrentLineSegment().getNameString());

			double lightPower = 0;
			for (int i = 0; i < virtualLightCount; ++i) {
				// Get a random VPL
				int x = i;
				if (factor != 1.0) {
					x = sampler.random(allVirtualRayLights.length);
				}
				XRayVRay vRayLight = allVirtualRayLights[x];

				for (int p = 0; p < vrlSamplingNumber; ++p) {
					double fraction = sampler.random();
					PointND rayLightPoint = vRayLight.getPointByFraction(fraction);
					PointND cameraPoint = camPathVertex.getEndPoint();

					// Calculate the contribution of the VPL - transmittance and angleProbability
					StraightLine shadowRay = new StraightLine(rayLightPoint, cameraPoint);
					shadowRay.normalize();

					if (showOutputonce) {
						showOutputonce = false;

						vizualizeRay(result, camPathVertex.getStartPoint(), cameraPoint, Color.GREEN, 20);
						vizualizeRay(result, emitterPoint, vRayLight.getVrlStart(), Color.BLUE, 20);
						vizualizeRay(result, vRayLight.getVrlStart(), vRayLight.getVrlEnd(), Color.YELLOW, 20);
						vizualizeRay(result, rayLightPoint, cameraPoint, Color.MAGENTA, 20);
					}

					double throughputVRL = vRayLight.getTransmittance(fraction);
					// Theta = Scatter angle
					double theta = nHelperFkt.getAngleInRad(vRayLight.getDirection(), shadowRay.getDirection()); // .multipliedBy(-1.0)
					double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(vRayLight.getEnergyEV(), theta);
					double energy = XRayTracerSampling.getScatteredPhotonEnergy(vRayLight.getEnergyEV(), theta);
					double throughputShadowRay = calculateTransmittance(rayLightPoint, cameraPoint, energy);

					theta = nHelperFkt.getAngleInRad(shadowRay.getDirection(),
							camPathVertex.getRayDir().multipliedBy(-1.0)); // .multipliedBy(-1.0)
					double cameraPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(vRayLight.getEnergyEV(), theta);
					energy = XRayTracerSampling.getScatteredPhotonEnergy(energy, theta);
					double throughputCamRay = calculateTransmittance(cameraPoint, camPathVertex.getStartPoint(),
							vRayLight.getEnergyEV());

					// TODO: Find the right PDF!
					double pdf = 1.0 / vRayLight.getDistance() / throughputCamRay;

					// lightPower += (throughputVRL * energy * throughputCamRay * lightPhaseFkt /
					// vrlSamplingNumber);
					lightPower += (energy * lightPhaseFkt * cameraPhaseFkt * throughputVRL * throughputShadowRay
							* throughputCamRay / vrlSamplingNumber);
				}
			}

			contribution[2] += (lightPower / (virtualLightCount));
			synchronized (grid) {
				detector.absorbPhoton(grid, camPathVertex.getStartPoint(), lightPower / (virtualLightCount));
			}
			detectorHits++;
		}
	}

	/**
	 * Cast Rays from the detector into the scene and connect it with multiple VRLs
	 * to collect an average of the energy
	 * 
	 * @param numberOfRays
	 *            = how many different points on the detector are picked
	 * 
	 * @param factor
	 *            = what percentage of VRLs are tested for each point -> 1 == check
	 *            every VPL
	 * 
	 *            vrlSamplingNumber = global variable about how many points on each
	 *            VRL are sampled
	 */
	// private void collVRLVolumeToVolume(int numberOfRays, double factor) {
	// //Savety first
	// if(numberOfRays <= 0) return;
	//
	// //How many different VRLs are tested for each point
	// int virtualLightCount = (int) (allVirtualRayLights.length * factor);
	//
	// //We test until we have enought points checked
	// for(long count = numberOfRays + simulatedRays; simulatedRays < count;
	// ++simulatedRays) {
	// PointND rayLightPoint = new PointND();
	// PointND cameraPoint = new PointND();
	// double lightPower = 0.0;
	//
	// //TODO: We could use coneSampling here?
	// XRayHitVertex contributingSegment = createCamRay(startEnergyEV, 0.0);
	// PointND randomDetectorPoint = contributingSegment.getEndPoint();
	// boolean infiniteRay = true;
	// boolean firstHit = true;
	//
	// //Sample over a number of VRLs
	// for (int f = 0; f < virtualLightCount; ++f) {
	// // Randomly pick a VRL
	// int x = sampler.random(allVirtualRayLights.length);
	// XRayVRay vRayLight = allVirtualRayLights[x];
	//
	// //How many points on each VRL are tested -> each combined with one different
	// point on the camRay
	// for(int i = 0; i < vrlSamplingNumber; ++i) {
	// //Find the unique points to connect
	//// double fraction = (double) i / (vrlSamplingNumber-1.f);
	//// rayLightPoint = vRayLight.getPointOnStraightLine(fraction);
	//// cameraPoint = contributingSegment.getPointOnStraightLine(fraction);
	//
	// //TODO: Check Hack around the infinite CamRay problem
	// if(!infiniteRay || firstHit || sampler.random() > 0.5) { //Get a random point
	// on the current segment
	// cameraPoint = contributingSegment.getPointOnStraightLine(sampler.random());
	// firstHit = false;
	// } else { //try to find a new Segment on the path
	// StraightLine extansionOfCam = new
	// StraightLine(contributingSegment.getEndPoint(),
	// contributingSegment.getRayDir());
	// XRayHitVertex infinteCamRay = tryToSetUpCamRay(extansionOfCam, startEnergyEV,
	// randomDetectorPoint);
	// if(infinteCamRay == null) {
	// infiniteRay = false;
	// cameraPoint = contributingSegment.getPointOnStraightLine(sampler.random());
	// } else {
	// contributingSegment = infinteCamRay;
	// cameraPoint = contributingSegment.getPointOnStraightLine(sampler.random());
	// }
	// }
	//
	// rayLightPoint = vRayLight.getPointOnStraightLine(sampler.random());
	//
	// //TODO: Lightpower phi of the point on the VRL - better distribution?
	// double energyChanged = vRayLight.getEnergyEV() / virtualLightCount;
	//
	// //Transmittance of u,v,w - cameraRay, lightRay(in lightPower included) and
	// shadowRay(connection)
	// double transmitU = calculateTransmittance(randomDetectorPoint, cameraPoint,
	// vRayLight.getEnergyEV());
	// double transmitW = calculateTransmittance(rayLightPoint, randomDetectorPoint,
	// vRayLight.getEnergyEV());
	//
	// //phaseFunctions u/w and v/w
	// SimpleVector wConnection =
	// SimpleOperators.subtract(rayLightPoint.getAbstractVector(),
	// cameraPoint.getAbstractVector());
	//
	// double thetaV = nHelperFkt.getAngleInRad(vRayLight.getDirection(),
	// wConnection);
	// double thetaU = nHelperFkt.getAngleInRad(contributingSegment.getRayDir(),
	// wConnection);
	//
	// double lightPhaseFkt =
	// XRayTracerSampling.comptonAngleCrossSection(vRayLight.getEnergyEV(), thetaV);
	// double cameraPhaseFkt =
	// XRayTracerSampling.comptonAngleCrossSection(vRayLight.getEnergyEV(), thetaU);
	//
	// lightPhaseFkt = nHelperFkt.clamp(lightPhaseFkt, 0.0, 1.0);
	// cameraPhaseFkt = nHelperFkt.clamp(cameraPhaseFkt, 0.0, 1.0);
	//
	// //TODO: pdf -> probability of choosing a point on the VRL and the
	// corresponding point on the camRay?!?
	// double pdf = 1.0f / vRayLight.getDistance() /
	// contributingSegment.getDistance();
	//
	// //Only use with very small ray count
	//// vizualizeRay(result, vRayLight.getVrlStart(), vRayLight.getVrlEnd(),
	// Color.RED, 5);
	//// vizualizeRay(result, contributingSegment.getStartPoint(),
	// contributingSegment.getEndPoint(), Color.GREEN, 5);
	//// vizualizeRay(result, rayLightPoint, cameraPoint, Color.BLUE, 10);
	//
	// lightPower += (energyChanged * transmitU * transmitW * lightPhaseFkt *
	// cameraPhaseFkt / pdf);
	// }
	//
	// //Each VRL-Cam Pair does count for 1 light therefore we only increase
	// afterwards
	// if (numberOfRays <= count) {
	// // Additional add since we otherwise miss the last contribution
	// synchronized (grid) {
	// detector.absorbPhoton(grid, randomDetectorPoint, (lightPower / (double)
	// virtualLightCount));
	// }
	// detectorHits++;
	// return;
	// }
	// }
	// synchronized (grid) {
	// detector.absorbPhoton(grid, randomDetectorPoint, (lightPower / (double)
	// virtualLightCount));
	// }
	// detectorHits++;
	// }
	// }

	
}
