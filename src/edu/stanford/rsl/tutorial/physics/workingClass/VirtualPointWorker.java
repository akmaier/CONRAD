/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics.workingClass;

import java.awt.Color;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ThreadLocalRandom;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;
import edu.stanford.rsl.tutorial.physics.XRayHitVertex;
import edu.stanford.rsl.tutorial.physics.XRayTracerSampling;
import edu.stanford.rsl.tutorial.physics.XRayVPL;
import edu.stanford.rsl.tutorial.physics.NHelperFkt;
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;

/**
 * Implementation of the two step algorithm using virtual point lights for the calculation of indirect illumination
 * @author Tobias Miksch
 */
public class VirtualPointWorker extends AbstractWorker {

	
	// For VPL
	private double vplCreation = 0.0;
	private double vplContrib = 0.5;
	private boolean vplByNumber = true;
	private double percentOfVPLs = (1.0 / 16.0);// 0.125 / 4;//8 Threads -> 0.125 = number per Thread
	
	private int vplPathNumber = 0;
	private int virtualLightsPerThread;
	
	private volatile int attempting[];
	private final CyclicBarrier barrierVPL;
	private volatile XRayVPL collectionOfAllVPLs[];
	
	public VirtualPointWorker(PriorityRayTracer raytracer, RaytraceResult res, long numRays, int startenergyEV,
			Grid2D grid, XRayDetector detector, Projection proj, boolean infinite, boolean writeAdditionalData,
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
			this.sampler = new XRayTracerSampling(ThreadLocalRandom.current(), (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		} else {
			this.sampler = new XRayTracerSampling(random, (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		}
		
		if (vpl) {
			XRayVPL collection[] = new XRayVPL[virtualLightsPerThread];
			simulatedRays = 0;
			// Set up VPLs
			collection = createVPLPoints();
			// collection = testVPLPoints();

			for (int i = 0; i < virtualLightsPerThread; ++i) {
				collectionOfAllVPLs[threadNumber * virtualLightsPerThread + i] = collection[i];
			}

			try {
				barrierVPL.await();
			} catch (InterruptedException e) {
				System.err.println("barrierVPL interrupted!");
				e.printStackTrace();
			} catch (BrokenBarrierException e) {
				System.err.println("barrierVPL broken!");
				e.printStackTrace();
			}
			// System.out.println("Feedback of Thread: " + Thread.currentThread().getId() + " barrier was succesfull");

			for (int i = 0; i < attempting.length; i++) {
				vplPathNumber += attempting[i];
			}
			
			if(threadNumber == 1) {
				System.out.println("For the creation of " + collectionOfAllVPLs.length +  " the system had to use " + vplPathNumber + " attemps.");
				
				//	for(int i = 0; i < 11; ++i) {
//					int sum = 0;
//					for (int j = 0; j < collectionOfAllVPLs.length; j++) {
//						if(collectionOfAllVPLs[j].getScattercount() == i)
//							sum++;
//					}
//					if(sum != 0)
//						System.out.println("Vpls with scattercount " + i + " is " + sum + " and resulting in: " + (double)sum / collectionOfAllVPLs.length);
//				}
			}

			// We split the number of simulated rays up so that a only fraction simulates compton scattering
			
			long restCount = (long) (numRays - simulatedRays);
			long distribution = (long) (restCount * vplContrib);

			if (threadNumber == 0) {
				System.out.println("Feedback of Thread: " + Thread.currentThread().getId() + " barrier was succesfull");

				System.out.println("Tracing with " + (restCount - distribution) + " rays of direct lighting and "
						+ distribution + " vpl-illumination");
				System.out.println("Number of VPLs: " + collectionOfAllVPLs.length + " testing per pixel: "	+ (collectionOfAllVPLs.length * getPercentOfVPLs()) 
						+ "\nVPL path # total:" + vplPathNumber	+ " resulting in = " + (collectionOfAllVPLs.length / (double) vplPathNumber) 
						+ "\n or scaled " + (collectionOfAllVPLs.length / (double) vplPathNumber)	* collectionOfAllVPLs[0].getLightpower());
			}

			collectDirectLighting(restCount - distribution);
			collectVPLScattering(distribution, collection, getPercentOfVPLs());
			
			simulatedRays *= vplContrib;
			return;
		}
	}

	//TODO: Visibility is package because of TestCrontrolUnit
	XRayVPL[] createVPLPoints() {

		XRayVPL collection[] = new XRayVPL[virtualLightsPerThread];

		int vplCount = 0;
		int attempts = 0;

		while (true) {
			// prevent infinite loop if there is no participating media
			if (vplCount == 0 && attempts == 100000) {
				System.err.println("Could not create a single virtual ray light!");
				throw new RuntimeException("vplNumber == 0");
			}

			SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
//			SimpleVector randomVector = new SimpleVector(0.5 * width, 0.5 * height);

			StraightLine ray = null;
			// if (vplCreation == 0.0) {
			ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));
			// } else {
			// SimpleVector evalDir = sampler.samplingDirection(proj.computeRayDirection(randomVector), vplCreation);
			// ray = new StraightLine(emitterPoint, evalDir);
			// }

			double currentEnergyEV = startEnergyEV;
			for (int i = 0; i < lightRayLength; ++i) {
				
				XRayHitVertex vertexHit = nextRayPoint(ray, currentEnergyEV);
				if(i == 0)
					attempts++;

				// Ray scatters out of the scene
				if ("background material".equals(vertexHit.getCurrentLineSegment().getNameString())) {
					break;
				}
				// if the ray hit the detector box, write the result to the grid
				if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
					if (allowDirectLighting) {
						synchronized (grid) {
							detector.absorbPhoton(grid, vertexHit.getEndPoint(), currentEnergyEV);
						}
//						detectorHits++;
//						simulatedRays++;
					}
					break;
				}
				
				writeStepResults(result, currentEnergyEV, vertexHit.getDistance(), vertexHit.getEndPoint(), null, Color.CYAN);
				double probNoRR = Math.pow((1.0 - (vertexHit.getPhotoAbsorption() / (vertexHit.getPhotoAbsorption() + vertexHit.getComptonAbsorption()))), 1.0);//TODO: +i

				collection[vplCount] = new XRayVPL(vertexHit.getEndPoint(), currentEnergyEV, i, ray.getPoint());
				collection[vplCount].setLightpower(probNoRR);
				++vplCount;
				
				// return asap enough vpls created or else index out of bounds exception
				if (vplCount >= virtualLightsPerThread) {
					attempting[threadNumber] = attempts;
					virtualLightSourceAttempts = attempts;
					return collection;
				}

				if (sampler.random() * (vertexHit.getPhotoAbsorption() + vertexHit.getComptonAbsorption()) <= vertexHit.getPhotoAbsorption()) {
					break;
				} else {
					SimpleVector dir = ray.getDirection().clone();
					currentEnergyEV = sampler.sampleComptonScattering(currentEnergyEV, dir);
					ray = new StraightLine(vertexHit.getEndPoint(), dir);
				}
			}
		}
	}
	

	/**
	 * Cast Rays from the detector towards the vpls and collect all the energy the
	 * VPLs contain while scattering them at randomInteractions
	 */
	void collectVPLScattering(long numberOfRays, XRayVPL collection[], double factor) {
		if (numberOfRays <= 0) {
			System.err.println("VPL scattering wont collect any energy!");
			return;
		}

		// Percentage of all VPLs we connect to any given detector point
		int vplSamplingCount;
		if(vplByNumber == true) {
			vplSamplingCount = 3;
		} else {
			vplSamplingCount = (int) (collectionOfAllVPLs.length * factor);
		}

		if(vplSamplingCount < 1) {
			System.out.println("The sampling count for the vpls was 0 or less! Changed to 1.");
			vplSamplingCount = 1;
		}
		
		show = 0;
		for (long count = numberOfRays + simulatedRays; simulatedRays < count; ++simulatedRays) {

			// Get random detector point
			SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
			PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
					sourceDetectorDistance, pixelDimX, pixelDimY, width, height);

			// Probability of choosing exactly this sample -> random value from X elements =
			// 1 / X
			
//			if(threadNumber == 1 && (simulatedRays % 1000 == 0)) {
//				System.out.println("Still working on this.");
//			}
			
			double pdf = (double) vplSamplingCount / (double) collectionOfAllVPLs.length;
			double lightPower = 0;

			for (int i = 0; i < vplSamplingCount; ++i) {
				// Get a random VPL
				segmentsTraced++;
				int x = i;
				if (factor != 1.0) {
					x = sampler.random(collectionOfAllVPLs.length);
				}
				XRayVPL vpLight = collectionOfAllVPLs[x];

				// Calculate the contribution of the VPL - transmittance and angleProbability
				StraightLine camera = new StraightLine(randomDetectorPoint, vpLight.getVplPos());
				camera.normalize();

				double theta = NHelperFkt.getAngleInRad(camera.getDirection().multipliedBy(-1.0), vpLight.getDirection()); // .multipliedBy(-1.0)
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(vpLight.getEnergyEV(), theta);

				double energy = XRayTracerSampling.getScatteredPhotonEnergy(vpLight.getEnergyEV(), theta);
				double transmittance = calculateTransmittance(randomDetectorPoint, vpLight.getVplPos(), energy);
				double g = sampler.gTerm(randomDetectorPoint, vpLight.getVplPos(), true);
				double pA = sampler.probabilityDensityOfArea();

				double scalingOfVPL = vpLight.getLightpower() / ((double) vplPathNumber) / pdf;
				lightPower += ((energy * scalingOfVPL) * transmittance * lightPhaseFkt * g / pA);
				scatteringNumber[vpLight.getScattercount() + 1]++;
				contribution[vpLight.getScattercount()+1] += ((energy * scalingOfVPL) * transmittance * lightPhaseFkt * g / pA);
				// vplPathNumber ) * transmittance * lightPhaseFkt * g) / pdf);

				if (showOutputonce && show < 1) {
					{
						show++;
						vizualizeRay(result, vpLight.getVplPos(), randomDetectorPoint, Color.green, 20);

						System.out.println("\nValues of point: " 
								+ "\n VplEnergy: " + vpLight.getEnergyEV()
								+ "\n VpLightPower: " + vpLight.getLightpower() 
								+ "\n ScatterCount: " + vpLight.getScattercount() 
								+ "\n Angle Theta: " + Math.toDegrees(theta)
								+ "\n PhaseFkt:   " + lightPhaseFkt 
								+ "\n Throughput: " + transmittance 
								+ "\n G-Term: "	+ g 
								+ "\n VplSamplingCount: " + vplSamplingCount 
								+ "\n Vpl pdf: " + pdf 
								+ "\n Energy: " + (energy * vpLight.getLightpower() / ((double) vplPathNumber) / pdf) 
								+ "\n Result: "	+ (lightPower / pdf / collectionOfAllVPLs.length));
					}
				}
			}
//			contribution[3] += lightPower;
			synchronized (grid) {
				detector.absorbPhoton(grid, randomDetectorPoint, lightPower);
				detectorHits++;
			}
		}
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
		for(int i = 0; i < width; ++i) {
			for(int j = 0; j < height; ++j) {
				SimpleVector randomVec = new SimpleVector(i, j);
				PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
				double throughput = calculateTransmittance(emitterPoint, randomDetectorPoint, startEnergyEV);

				contribution[0] += throughput * startEnergyEV;
				synchronized (grid) {
					detector.absorbPhoton(grid, randomDetectorPoint, throughput * startEnergyEV);
				}
				detectorHits++;
			}
		}

		for (long count = simulatedRays + numberOfRays - (width*height); simulatedRays < count; ++simulatedRays) {
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

	public double getPercentOfVPLs() {
		return percentOfVPLs;
	}

	public void setPercentOfVPLs(double percentOfVPLs) {
		this.percentOfVPLs = percentOfVPLs;
	}
	
}
