/*
 * Copyright (C) 2014 - Andreas Maier, Fabian Rückert, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics.workingClass;

import java.awt.Color;
import java.util.Random;
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
import edu.stanford.rsl.tutorial.physics.NHelperFkt;
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;

/**
 * Implementation of the ray tracing and bidirectional path tracing algorithms for the calculation of indirect illumination
 * @author Tobias Miksch based on work of Fabian Rückert
 */
public class RayWorker extends AbstractWorker {

	private int bdptConnections = 2;
	
	public RayWorker(PriorityRayTracer raytracer, RaytraceResult res, long numRays, int startenergyEV, Grid2D grid,
			XRayDetector detector, Projection proj, boolean infinite, boolean writeAdditionalData,
			double sourceDetectorDist, double pixelDimensionX, double pixelDimensionY, Material background,
			int versionNumber, int threadNumber, Random random, int lightRayLength) {
		
		super(raytracer, res, numRays, startenergyEV, grid, detector, proj, infinite, writeAdditionalData,
				sourceDetectorDist, pixelDimensionX, pixelDimensionY, background, versionNumber, threadNumber, random, lightRayLength);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void run() {
		
		if (random == null) {
			this.sampler = new XRayTracerSampling(ThreadLocalRandom.current(), (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		} else {
			this.sampler = new XRayTracerSampling(random, (width * pixelDimX), (height * pixelDimY), normalOfDetector);
		}

		// Appliction of the constructed scheme
		double energyEV = startEnergyEV;
		for (; simulatedRays < (numRays);) {
			// print out progress after 2000 rays
			if (threadNumber == 1 && simulatedRays % 10000 == 0 && !infiniteSimulation) {
				double input = (simulatedRays * 100.0 * 100.0 / numRays) / 100.0;
				System.out.println(String.format("%2.2f", input)+ "%");
			}

/*		while(detectorHits < numRays) {
			if(threadNumber == 1 && simulatedRays % 10000 == 0 && !infiniteSimulation) {
				double input = (detectorHits * 100.0 * 100.0 / numRays) / 100.0;
				System.out.println(String.format("%2.2f", input)+ "%");
			}
*/
			
			//find a random point on the detector to determine the direction of the photon - Cone beam radiology
			SimpleVector randomVector = new SimpleVector(sampler.random() * width, sampler.random() * height);
			StraightLine ray = new StraightLine(emitterPoint, proj.computeRayDirection(randomVector));
			
			//Trace a photon through the scene until it is determinated
			raytrace(ray, energyEV, 0, 0);
			++simulatedRays;
		}
	}

	/**
	 * Trace a single photon until it is absorbed or it left the scene.
	 * 
	 * @param ray
	 *            The position and direction of the photon
	 * @param energyEV
	 *            The current energy of the photon in eV
	 * @param scatterCount
	 *            The number of times it already scattered
	 * @param totalDistance
	 *            The total traveled distance
	 * @param rayTraceResult
	 *            The RaytraceResult where the results from each step should be
	 *            saved to
	 */
	public void raytrace(StraightLine ray, double energyEV, int scatterCount, double totalDistance) {
		if (energyEV <= 1 || scatterCount > 200) {
			// should not happen
			System.out.println("energy low, times scattered: " + scatterCount);
			return;
		}

		int testingValue = lightRayLength;
		
		XRayHitVertex vertexHit = nextRayPoint(ray, energyEV);
		segmentsTraced++;
		
		//Ray scatters out of the scene
		if("background material".equals(vertexHit.getCurrentLineSegment().getNameString())) {
			return;
		}
		
		//Use this version to dismiss the contribution of direct lighting 
		if ("detector".equals(vertexHit.getCurrentLineSegment().getNameString())) {
			if(allowDirectLighting || (scatterCount != 0 && scatterCount <= testingValue)) {// != 0	
				if (!bdpt) {
					contribution[0] += energyEV;
					synchronized (grid) {
						detector.absorbPhoton(grid, vertexHit.getEndPoint(), energyEV);
					}
					detectorHits++;

//					if (showOutputonce) {
//						vizualizeRay(result, ray.getPoint(), vertexHit.getEndPoint(), Color.RED,(int) Math.round(vertexHit.getDistance()));
//					}
				}
			}
			return;
		}

		
		// choose compton or photoelectric effect -> Since we only simulate those 2 effects we have to assess them to reach a combined probability of 100%
		if (sampler.random() * (vertexHit.getPhotoAbsorption() + vertexHit.getComptonAbsorption()) <= vertexHit.getPhotoAbsorption()) {
			// photoelectric absorption (equals Russian Roulette)
			//scatteringNumber[1]++;
			return;
		} 
//		else {
//			scatteringNumber[2]++;
//		}

		
		
		//Here we use BPDT to get a connection to the detector using a shadow ray
		if (bdpt) { //scatterCount >= 1 &&
			//Design decision to only work with rays of length one from the detector since only participating media is concerned!
/*			boolean camPathLength = false;
			if (camPathLength && scatterCount <= (testingValue - 2)) {

				XRayHitVertex camPathVertex = createCamRay(energyEV, coneSample);
				StraightLine shadowRay = new StraightLine(vertexHit.getEndPoint(), camPathVertex.getEndPoint());
				shadowRay.normalize();
				double energyContrib = energyEV;
				
				double radLight  = nHelperFkt.getAngleInRad(shadowRay.getDirection(), vertexHit.getRayDir());
				double radCamera = nHelperFkt.getAngleInRad(shadowRay.getDirection(), camPathVertex.getRayDir().multipliedBy(-1.0));
				
				//Calc c(s,t)
				double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyContrib, radLight);
				energyContrib = XRayTracerSampling.getScatteredPhotonEnergy(energyContrib, radLight);
				double transShadowR = calculateTransmittance(vertexHit.getEndPoint(), camPathVertex.getEndPoint(),energyContrib);
				double camerPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyContrib, radCamera);
				double dist = sampler.gTerm(vertexHit.getEndPoint(), camPathVertex.getEndPoint(), false);
				double cST = lightPhaseFkt * dist * transShadowR * camerPhaseFkt;
				
				//Power of simulated camPath alpha(E,s)
				double pA = 1.0 / sampler.probabilityDensityOfArea();
				
				energyContrib = XRayTracerSampling.getScatteredPhotonEnergy(energyContrib, radCamera);
				
				double probabilitynoRR = 1.0 - (camPathVertex.getPhotoAbsorption() / (camPathVertex.getPhotoAbsorption() + camPathVertex.getComptonAbsorption()));		
				double alphaE2 = 1.0 / probabilitynoRR;
				
				double lightEnergyEV = energyContrib * pA * cST / camPathVertex.getDistanceFromStartingPoint();

				
//				if(show++ < 10 && threadNumber == 1) {
//				System.out.println("\nWe have a combination of the following values for a shadow ray: " 
//						+ "\n Energy: " + energyEV 
//						+ "\n Mutated energy: " + energyContrib 
//						+ "\n Resulting light: " + lightEnergyEV
//						+ "\n Angle on light: " + Math.toDegrees(radLight)
//						+ "\n Angle on camera: " + Math.toDegrees(radCamera)
//						+ "\n Phase at light: " + lightPhaseFkt
//						+ "\n Phase at camPa: " + camerPhaseFkt
//						+ "\n transmitShadow:  " + transShadowR
//						+ "\n transmitCamDir: " + transCamPath
//						+ "\n G-Term of Light: " + (1 / (dist * dist))
//						+ "\n Prob of no RR: " + probabilitynoRR
//						+ "\n G-Term of detec: " + g
//						
//					);
//				vizualizeRay(result, camCenter, vertexHit.getEndPoint(), Color.yellow, 20);
//				vizualizeRay(result, vertexHit.getEndPoint(), camPathVertex.getEndPoint(), Color.magenta, 20);
//				vizualizeRay(result, camPathVertex.getEndPoint(), camPathVertex.getStartPoint(), Color.cyan, 20);			
//				}

				contribution[1] += lightEnergyEV;
				synchronized (grid) {
					detector.absorbPhoton(grid, camPathVertex.getStartPoint(), lightEnergyEV);
				}
				detectorHits++;
			}
*/			
			if(scatterCount <= (testingValue - 1)) {
				boolean multiConnection = false;
				
				if (multiConnection) { //Array Index out of Bounds
					int sum = bdptConnections * bdptConnections;
					show++;
					for(int x = 0; x < bdptConnections; x++) {
						for(int y = 0; y < bdptConnections; y++) {

							SimpleVector randomVec = new SimpleVector( ((double) x + sampler.random()) /bdptConnections * width, ((double) y + sampler.random())/bdptConnections * height);
							
							PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec, sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
							StraightLine camRay = new StraightLine(randomDetectorPoint, vertexHit.getEndPoint());
							camRay.normalize();

							double theta = NHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0), vertexHit.getRayDir());
							double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

							double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
							double transmittance = calculateTransmittance(randomDetectorPoint, vertexHit.getEndPoint(), energy);
							double g = sampler.gTerm(randomDetectorPoint, vertexHit.getEndPoint(), true);
							double pA = 1.0 / sampler.probabilityDensityOfArea();

							double lightPower = (energy * transmittance * lightPhaseFkt * g * pA);
							
//							if(show < 10 && threadNumber == 2) {
//								System.out.println("Points of interest: " + randomVec
//										+ "Resulting in a single contribution of: "+ lightPower / sum);
//								show++;
//							}
							contribution[2] += lightPower / sum;
							synchronized (grid) {
								detector.absorbPhoton(grid, randomDetectorPoint, lightPower / sum);
							}
							segmentsTraced++;
						}
						detectorHits++;
//						simulatedRays++;
					}
				} else {
					SimpleVector randomVec = new SimpleVector(sampler.random() * width, sampler.random() * height);
					PointND randomDetectorPoint = proj.computeDetectorPoint(emitterPoint.getAbstractVector(), randomVec,
							sourceDetectorDistance, pixelDimX, pixelDimY, width, height);
					StraightLine camRay = new StraightLine(randomDetectorPoint, vertexHit.getEndPoint());
					camRay.normalize();

					double theta = NHelperFkt.getAngleInRad(camRay.getDirection().multipliedBy(-1.0),
							vertexHit.getRayDir());
					double lightPhaseFkt = XRayTracerSampling.comptonAngleCrossSection(energyEV, theta);

					double energy = XRayTracerSampling.getScatteredPhotonEnergy(energyEV, theta);
					double transmittance = calculateTransmittance(randomDetectorPoint, vertexHit.getEndPoint(), energy);
					double g = sampler.gTerm(randomDetectorPoint, vertexHit.getEndPoint(), true);
					double pA = 1.0 / sampler.probabilityDensityOfArea();

					double lightPower = (energy * transmittance * lightPhaseFkt * g * pA);
					contribution[2] += lightPower;
					synchronized (grid) {
						detector.absorbPhoton(grid, randomDetectorPoint, lightPower);
					}
					detectorHits++;
					segmentsTraced++;
//					simulatedRays++;
				}
			}
		}

		// Compton effect -> calculate new energy and new direction
//		for(int k=0; k < 2; ++k)
		{	
			SimpleVector dir = ray.getDirection().clone();
			double changedEnergyEV = sampler.sampleComptonScattering(energyEV, dir);
			
//			writeStepResults(result, changedEnergyEV, vertexHit.getDistanceFromStartingPoint(), vertexHit.getEndPoint(), null, Color.CYAN);
			StraightLine newRay = new StraightLine(vertexHit.getEndPoint(), dir);
			raytrace(newRay, changedEnergyEV, scatterCount + 1, totalDistance + vertexHit.getDistance());
		}
	}
	
}
