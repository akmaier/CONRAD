package edu.stanford.rsl.tutorial.physics;

import java.awt.Color;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.bounds.BoundingBox;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
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
import edu.stanford.rsl.tutorial.physics.XRayTracer.RaytraceResult;

/**
 * Raytraces x-rays and saves the results.
 * 
 * @author Fabian Rückert
 */

public class XRayWorker implements Runnable {

	private StraightLine startRay;
	private AbstractRayTracer raytracer;
	private RaytraceResult result;
	private int numRays;
	private int startEnergyEV;
	private Grid2D grid;
	XRayDetector detector;
	private boolean writeAdditionalData = true;
	Projection proj;

	//set a ThreadLocalRandom generator for improved performance, 
	//however you can't set the random seed explicitly for this one and it seems to produce the same random numbers every time for a thread
	//TODO use a better random number generator
	XRayTracerSampling sampler = new XRayTracerSampling(ThreadLocalRandom.current());
	private boolean infiniteSimulation;

	//variables for statistics
	private volatile long simulatedRays = 0;
	private volatile long detectorHits = 0;

	public XRayWorker(StraightLine startRay, PriorityRayTracer raytracer, RaytraceResult res, int numRays,
			int startenergyEV, Grid2D grid, XRayDetector detector, Projection proj, boolean infinite,
			boolean writeAdditionalData) {
		this.startRay = startRay;
		this.raytracer = raytracer;
		this.result = res;
		this.numRays = numRays;
		this.startEnergyEV = startenergyEV;
		this.grid = grid;
		this.detector = detector;
		this.proj = proj;
		this.infiniteSimulation = infinite;
		this.writeAdditionalData = writeAdditionalData;

		if (infiniteSimulation) {
			if (writeAdditionalData) {
				System.err.println("Warning: Not saving additional data in infinite simulation.");
				this.writeAdditionalData = false;
			}
		}

	}

	/**
	 * Simulate the number of photons this thread has been assigned to simulate.
	 */
	@Override
	public void run() {
		//if infiniteSimulation send infinite rays until aborted by the user
		for (simulatedRays = 0; simulatedRays < numRays || infiniteSimulation; ++simulatedRays) {
			//print out progress after 2000 rays
			if (!infiniteSimulation && simulatedRays % 2000 == 0 && simulatedRays != 0) {
				System.out.println((float) (simulatedRays * 100 * 100 / numRays) / 100 + "%");
			}

			if (XRayTracer.TEST_MODE) {
				//send every photon in the direction of the startray
				raytrace(startRay, startEnergyEV, 0, 0);
			} else {
				//send a ray to a random pixel of the detector
				StraightLine ray = new StraightLine(startRay.getPoint(), proj.computeRayDirection(
						new SimpleVector(sampler.random() * grid.getWidth(), sampler.random() * grid.getHeight())));
				raytrace(ray, startEnergyEV, 0, 0);
			}

		}
	}

	public long getRayCount() {
		//a lock is not needed as reading of a volatile long is atomic
		return simulatedRays;
	}

	public long getDetectorHits() {
		//a lock is not needed as reading of a volatile long is atomic
		return detectorHits;
	}

	/**
	 * Combine the results of this worker with the finalresult
	 * @param finalResult	The combined result
	 */
	public void addResults(RaytraceResult finalResult) {
		finalResult.addResults(result);
	}

	/**
	 * Trace a single photon until it is absorbed or it left the scene.
	 * @param ray The position and direction of the photon
	 * @param energyEV The current energy of the photon in eV
	 * @param scatterCount The number of times it already scattered
	 * @param totalDistance	The total traveled distance
	 * @param rayTraceResult	The RaytraceResult where the results from each step should be saved to
	 */
	private void raytrace(StraightLine ray, double energyEV, int scatterCount, double totalDistance) {
		if (energyEV <= 1 || scatterCount > 20000) {
			//should not happen
			System.out.println("energy low, times scattered: " + scatterCount);
			return;
		}

		//check if ray is inside the scene limits
		BoundingBox bb = new BoundingBox(raytracer.getScene().getMin(), raytracer.getScene().getMax());
		if (!bb.isSatisfiedBy(ray.getPoint())) {
			return;
		}

		ArrayList<PhysicalObject> physobjs = raytracer.castRay(ray);
		if (physobjs == null) {
			//should not happen because ray is inside of the bounding box
			//return;
			System.err.println("No background material set?");
			throw new RuntimeException("physobjs == null");
		}

		// check if the direction of the line segments is the same direction of they ray or the opposite direction
		boolean reversed = false;

		Edge e1 = (Edge) physobjs.get(0).getShape();
		SimpleVector diff = e1.getEnd().getAbstractVector().clone();
		diff.subtract(e1.getPoint().getAbstractVector());
		diff.normalizeL2();
		diff.subtract(ray.getDirection());

		if (diff.normL2() > CONRAD.SMALL_VALUE) {
			reversed = true;
		}

		double totalDistUntilNextInteraction = 0;
		Material currentMaterial = null;
		PointND rayPoint = ray.getPoint().clone();

		// pass through materials until an interaction is happening
		PhysicalObject currentLineSegment = null;
		PointND end = null;
		double photoAbsorption = 0;
		double comptonAbsorption = 0;
		boolean foundStartSegment = false;
		// the line segments are sorted, we still need to find the first line segment containing the point
		for (int i = 0; i < physobjs.size(); ++i) {
			currentLineSegment = physobjs.get(reversed ? physobjs.size() - i - 1 : i);
			double distToNextMaterial = Double.NaN;

			Edge e = (Edge) currentLineSegment.getShape();
			PointND start = reversed ? e.getEnd() : e.getPoint();
			end = reversed ? e.getPoint() : e.getEnd();

			if (foundStartSegment
					|| isBetween(start.getAbstractVector(), end.getAbstractVector(), rayPoint.getAbstractVector())) {
				// get length between the raypoint and the next material in ray direction
				distToNextMaterial = rayPoint.euclideanDistance(end);

			} else {
				// continue with next line segment
				continue;
			}

			// if the raypoint is very close to the endpoint continue with next line segment
			if (distToNextMaterial > CONRAD.SMALL_VALUE) {
				// get distance until next physics interaction
				currentMaterial = currentLineSegment.getMaterial();

				//this lookup slows down the RayTracer because it is synchronized
				photoAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.PHOTOELECTRIC_ABSORPTION);
				comptonAbsorption = currentMaterial.getAttenuation(energyEV / 1000,
						AttenuationType.INCOHERENT_ATTENUATION);

				double distToNextInteraction = 10
						* sampler.getDistanceUntilNextInteractionCm(photoAbsorption, comptonAbsorption);
				if (distToNextInteraction < distToNextMaterial) {
					// ray interacts in current line segment, move the photon to the interaction point
					SimpleVector p = rayPoint.getAbstractVector();
					p.add(ray.getDirection().multipliedBy(distToNextInteraction));
					totalDistUntilNextInteraction += distToNextInteraction;
					break;
				} else {
					// ray exceeded boundary of current material, continue with next segment
					rayPoint = new PointND(end);
					totalDistUntilNextInteraction += distToNextMaterial;

				}
			}

			// take the next line segment
			foundStartSegment = true;

		}

		if (currentMaterial == null) {
			//ray left the scene
			return;
		}

		writeStepResults(result, energyEV, totalDistUntilNextInteraction, rayPoint,
				new Edge(ray.getPoint().clone(), rayPoint.clone()));

		//if the ray hit the detector box, write the result to the grid
		if (currentLineSegment.getNameString() != null && currentLineSegment.getNameString().equals("detector")) {
			synchronized (grid) {
				((XRayDetector) currentLineSegment.getParent()).absorbPhoton(grid, rayPoint, energyEV);
			}
			detectorHits++;
			return;
		}

		// choose compton or photoelectric effect
		if (sampler.random() * (photoAbsorption + comptonAbsorption) <= photoAbsorption) {
			// photoelectric absorption
			energyEV = 0;
			return;
		} else {
			// compton effect
			// calculate new energy and new direction
			SimpleVector dir = ray.getDirection().clone();

			if (XRayTracer.SAMPLE_GEANT4) {
				energyEV = sampler.sampleComptonScatteringGeant4(energyEV, dir);
			} else {
				energyEV = sampler.sampleComptonScattering(energyEV, dir);
			}

			StraightLine newRay = new StraightLine(rayPoint, dir);
			// send new ray
			raytrace(newRay, energyEV, scatterCount + 1, totalDistance + totalDistUntilNextInteraction);
		}

	}

	/**
	 * Check if a point x lies between the points a and b, when it is already know that all three points are on a straight line.
	 * @return If x is between a and b
	 */
	private static boolean isBetween(SimpleVector a, SimpleVector b, SimpleVector x) {
		SimpleVector bminusa = b.clone();
		bminusa.subtract(a);

		SimpleVector xminusa = x.clone();
		xminusa.subtract(a);

		double dot = SimpleOperators.multiplyInnerProd(bminusa, xminusa);
		if (dot < 0)
			return false;

		double squaredlength = bminusa.getElement(0) * bminusa.getElement(0)
				+ bminusa.getElement(1) * bminusa.getElement(1) + bminusa.getElement(2) * bminusa.getElement(2);
		return dot <= squaredlength;
	}

	/**
	 * Saves the results for this ray interaction
	 * @param result	The RaytraceResult where the the current result should be saved to
	 * @param energyEV	The energy in eV
	 * @param totalDistUntilNextInteraction The total path length for this step
	 * @param rayPoint The point of interaction
	 * @param e	The edge between the last interaction point and the current one
	 */
	private void writeStepResults(RaytraceResult result, double energyEV, double totalDistUntilNextInteraction,
			PointND rayPoint, Edge e) {

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
		float factor = (((float) energyEV / startEnergyEV));
		result.colors.add(new Color(factor, 1 - factor, 0));

		result.edges.add(e);
	}

}
