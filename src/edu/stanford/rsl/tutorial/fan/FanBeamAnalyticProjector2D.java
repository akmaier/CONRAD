package edu.stanford.rsl.tutorial.fan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.rendering.PriorityRayTracer;

/**
 * Analytic (exact ray-shape-intersection) counterpart of {@link FanBeamProjector2D}.
 * <p>
 * Instead of sampling a rasterized grid along each ray, this projector casts each
 * fan-beam ray through a {@link PrioritizableScene} of analytic shapes with a
 * {@link PriorityRayTracer} and accumulates the exact intersection path length per
 * material (using {@link XRayDetector#accumulatePathLenghtForEachMaterial}). Object
 * priority / overlap (e.g. inserts overriding a body) is resolved by the ray tracer,
 * so no hand-rolled "body minus inserts" bookkeeping is needed.
 * <p>
 * The view/detector geometry is identical to
 * {@link FanBeamProjector2D#projectRayDriven(Grid2D)} (same source, detector line,
 * bin centering and ray construction), so the resulting per-material path-length
 * sinograms are drop-in compatible with the {@code tutorial.fan} reconstruction
 * chain (CosineFilter -&gt; RamLakKernel -&gt; FanBeamBackprojector2D).
 * <p>
 * Output is a {@link MultiChannelGrid2D} sized {@code [maxTIndex, maxBetaIndex]} with
 * ONE CHANNEL PER MATERIAL (path length in mm). The stable material-&gt;channel index
 * map is exposed via {@link #getMaterials()} / {@link #getMaterialIndex(Material)} and
 * the grid's channel names via {@code getChannelNames()}.
 *
 * @author Andreas Maier
 */
public class FanBeamAnalyticProjector2D {

	private double focalLength, maxBeta, deltaBeta, maxT, deltaT;
	int maxTIndex, maxBetaIndex;

	/** Stable material -&gt; channel index map (insertion order). */
	private final LinkedHashMap<Material, Integer> materialChannelMap = new LinkedHashMap<Material, Integer>();

	/**
	 * Creates a new instance of an analytic fan-beam projector. Constructor is
	 * identical to {@link FanBeamProjector2D}.
	 *
	 * @param focalLength the focal length (source-to-isocenter distance)
	 * @param maxBeta     the maximal rotation angle
	 * @param deltaBeta   the step size between source positions
	 * @param maxT        the length of the (virtual) detector array
	 * @param deltaT      the size of one detector element
	 */
	public FanBeamAnalyticProjector2D(double focalLength, double maxBeta, double deltaBeta, double maxT, double deltaT) {
		this.focalLength = focalLength;
		this.maxBeta = maxBeta;
		this.maxT = maxT;
		this.deltaBeta = deltaBeta;
		this.deltaT = deltaT;
		this.maxBetaIndex = (int) (maxBeta / deltaBeta);
		this.maxTIndex = (int) (maxT / deltaT);
	}

	/**
	 * Collect the distinct materials present in the scene, preserving the scene's
	 * (insertion) order, and build the stable material-&gt;channel index map.
	 *
	 * @param scene the scene
	 * @return an array of channel names (material names) in channel order
	 */
	private String[] buildMaterialMap(PrioritizableScene scene) {
		materialChannelMap.clear();
		for (PhysicalObject o : scene) {
			Material m = o.getMaterial();
			if (m != null && !materialChannelMap.containsKey(m)) {
				materialChannelMap.put(m, materialChannelMap.size());
			}
		}
		String[] names = new String[materialChannelMap.size()];
		for (Map.Entry<Material, Integer> e : materialChannelMap.entrySet()) {
			names[e.getValue()] = e.getKey().getName();
		}
		return names;
	}

	/**
	 * Analytically project the scene, producing a per-material path-length
	 * sinogram (path length in mm) as a {@link MultiChannelGrid2D} sized
	 * {@code [maxTIndex, maxBetaIndex]} with one channel per material.
	 * <p>
	 * The view/detector loop is a verbatim copy of
	 * {@link FanBeamProjector2D#projectRayDriven(Grid2D)}, but the grid-sampling
	 * block is replaced by analytic ray tracing.
	 *
	 * @param scene the scene of analytic shapes with materials
	 * @return the multi-channel path-length sinogram [mm]
	 */
	public MultiChannelGrid2D projectRayDrivenMaterials(PrioritizableScene scene) {
		String[] channelNames = buildMaterialMap(scene);
		int numMaterials = Math.max(channelNames.length, 1);

		MultiChannelGrid2D sino = new MultiChannelGrid2D(maxTIndex, maxBetaIndex, numMaterials);
		sino.setChannelNames(channelNames);
		sino.setSpacing(deltaT, deltaBeta);

		PriorityRayTracer rt = new PriorityRayTracer();
		rt.setScene(scene);

		// iterate over the rotation angle
		for (int i = 0; i < maxBetaIndex; i++) {
			// compute the current rotation angle and its sine and cosine
			double beta = deltaBeta * i;
			double cosBeta = Math.cos(beta);
			double sinBeta = Math.sin(beta);
			// compute source position
			PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta, 0.d);
			// compute end point of detector
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta, 0.d);

			// create an unit vector that points along the detector
			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			dirDetector.normalizeL2();

			// iterate over the detector elements
			for (int t = 0; t < maxTIndex; t++) {
				// calculate current bin position
				// the detector elements' position are centered
				double stepsDirection = 0.5f * deltaT + t * deltaT;
				PointND p = new PointND(p0);
				p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));

				// create a straight line between detector bin and source
				StraightLine line = new StraightLine(a, p);

				// analytic ray tracing: exact intersection segments through the scene
				ArrayList<PhysicalObject> segments = rt.castRay(line);
				if (segments == null || segments.isEmpty()) {
					continue;
				}

				// accumulate the exact path length per material (priority already resolved)
				HashMap<Material, Double> paths = XRayDetector.accumulatePathLenghtForEachMaterial(segments);
				for (Map.Entry<Material, Double> e : paths.entrySet()) {
					Integer channel = materialChannelMap.get(e.getKey());
					if (channel != null) {
						sino.putPixelValue(t, i, channel, e.getValue().floatValue());
					}
				}
			}
		}
		return sino;
	}

	/**
	 * @return the ordered list of materials, index == channel in the
	 *         {@link MultiChannelGrid2D} produced by the last
	 *         {@link #projectRayDrivenMaterials(PrioritizableScene)} call.
	 */
	public List<Material> getMaterials() {
		Material[] arr = new Material[materialChannelMap.size()];
		for (Map.Entry<Material, Integer> e : materialChannelMap.entrySet()) {
			arr[e.getValue()] = e.getKey();
		}
		ArrayList<Material> list = new ArrayList<Material>();
		for (Material m : arr) {
			list.add(m);
		}
		return list;
	}

	/**
	 * @param m a material
	 * @return the channel index for the given material, or -1 if unknown
	 */
	public int getMaterialIndex(Material m) {
		Integer c = materialChannelMap.get(m);
		return c == null ? -1 : c.intValue();
	}

	// ---------------------------------------------------------------------
	// EID / PCD combination using CONRAD physics (Material.getAttenuation).
	// The per-material path lengths above are in mm; CONRAD attenuation is in
	// 1/cm, so path lengths are converted to cm (*0.1) here.
	// ---------------------------------------------------------------------

	/**
	 * Energy-integrating detector (EID) line-integral sinogram.
	 * <p>
	 * For a polychromatic spectrum {@code {E, w(E)}} (weights need not be
	 * normalized) the detected signal per ray is
	 * {@code S = sum_E w(E)*E*exp(-sum_material mu_material(E)*L_material)} and the
	 * output is {@code p_eid = -log(S / S_air)} with
	 * {@code S_air = sum_E w(E)*E}.
	 *
	 * @param materialSino per-material path length sinogram [mm] (from this projector)
	 * @param energies     spectrum bin energies [keV]
	 * @param weights      spectrum bin photon fluxes (arbitrary units)
	 * @return the EID line-integral sinogram [maxTIndex, maxBetaIndex]
	 */
	public Grid2D combineEID(MultiChannelGrid2D materialSino, double[] energies, double[] weights) {
		List<Material> mats = getMaterials();
		int nE = energies.length;

		// precompute per-material mu(E) [1/cm]
		double[][] mu = new double[mats.size()][nE];
		AttenuationType at = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;
		for (int m = 0; m < mats.size(); m++) {
			for (int j = 0; j < nE; j++) {
				mu[m][j] = mats.get(m).getAttenuation(energies[j], at);
			}
		}
		double sAir = 0.0;
		for (int j = 0; j < nE; j++) {
			sAir += weights[j] * energies[j];
		}

		Grid2D out = new Grid2D(maxTIndex, maxBetaIndex);
		out.setSpacing(deltaT, deltaBeta);
		for (int i = 0; i < maxBetaIndex; i++) {
			for (int t = 0; t < maxTIndex; t++) {
				double s = 0.0;
				for (int j = 0; j < nE; j++) {
					double tau = 0.0;
					for (int m = 0; m < mats.size(); m++) {
						double Lcm = 0.1 * materialSino.getPixelValue(t, i, m); // mm -> cm
						tau += mu[m][j] * Lcm;
					}
					s += weights[j] * energies[j] * Math.exp(-tau);
				}
				double p = -Math.log(Math.max(s, 1e-12) / sAir);
				out.setAtIndex(t, i, (float) p);
			}
		}
		return out;
	}

	/**
	 * Photon-counting detector (PCD) per-bin line-integral sinograms.
	 * <p>
	 * The spectrum {@code {E, w(E)}} is partitioned into energy bins by
	 * {@code binEdges} (length nBins+1). Per bin {@code b} the detected count is
	 * {@code C_b = sum_{E in bin} w(E)*exp(-sum mu*L)} and the output is
	 * {@code -log(C_b / C_air_b)}.
	 *
	 * @param materialSino per-material path length sinogram [mm]
	 * @param energies     spectrum bin energies [keV]
	 * @param weights      spectrum bin photon fluxes
	 * @param binEdges     energy bin edges [keV], length nBins+1, ascending
	 * @return an array of nBins line-integral sinograms
	 */
	public Grid2D[] combinePCD(MultiChannelGrid2D materialSino, double[] energies, double[] weights, double[] binEdges) {
		List<Material> mats = getMaterials();
		int nE = energies.length;
		int nBins = binEdges.length - 1;

		double[][] mu = new double[mats.size()][nE];
		AttenuationType at = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;
		for (int m = 0; m < mats.size(); m++) {
			for (int j = 0; j < nE; j++) {
				mu[m][j] = mats.get(m).getAttenuation(energies[j], at);
			}
		}
		// bin assignment for each energy
		int[] binOf = new int[nE];
		double[] cAir = new double[nBins];
		for (int j = 0; j < nE; j++) {
			int b = -1;
			for (int k = 0; k < nBins; k++) {
				if (energies[j] >= binEdges[k] && energies[j] < binEdges[k + 1]) {
					b = k;
					break;
				}
			}
			binOf[j] = b;
			if (b >= 0) {
				cAir[b] += weights[j];
			}
		}

		Grid2D[] out = new Grid2D[nBins];
		for (int b = 0; b < nBins; b++) {
			out[b] = new Grid2D(maxTIndex, maxBetaIndex);
			out[b].setSpacing(deltaT, deltaBeta);
		}
		for (int i = 0; i < maxBetaIndex; i++) {
			for (int t = 0; t < maxTIndex; t++) {
				double[] c = new double[nBins];
				for (int j = 0; j < nE; j++) {
					int b = binOf[j];
					if (b < 0) {
						continue;
					}
					double tau = 0.0;
					for (int m = 0; m < mats.size(); m++) {
						double Lcm = 0.1 * materialSino.getPixelValue(t, i, m);
						tau += mu[m][j] * Lcm;
					}
					c[b] += weights[j] * Math.exp(-tau);
				}
				for (int b = 0; b < nBins; b++) {
					double p = -Math.log(Math.max(c[b], 1e-12) / Math.max(cAir[b], 1e-12));
					out[b].setAtIndex(t, i, (float) p);
				}
			}
		}
		return out;
	}
}
/*
 * Copyright (C) 2010-2024 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
