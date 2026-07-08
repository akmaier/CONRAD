/*
 * Copyright (C) 2026 - Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;

/**
 * GPU energy-resolving photon-counting detector (PCD). Sorts transmitted photons
 * into energy bins defined by {@code binEdges} [keV] and returns one count image
 * per bin: counts_b(p) = sum_{E in bin b} n_E(p), with
 * n_E(p) = flux_E * exp(-sum_m mu[m,E] * pathlen[m,p]). When noise is enabled the
 * counts are Poisson-sampled per energy before binning (correct per-bin Poisson
 * statistics). Distinct from {@link OpenCLEnergyIntegratingDetector}: this is a
 * photon-counting read-out (no energy weighting), so it is a separate class.
 *
 * On-device analogue of
 * {@link edu.stanford.rsl.conrad.physics.photoncounting.PhotonCountingEnergyResolvingDetector}.
 */
public class OpenCLPhotonCountingDetector extends OpenCLSpectralDetector {

	private static final int MAX_BINS = 16;   // matches the kernel's private accumulator

	public OpenCLPhotonCountingDetector(PolychromaticXRaySpectrum spectrum, Material[] materials) {
		super(spectrum, materials);
	}

	/** Noise-free per-bin count images for the given bin edges [keV]. */
	public MultiChannelGrid2D project(MultiChannelGrid2D pathLengths, double[] binEdges) {
		return project(pathLengths, binEdges, false, 0);
	}

	/**
	 * @param pathLengths per-material path lengths [mm] (channel order == materials)
	 * @param binEdges    energy thresholds [keV], length nBins+1, ascending
	 * @param noise       apply per-energy Poisson quantum noise
	 * @param seed        RNG seed (vary across noise realizations)
	 * @return one count image per energy bin
	 */
	public MultiChannelGrid2D project(MultiChannelGrid2D pathLengths, double[] binEdges,
			boolean noise, int seed) {
		int nBins = binEdges.length - 1;
		if (nBins < 1 || nBins > MAX_BINS)
			throw new IllegalArgumentException("nBins must be in 1.." + MAX_BINS + ", got " + nBins);
		init();
		int w = pathLengths.getWidth(), h = pathLengths.getHeight(), npix = w * h;

		// map each sampled energy to its bin (or -1 if outside all bins)
		int[] binOfEnergy = new int[nE];
		for (int e = 0; e < nE; e++) {
			binOfEnergy[e] = -1;
			for (int b = 0; b < nBins; b++) {
				if (energies[e] >= binEdges[b] && energies[e] < binEdges[b + 1]) {
					binOfEnergy[e] = b;
					break;
				}
			}
		}

		CLBuffer<FloatBuffer> pathBuf = packPathLengths(pathLengths, npix);
		CLBuffer<IntBuffer> binBuf = context.createIntBuffer(nE, Mem.READ_ONLY);
		binBuf.getBuffer().put(binOfEnergy).rewind();
		CLBuffer<FloatBuffer> outBuf = context.createFloatBuffer(nBins * npix, Mem.WRITE_ONLY);

		CLCommandQueue queue = device.createCommandQueue();
		CLKernel kernel = program.createCLKernel("photonCountingDetector");
		kernel.putArg(pathBuf).putArg(muBuf).putArg(fluxBuf).putArg(binBuf)
				.putArg(nMat).putArg(nE).putArg(npix).putArg(nBins)
				.putArg(noise ? 1 : 0).putArg(seed).putArg(outBuf);
		int local = Math.min(device.getMaxWorkGroupSize(), 256);
		int global = OpenCLUtil.roundUp(local, npix);
		queue.putWriteBuffer(pathBuf, false).putWriteBuffer(binBuf, true)
				.put1DRangeKernel(kernel, 0, global, local)
				.putReadBuffer(outBuf, true);

		MultiChannelGrid2D out = new MultiChannelGrid2D(w, h, nBins);
		FloatBuffer fb = outBuf.getBuffer();
		fb.rewind();
		float[] tmp = new float[npix];
		for (int b = 0; b < nBins; b++) {
			fb.position(b * npix);
			fb.get(tmp);
			out.setChannel(b, new Grid2D(tmp.clone(), w, h));
		}
		kernel.release();
		queue.release();
		pathBuf.release();
		binBuf.release();
		outBuf.release();
		return out;
	}
}
