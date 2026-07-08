/*
 * Copyright (C) 2026 - Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.nio.FloatBuffer;

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
 * GPU energy-integrating detector (EID). Integrates the transmitted spectrum
 * weighted by photon energy: intensity(p) = sum_E n_E(p) * E, with
 * n_E(p) = flux_E * exp(-sum_m mu[m,E] * pathlen[m,p]). When noise is enabled the
 * counts n_E are Poisson-sampled per energy BEFORE the energy weighting, so the
 * read-out has the correct energy-integrated variance sum_E E^2 * mean_E (which a
 * single Poisson/Gaussian draw on the summed signal would not reproduce).
 *
 * On-device analogue of
 * {@link edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel}
 * with {@code energyIntegrating = true}.
 */
public class OpenCLEnergyIntegratingDetector extends OpenCLSpectralDetector {

	public OpenCLEnergyIntegratingDetector(PolychromaticXRaySpectrum spectrum, Material[] materials) {
		super(spectrum, materials);
	}

	/** Noise-free integrated intensity image. */
	public Grid2D project(MultiChannelGrid2D pathLengths) {
		return project(pathLengths, false, 0);
	}

	/**
	 * @param pathLengths per-material path lengths [mm] (channel order == materials)
	 * @param noise       apply per-energy Poisson quantum noise
	 * @param seed        RNG seed (vary across noise realizations)
	 * @return integrated intensity image (same size as the input channels)
	 */
	public Grid2D project(MultiChannelGrid2D pathLengths, boolean noise, int seed) {
		init();
		int w = pathLengths.getWidth(), h = pathLengths.getHeight(), npix = w * h;
		CLBuffer<FloatBuffer> pathBuf = packPathLengths(pathLengths, npix);
		CLBuffer<FloatBuffer> outBuf = context.createFloatBuffer(npix, Mem.WRITE_ONLY);
		CLCommandQueue queue = device.createCommandQueue();
		CLKernel kernel = program.createCLKernel("energyIntegratingDetector");
		kernel.putArg(pathBuf).putArg(muBuf).putArg(fluxBuf).putArg(energyBuf)
				.putArg(nMat).putArg(nE).putArg(npix)
				.putArg(noise ? 1 : 0).putArg(seed).putArg(outBuf);
		int local = Math.min(device.getMaxWorkGroupSize(), 256);
		int global = OpenCLUtil.roundUp(local, npix);
		queue.putWriteBuffer(pathBuf, true)
				.put1DRangeKernel(kernel, 0, global, local)
				.putReadBuffer(outBuf, true);
		Grid2D out = toGrid2D(outBuf.getBuffer(), w, h);
		kernel.release();
		queue.release();
		pathBuf.release();
		outBuf.release();
		return out;
	}
}
