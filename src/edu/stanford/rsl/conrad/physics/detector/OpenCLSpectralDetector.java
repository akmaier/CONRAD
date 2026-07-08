/*
 * Copyright (C) 2026 - Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;

/**
 * Base class for GPU polychromatic detectors that integrate the Beer-Lambert law
 * over energy on the device. It holds the sampled spectrum and the per-material
 * linear attenuation tables (from CONRAD's material database, identical to
 * {@link edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel})
 * and uploads them to the OpenCL device once. Subclasses add the energy-integrating
 * ({@link OpenCLEnergyIntegratingDetector}) and photon-counting
 * ({@link OpenCLPhotonCountingDetector}) read-out; these are deliberately separate
 * classes -- they are different physical detector concepts.
 *
 * Input is a {@link MultiChannelGrid2D} of per-material path lengths [mm], one
 * channel per material in the same order as the {@code materials} array (e.g. the
 * output of {@link edu.stanford.rsl.conrad.opencl.OpenCLMaterialPathLengthPhantomRenderer}).
 */
public abstract class OpenCLSpectralDetector {

	protected final PolychromaticXRaySpectrum spectrum;
	protected final Material[] materials;
	protected final double[] energies;
	protected final int nE;
	protected final int nMat;
	protected final float[] fluxF;       // [nE] incident photons per energy
	protected final float[] energiesF;   // [nE] keV
	protected final float[] muFlat;      // [nMat*nE] linear att /10 (mm->cm)

	protected CLContext context;
	protected CLDevice device;
	protected CLProgram program;
	protected CLBuffer<FloatBuffer> muBuf, fluxBuf, energyBuf;

	protected OpenCLSpectralDetector(PolychromaticXRaySpectrum spectrum, Material[] materials) {
		this.spectrum = spectrum;
		this.materials = materials;
		this.energies = spectrum.getPhotonEnergies();
		this.nE = energies.length;
		this.nMat = materials.length;
		this.fluxF = new float[nE];
		this.energiesF = new float[nE];
		for (int e = 0; e < nE; e++) {
			fluxF[e] = (float) spectrum.getPhotonFlux(energies[e]);
			energiesF[e] = (float) energies[e];
		}
		this.muFlat = new float[nMat * nE];
		AttenuationType att = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;
		for (int m = 0; m < nMat; m++)
			for (int e = 0; e < nE; e++)
				// linear attenuation [1/cm]; /10 so tau = sum mu * pathlen_mm is dimensionless
				muFlat[m * nE + e] = (float) (materials[m].getAttenuation(energies[e], att) / 10.0);
	}

	/** Lazily create the OpenCL context/program and upload the spectral tables. */
	protected synchronized void init() {
		if (context != null)
			return;
		context = OpenCLUtil.createContext();
		device = context.getMaxFlopsDevice();
		try {
			program = context.createProgram(
					OpenCLSpectralDetector.class.getResourceAsStream("SpectralDetector.cl")).build();
		} catch (IOException e) {
			throw new RuntimeException("Could not load SpectralDetector.cl", e);
		}
		muBuf = context.createFloatBuffer(nMat * nE, Mem.READ_ONLY);
		muBuf.getBuffer().put(muFlat).rewind();
		fluxBuf = context.createFloatBuffer(nE, Mem.READ_ONLY);
		fluxBuf.getBuffer().put(fluxF).rewind();
		energyBuf = context.createFloatBuffer(nE, Mem.READ_ONLY);
		energyBuf.getBuffer().put(energiesF).rewind();
		CLCommandQueue queue = device.createCommandQueue();
		queue.putWriteBuffer(muBuf, false).putWriteBuffer(fluxBuf, false).putWriteBuffer(energyBuf, true);
		queue.release();
	}

	/** Pack a per-material path-length stack into a device buffer [nMat*npix]. */
	protected CLBuffer<FloatBuffer> packPathLengths(MultiChannelGrid2D pathLengths, int npix) {
		if (pathLengths.getNumberOfChannels() != nMat)
			throw new IllegalArgumentException("path-length stack has " + pathLengths.getNumberOfChannels()
					+ " channels, expected " + nMat + " (one per material, same order)");
		CLBuffer<FloatBuffer> buf = context.createFloatBuffer(nMat * npix, Mem.READ_ONLY);
		for (int m = 0; m < nMat; m++) {
			float[] ch = pathLengths.getChannel(m).getBuffer();
			for (int i = 0; i < npix; i++)
				buf.getBuffer().put(ch[i]);
		}
		buf.getBuffer().rewind();
		return buf;
	}

	protected Grid2D toGrid2D(FloatBuffer fb, int w, int h) {
		float[] out = new float[w * h];
		fb.rewind();
		fb.get(out);
		return new Grid2D(out, w, h);
	}

	public void release() {
		if (context != null) {
			context.release();
			context = null;
		}
	}

	public PolychromaticXRaySpectrum getSpectrum() {
		return spectrum;
	}
}
