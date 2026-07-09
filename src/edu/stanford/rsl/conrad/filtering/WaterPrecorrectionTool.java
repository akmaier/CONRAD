package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Single-material (water) beam-hardening precorrection in the projection domain.
 * <p>
 * CONRAD ships the Joseph &amp; Spital <i>dual</i>-material correction
 * ({@link ApplyLambdaWeightingTool}, {@link edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients#getBeamHardeningLookupTable})
 * but that tool's own Javadoc lists "a water-corrected reconstruction is required"
 * as a <b>prerequisite</b> it assumes has already been performed elsewhere. This
 * tool performs exactly that missing step-0: it linearizes (de-cups) the
 * polychromatic water line integral so a uniform water object reconstructs flat.
 * <p>
 * <b>Algorithm.</b> For a detector with an arbitrary per-energy effective
 * weighting {@code w_eff(E)} we tabulate the polychromatic water line integral
 * <pre>
 *   p_poly(L) = -log( sum_E w_eff(E) * exp(-mu_water(E) * L) )   (w_eff normalized)
 * </pre>
 * over a range of water thicknesses {@code L in [0, Lmax]} (cm), and the target
 * monochromatic (linear) response
 * <pre>
 *   p_mono(L) = mu_ref * L,   mu_ref = sum_E w_eff(E)*mu_water(E) / sum_E w_eff(E)
 * </pre>
 * We then fit a polynomial {@code c} such that {@code p_mono ~= polyval(c, p_poly)}.
 * Applying {@code polyval(c, .)} to a measured water line integral removes the
 * convex cupping, mapping it back onto the linear monochromatic response at the
 * spectrum-weighted mean attenuation {@code mu_ref}.
 * <p>
 * <b>Multi-energy / photon-counting (PCD) support.</b> The whole method is driven
 * by the arbitrary per-energy weighting {@code w_eff(E)}:
 * <ul>
 * <li>An energy-integrating detector (EID) uses {@code w_eff(E) = flux(E) * E}.</li>
 * <li>A photon-counting bin uses that bin's own detection weighting
 *     {@code w_eff(E) = flux(E) * binResponse(E)} (e.g. a rectangular window
 *     between two energy thresholds, or a full detector-response function).</li>
 * </ul>
 * Because the same {@link #calibrate(double[], double[], Material, double, int)}
 * entry point takes {@code w_eff} directly, the identical class calibrates a
 * per-detector or a per-bin polynomial. For a multi-bin PCD, instantiate one tool
 * per bin (each holding that bin's polynomial), or keep an array of tools; every
 * bin's sinogram is then precorrected with its own {@link #apply(Grid2D)}.
 * <p>
 * This is the CONRAD-API counterpart of {@code water_precorrection_poly} in the
 * SPIONvsXRay project's {@code conrad_ct.py}.
 *
 * @author akmaier
 * @see ApplyLambdaWeightingTool
 * @see edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients
 */
public class WaterPrecorrectionTool extends IndividualImageFilteringTool {

	private static final long serialVersionUID = 7716351209471455681L;

	/** Fitted precorrection polynomial coefficients, highest degree first (numpy.polyfit convention). */
	private double[] coefficients = null;

	/** Spectrum-weighted mean water attenuation used as the monochromatic slope [1/cm]. */
	private double muRef = 0.0;

	public WaterPrecorrectionTool() {
	}

	/**
	 * Convenience constructor that installs a set of pre-computed coefficients.
	 * @param coefficients precorrection polynomial (highest degree first)
	 */
	public WaterPrecorrectionTool(double[] coefficients) {
		this.coefficients = coefficients.clone();
		this.configured = true;
	}

	// ------------------------------------------------------------------
	// Calibration
	// ------------------------------------------------------------------

	/**
	 * Calibrate the precorrection polynomial for an arbitrary per-energy detector
	 * weighting. This is the general multi-energy / per-PCD-bin entry point.
	 *
	 * @param energies        sampled photon energies [keV]
	 * @param detectorWeights per-energy effective weighting w_eff(E) (need not be
	 *                        normalized). EID: flux*E; PCD bin: flux*binResponse.
	 * @param water           the (water) material whose attenuation is linearized
	 * @param Lmax            maximum water path length to tabulate [cm]
	 * @param degree          polynomial degree of the fit
	 */
	public void calibrate(double[] energies, double[] detectorWeights, Material water, double Lmax, int degree) {
		if (energies.length != detectorWeights.length) {
			throw new IllegalArgumentException("energies and detectorWeights must have equal length");
		}
		final int nE = energies.length;

		// normalize the per-energy weighting w_eff
		double[] w = new double[nE];
		double wSum = 0.0;
		for (int i = 0; i < nE; i++) {
			w[i] = Math.max(detectorWeights[i], 0.0);
			wSum += w[i];
		}
		if (wSum <= 0.0) throw new IllegalArgumentException("detectorWeights sum to zero");
		for (int i = 0; i < nE; i++) w[i] /= wSum;

		// water linear attenuation [1/cm] per energy (TOTAL with coherent scattering)
		double[] mu = new double[nE];
		muRef = 0.0;
		for (int i = 0; i < nE; i++) {
			mu[i] = water.getAttenuation(energies[i], AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION);
			muRef += w[i] * mu[i];  // spectrum-weighted mean attenuation = monochromatic slope
		}

		// tabulate p_poly(L) and target p_mono(L) = muRef * L
		final int nL = 400;
		double[] pPoly = new double[nL];
		double[] pMono = new double[nL];
		for (int k = 0; k < nL; k++) {
			double L = Lmax * k / (nL - 1);
			double transmission = 0.0;
			for (int i = 0; i < nE; i++) {
				transmission += w[i] * Math.exp(-mu[i] * L);
			}
			pPoly[k] = -Math.log(transmission);
			pMono[k] = muRef * L;
		}

		this.coefficients = polyfit(pPoly, pMono, degree);
		this.configured = true;
	}

	/**
	 * Convenience calibration from a CONRAD spectrum for an energy-integrating
	 * detector: builds w_eff(E) = flux(E) * E and calibrates against water from
	 * the materials database.
	 *
	 * @param spectrum the polychromatic X-ray spectrum
	 * @param Lmax     maximum water path length [cm]
	 * @param degree   polynomial degree
	 */
	public void calibrateFromSpectrum(PolychromaticXRaySpectrum spectrum, double Lmax, int degree) {
		calibrateFromSpectrum(spectrum, null, MaterialsDB.getMaterial("water"), Lmax, degree);
	}

	/**
	 * Convenience calibration from a CONRAD spectrum, with an optional per-energy
	 * detector-response window. This is the PCD-friendly convenience method.
	 *
	 * @param spectrum     the polychromatic X-ray spectrum
	 * @param binResponse  per-energy detector response (same length as the
	 *                     spectrum energies); if {@code null}, an energy-integrating
	 *                     detector (E-weighting) is assumed. For a PCD bin, pass a
	 *                     window (e.g. 1 inside [E_low, E_high), 0 outside).
	 * @param water        water material
	 * @param Lmax         maximum water path length [cm]
	 * @param degree       polynomial degree
	 */
	public void calibrateFromSpectrum(PolychromaticXRaySpectrum spectrum, double[] binResponse,
			Material water, double Lmax, int degree) {
		double[] energies = spectrum.getPhotonEnergies();
		double[] flux = spectrum.getPhotonFlux();
		double[] wEff = new double[energies.length];
		for (int i = 0; i < energies.length; i++) {
			// EID: energy-integrating weighting flux*E; PCD bin: flux*binResponse.
			double response = (binResponse == null) ? energies[i] : binResponse[i];
			wEff[i] = flux[i] * response;
		}
		calibrate(energies, wEff, water, Lmax, degree);
	}

	// ------------------------------------------------------------------
	// Application
	// ------------------------------------------------------------------

	/**
	 * Apply the calibrated precorrection to a single water line-integral value.
	 * @param p measured (polychromatic) water line integral
	 * @return linearized (monochromatic) line integral
	 */
	public double applyToValue(double p) {
		if (coefficients == null) throw new IllegalStateException("WaterPrecorrectionTool not calibrated");
		double r = 0.0;
		for (int i = 0; i < coefficients.length; i++) {
			r = r * p + coefficients[i];  // Horner, highest degree first
		}
		return r;
	}

	/**
	 * Apply the calibrated precorrection to a whole sinogram / projection in place.
	 * @param sinogram the projection data (modified in place and returned)
	 * @return the same grid, precorrected
	 */
	public Grid2D apply(Grid2D sinogram) {
		int w = sinogram.getWidth();
		int h = sinogram.getHeight();
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				sinogram.putPixelValue(i, j, applyToValue(sinogram.getPixelValue(i, j)));
			}
		}
		return sinogram;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		return apply(imageProcessor);
	}

	/** @return the fitted precorrection polynomial (highest degree first), or {@code null}. */
	public double[] getCoefficients() {
		return coefficients == null ? null : coefficients.clone();
	}

	/** @return the spectrum-weighted mean water attenuation [1/cm] used as the monochromatic slope. */
	public double getMuRef() {
		return muRef;
	}

	// ------------------------------------------------------------------
	// Least-squares polynomial fit (numpy.polyfit convention: highest degree first)
	// ------------------------------------------------------------------

	private static double[] polyfit(double[] x, double[] y, int degree) {
		int n = x.length;
		int m = degree + 1;
		// Vandermonde matrix with columns [x^degree, ..., x^1, x^0] to match
		// the highest-degree-first coefficient ordering used on application.
		SimpleMatrix A = new SimpleMatrix(n, m);
		for (int r = 0; r < n; r++) {
			double pow = 1.0;
			for (int c = m - 1; c >= 0; c--) {
				A.setElementValue(r, c, pow);
				pow *= x[r];
			}
		}
		SimpleVector b = new SimpleVector(n);
		for (int r = 0; r < n; r++) b.setElementValue(r, y[r]);
		SimpleVector sol = Solvers.solveLinearLeastSquares(A, b);
		double[] coeffs = new double[m];
		for (int c = 0; c < m; c++) coeffs[c] = sol.getElement(c);
		return coeffs;
	}

	// ------------------------------------------------------------------
	// ImageFilteringTool boilerplate
	// ------------------------------------------------------------------

	@Override
	public IndividualImageFilteringTool clone() {
		WaterPrecorrectionTool clone = new WaterPrecorrectionTool();
		clone.coefficients = (coefficients == null) ? null : coefficients.clone();
		clone.muRef = muRef;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Water Beam Hardening Precorrection (Single Material)";
	}

	@Override
	public void configure() throws Exception {
		// Default: calibrate from a standard CONRAD tungsten spectrum for an
		// energy-integrating detector. Callers that need a specific spectrum or a
		// PCD bin weighting should use calibrate(...) / calibrateFromSpectrum(...).
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum();
		calibrateFromSpectrum(spectrum, 30.0, 4);
		setConfigured(true);
	}

	@Override
	public boolean isDeviceDependent() {
		// Models the physical (spectrum + detector) beam-hardening response.
		return true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@article{Herman79-CDA,\n" +
			"  author = {{Herman}, G. T.},\n" +
			"  title = {{Correction for beam hardening in computed tomography}},\n" +
			"  journal = {{Physics in Medicine and Biology}},\n" +
			"  volume = {24},\n" +
			"  number = {1},\n" +
			"  pages = {81-106},\n" +
			"  year = {1979}\n" +
			"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Herman GT. Correction for beam hardening in computed tomography. Phys Med Biol 1979;24(1):81-106.";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
