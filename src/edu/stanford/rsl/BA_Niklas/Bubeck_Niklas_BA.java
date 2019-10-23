package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.BA_Niklas.Regression;
import edu.stanford.rsl.BA_Niklas.PolynomialRegression;
import edu.stanford.rsl.BA_Niklas.ListInFile;
import com.zetcode.linechartex.ScatterPlot;
import com.zetcode.linechartex.LineChartEx;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.BA_Niklas.PhaseContrastImages;
import edu.stanford.rsl.BA_Niklas.ProjectorAndBackprojector;
import edu.stanford.rsl.apps.Conrad;
import edu.stanford.rsl.conrad.data.numeric.*;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.builder.ReflectionToStringBuilder;
import org.math.plot.utils.Array;

import org.apache.commons.math3.distribution.NormalDistribution;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.io.*;

/**
 * Bachelor thesis of Niklas Bubeck
 * 
 * This class simulates data, forward-projects it and does a
 * filtered-backprojection.
 * 
 * @author Lina Felsner and Niklas Bubeck
 *
 */
public class Bubeck_Niklas_BA {

	static int size = 128;
	static int xstart;
	static int xend;
	static int xstart2;
	static int xend2;
	static int ystart;
	static int yend;
	static int ystart2;
	static int yend2;
	static boolean only_dark;
	static int value;
	static int nr_ellipses;
	static int iter_num;
	static float error_val;
	static String noisetype;
	static Boolean simchecked;
	static Boolean iterchecked;
	static Boolean trcchecked;
	static Boolean noisechecked;
	static Boolean vischecked;
	static Boolean singleMaterialCheck;
	static Boolean multipleMaterialCheck;
	static Boolean saveImagesCheck;
	static String path;
	static SimpleMatrix[] R; // rotation matrices.
	static double[][] t; // displacement vectors.
	static double[][] ab; // the length of the principal axes.
	static double[] rho; // signal intensity.
	static double[] df; // dark-field signal intensity.
	static int detector_width = 200;
	static int nr_of_projections = 360;
	static List<Float> errorlist = new ArrayList<Float>();
	static PhaseContrastImages pci;
	static PhaseContrastImages pci_sino;
	static PhaseContrastImages pci_sino_truncated;
	static NumericGrid trunc_reko;
	static NumericGrid dark_reco;
	static NumericGrid splitted_dark;
	static NumericGrid instreko;
	static String pathtofile = "/CONRAD/src/edu/stanford/rsl/BA_Niklas/";
	static String pathtoproject;

	/**
	 * creates random ellipses
	 */
	public static void create_ellipse() {
		R = new SimpleMatrix[nr_ellipses];
		t = new double[nr_ellipses][2];
		ab = new double[nr_ellipses][2];
		rho = new double[nr_ellipses];
		df = new double[nr_ellipses];

		t[0][0] = 0.42;
		t[0][1] = 0.21;
		t[1][0] = -0.49;
		t[1][1] = 0.05;
//		t[2][0] = 0.42;
//		t[2][1] = -0.75;
//		t[3][0] = -0.38;
//		t[3][1] = -0.40;
//		t[4][0] = 0.38;
//		t[4][1] = -0.10;

		ab[0][0] = 0.45;
		ab[0][1] = 0.16;
		ab[1][0] = 0.55;
		ab[1][1] = 0.60;
//		ab[2][0] = 0.21;
//		ab[2][1] = 0.16;
//		ab[3][0] = 0.30;
//		ab[3][1] = 0.35;
//		ab[4][0] = 0.25;
//		ab[4][1] = 0.40;

		double alpha0 = 33.16;
		R[0] = new SimpleMatrix(
				new double[][] { { Math.cos(alpha0), -Math.sin(alpha0) }, { Math.sin(alpha0), Math.cos(alpha0) } });

		double alpha1 = 6.6;
		R[1] = new SimpleMatrix(
				new double[][] { { Math.cos(alpha1), -Math.sin(alpha1) }, { Math.sin(alpha1), Math.cos(alpha1) } });

//		double alpha2 = 15.16;
//		R[2] = new SimpleMatrix(
//				new double[][] { { Math.cos(alpha2), -Math.sin(alpha2) }, { Math.sin(alpha2), Math.cos(alpha2) } });
//		
//		double alpha3 = 172;
//		R[3] = new SimpleMatrix(
//				new double[][] { { Math.cos(alpha3), -Math.sin(alpha3) }, { Math.sin(alpha3), Math.cos(alpha3) } });
//		
//		double alpha4 = 172;
//		R[4] = new SimpleMatrix(
//				new double[][] { { Math.cos(alpha4), -Math.sin(alpha4) }, { Math.sin(alpha4), Math.cos(alpha4) } });
//		
		rho[0] = 0.92;
		rho[1] = 0.30;
//		rho[2] = 0.20; 
//		rho[3] = 0.75;
//		rho[4] = 0.6;

		df[0] = 0.61;
		df[1] = 0.98;
//		df[2] = 0.5;
//		df[3] = 0.25;
//		df[4] = 0.1;
//		for (int i = 0; i < nr_ellipses; i++) {
//
//			// translation between -0.5 and 0.5
//			t[i][0] = Math.random() - 0.5; // delta_x
//			t[i][1] = Math.random() - 0.5; // delta_y
//			
//			System.out.println("translation x: " + t[i][0]);
//			System.out.println("translation y: " + t[i][1]);
//			// size between 0 and 0.75
//			ab[i][0] = Math.random() * 0.75; // a
//			ab[i][1] = Math.random() * 0.75; // b
//			
//			System.out.println("size a: " + ab[i][0]);
//			System.out.println("size b: " + ab[i][1]);
//			
//			// rotation
//			double alpha = Math.random() * 90;
//			R[i] = new SimpleMatrix(
//					new double[][] { { Math.cos(alpha), -Math.sin(alpha) }, { Math.sin(alpha), Math.cos(alpha) } });
//			
//			System.out.println("rotation " + alpha);
//			
//			// values
//			rho[i] = Math.random();
//			df[i] = Math.random();
//			
//			System.out.println("signal intensity : " + rho[i]);
//			System.out.println("dark-field signal intensity: " + df[i]);
//
//		}
	}

	/**
	 * computes the absorption and dark-field signal at position {x,y}
	 * 
	 * @param x - position x-direction
	 * @param y - position y-direction
	 * @return array od absorption and dark-field value
	 */
	public static double[] getSignal(double x, double y) {

		double[] r = { x, y };
		double[] p = new double[2];
		double sum = 0.0;

		double signal_absorption = 0.0;
		double signal_dark_field = 0.0;

		for (int i = 0; i < nr_ellipses; i++) { // loop through each of the ellipsoids
			SimpleVector pos = new SimpleVector(new double[] { r[0] - t[i][0], r[1] - t[i][1] });
			p = SimpleOperators.multiply(R[i], pos).copyAsDoubleArray();
			sum = Math.pow(p[0] / ab[i][0], 2) + Math.pow(p[1] / ab[i][1], 2);
			signal_absorption += (sum <= 1.0) ? rho[i] : 0;
			signal_dark_field += (sum <= 1.0) ? df[i] : 0;
		}

		return new double[] { signal_absorption, signal_dark_field };
	}

	/**
	 * simulates ellipses
	 * 
	 * @return Phase contrast images containing absorption (phase) and dark-field
	 *         phantoms
	 */
	public static PhaseContrastImages simulate_data() {

		Grid2D amplitude = new Grid2D(size, size);
		Grid2D phase = new Grid2D(size, size);
		Grid2D dark_field = new Grid2D(size, size);

		create_ellipse();

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double[] value = getSignal(((float) ((size / 2) - i)) / (size / 2),
						((float) ((size / 2) - j)) / (size / 2));
				amplitude.setAtIndex(i, j, (float) value[0]);
				dark_field.setAtIndex(i, j, (float) value[1]);
			}
		}

		PhaseContrastImages pci = new PhaseContrastImages(amplitude, phase, dark_field);
		return pci;
	}

//	public static Grid2D blur(NumericGrid grid, double sigma, int kernelsize) {
//		Grid2D img = (Grid2D) grid;
//		double[] kernel = createKernel(sigma, kernelsize);
//		for (int i = 0; i < img.getWidth(); i++) {
//			for (int j = 0; j < img.getHeight(); j++) {
//				double overflow = 0;
//				int counter = 0;
//				int kernelhalf = (kernelsize - 1) / 2;
//				double value = 0;
//				for (int k = i - kernelhalf; k < i + kernelhalf; k++) {
//					for (int l = j - kernelhalf; l < j + kernelhalf; l++) {
//						if (k < 0 || k >= img.getWidth() || l < 0 || l >= img.getHeight()) {
//							counter++;
//							overflow += kernel[counter];
//							continue;
//						}
//
//						double val = img.getAtIndex(k, l);
//						value += val * kernel[counter];
//
//						counter++;
//					}
//					counter++;
//				}
//
//				if (overflow > 0) {
//					value = 0;
//
//					counter = 0;
//					for (int k = i - kernelhalf; k < i + kernelhalf; k++) {
//						for (int l = j - kernelhalf; l < j + kernelhalf; l++) {
//							if (k < 0 || k >= img.getWidth() || l < 0 || l >= img.getHeight()) {
//								counter++;
//								continue;
//							}
//
//							double val = img.getAtIndex(k, l);
//							value += val * kernel[counter] * (1 / (1 - overflow));
//
//							counter++;
//						}
//						counter++;
//					}
//				}
//
//				img.setAtIndex(i, j, (float) value);
//			}
//		}
//		return img;
//	}
//
//	public static double[] createKernel(double sigma, int kernelsize) {
//		double[] kernel = new double[kernelsize * kernelsize];
//		for (int i = 0; i < kernelsize; i++) {
//			double x = i - (kernelsize - 1) / 2;
//			for (int j = 0; j < kernelsize; j++) {
//				double y = j - (kernelsize - 1) / 2;
//				kernel[j + i * kernelsize] = 1 / (2 * Math.PI * sigma * sigma)
//						* Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
//			}
//		}
//		float sum = 0;
//		for (int i = 0; i < kernelsize; i++) {
//			for (int j = 0; j < kernelsize; j++) {
//				sum += kernel[j + i * kernelsize];
//			}
//		}
//		for (int i = 0; i < kernelsize; i++) {
//			for (int j = 0; j < kernelsize; j++) {
//				kernel[j + i * kernelsize] /= sum;
//			}
//		}
//		return kernel;
//	}

	public static NumericGrid add_noise(NumericGrid img, double mean, double variance, String distribution) {
		Grid2D image = (Grid2D) img;
		for (int i = 0; i < image.getWidth(); i++) {
			for (int j = 0; j < image.getHeight(); j++) {
				if (image.getAtIndex(i, j) == (float) 0.0) {
					continue;
				}

				double noise = 0;

				if (distribution == "gaussian") {
					java.util.Random r = new java.util.Random();
					noise = r.nextGaussian() * Math.sqrt(variance) + mean;
				}

				float val = image.getAtIndex(i, j);
				image.setAtIndex(i, j, val + (float) noise);

//				((Grid2D) pci.getAmp()).setAtIndex(i, j, val + (float) noise);
//				((Grid2D) pci.getDark()).setAtIndex(i, j, val + (float) noise);
			}
		}
		image.show("nooiiise");
		return image;
	}

	private static double calculate_mean_value(String[] mean_values) {
		int counter = 0;
		double sum = 0;
		double mean;
		for (int i = 0; i < mean_values.length; i++) {
			if (mean_values[i] == null) {
				continue;
			} else {
				counter++;
				sum += Double.parseDouble(mean_values[i]);
			}
		}
		mean = sum / counter;
		return mean;
	}

	public static NumericGrid[] get_segmentation(NumericGrid dark_gt, double noise)
			throws IOException, InterruptedException {
		Grid2D darkgt = (Grid2D) dark_gt;
		Float[] values = new Float[darkgt.getHeight() * darkgt.getWidth()];

		int counter = 0;
		for (int i = 0; i < darkgt.getWidth(); i++) {
			for (int j = 0; j < darkgt.getHeight(); j++) {
				float value = darkgt.getAtIndex(i, j);
				values[counter] = (float) ((double) Math.round(value * 100d) / 100d);
				counter++;
			}
		}

//		String[] means = new String[20];
//		String[][] mean_values = new String[20][darkgt.getHeight() * darkgt.getWidth()];
//		means[0] = "0.0";
//		int matcounter = 0;
//		int test = 0;
//		for (int z = 0; z < matcounter; z++) {
//
//			for (int y = 0; y < values.length; y++) {
//
//				System.out.println(
//						"wert: " + values[y] + " untere Grenz: " + (Double.parseDouble(means[matcounter]) - noise)
//								+ " obere Grenze: " + Double.parseDouble(means[matcounter]) + noise);
//				if ((values[y] > Double.parseDouble(means[matcounter]) - noise)
//						&& (values[y] < Double.parseDouble(means[matcounter]) + noise)) {
//					System.out.println("ich bin in der ersten schleife");
//					mean_values[matcounter][y] = values[y].toString();
//					double mean = calculate_mean_value(mean_values[matcounter]);
//					means[matcounter] = Double.toString(mean);
//				} else {
//					System.out.println("ich bin im Else fall");
//					matcounter++;
//					means[matcounter] = values[y].toString();
//
////				if(means[matcounter] == null) {
////				}
//					mean_values[matcounter][y] = values[y].toString();
//				}
//
//			}
//
//			if (means[matcounter + 1] == null) {
//				System.out.println("test");
//				break;
//			}
//
//		}

		if (noisechecked) {
			int test = 0;
			int ctr = 0;
			float[] histotemp = new float[darkgt.getHeight() * darkgt.getWidth()];
			float[] numbertemp = new float[darkgt.getHeight() * darkgt.getWidth()];
			for (int n = 0; n < values.length; n++) {
				if (values[n] == (float) 0.0) {
					test++;
					continue;
				}
				boolean indicator = false;
				for (int m = 0; m < values.length; m++) {
					if (values[n] == histotemp[m]) {
						numbertemp[m]++;
						indicator = true;
					}

				}
				if (!indicator) {
					histotemp[ctr] = values[n];
					ctr++;
				}

			}

			Float[] histo = new Float[ctr];
			Float[] number = new Float[ctr];
			for (int h = 0; h < ctr; h++) {
				histo[h] = histotemp[h];
				number[h] = numbertemp[h];
			}

			String pathhisto = pathtoproject + pathtofile + "Datasheets/values.csv";
			String pathcount = pathtoproject + pathtofile + "Datasheets/count.csv";
			String path_nr_ell = pathtoproject + pathtofile + "Datasheets/nr_ell.csv";
			Float[] nr_ell = { (float) nr_ellipses };
			ListInFile.export(Arrays.asList(histo), pathhisto, "histo-values");
			ListInFile.export(Arrays.asList(number), pathcount, "count");
			ListInFile.export(Arrays.asList(nr_ell), path_nr_ell, "Number of ellipses");

			String command = "py " + pathtoproject + pathtofile + "PythonScripts/kmean.py";
			try {
				Process process = Runtime.getRuntime().exec(command);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		// only use unique values
		Float[] unique = Arrays.stream(values).distinct().toArray(Float[]::new);

		// sort with increasing number
//		System.out.println(test);
//		for (int i = 0; i < means.length; i++) {
//			System.out.println(means[i]);
//		}

		Arrays.sort(unique);

		// calc thresh values
		Float[] thresh = new Float[unique.length - 1];
		for (int k = 0; k < unique.length - 1; k++) {
			thresh[k] = ((unique[k]) + (unique[k + 1])) / 2;
		}

		if (noisechecked) {
			TimeUnit.SECONDS.sleep(3);
			String paththresh = pathtoproject + pathtofile + "Datasheets/thresholds.csv";

			BufferedReader threshReader = new BufferedReader(new FileReader(paththresh));

			String[] threshs = new String[20];

			int threshcounter = 0;
			String threshline;
			while ((threshline = threshReader.readLine()) != null) {
				threshs[threshcounter] = threshline;
				threshcounter++;
				// do something with the data

			}
			int thctr = 0;
			Float[] thresholds = new Float[nr_ellipses];
			for (int o = 0; o < threshs.length; o++) {
				if (threshs[o] != null && threshs[o].length() > 0) {
					thresholds[thctr] = Float.parseFloat(threshs[o]);
					thctr++;
				}
			}

			System.out.println("Hier sind die threshs: " + threshs[4]);
			thresh = thresholds;
		}

		System.out.println((thresh.length));
		System.out.println(Arrays.toString(values));
		System.out.println(Arrays.toString(unique));
		System.out.println(Arrays.toString(thresh));

		NumericGrid[] materials = new NumericGrid[thresh.length + 1];
		for (int l = 0; l <= thresh.length; l++) {
			Grid2D mat = (Grid2D) darkgt.clone();
//			if(thresh[l] == null) {
//				continue;
//			}
			for (int i = 0; i < darkgt.getWidth(); i++) {
				for (int j = 0; j < darkgt.getHeight(); j++) {

					// picture with the background color
					if (l == 0) {
						if (darkgt.getAtIndex(i, j) > thresh[0]) {
							mat.setAtIndex(i, j, 0);

						}

					}

					else if (l == thresh.length) {
						if (darkgt.getAtIndex(i, j) < thresh[l - 1]) {
							mat.setAtIndex(i, j, 0);

						}

					} else {

						if (darkgt.getAtIndex(i, j) < thresh[l - 1] || darkgt.getAtIndex(i, j) > thresh[l]) {
							mat.setAtIndex(i, j, 0);
//							System.out.println("bin drin");
						}

					}

				}
			}
			ImagePlus imp8 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) mat).createImage());
			String path8 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/IchbinMat" + l;
			IJ.saveAs(imp8, "png", path8);
			materials[l] = mat;

		}

		if (multipleMaterialCheck) {
			NumericGrid[] wo_bg = new NumericGrid[materials.length - 1];
			int cter = 0;
			for (int i = 1; i < materials.length; i++) {
				wo_bg[cter] = materials[i];
				cter++;
			}
			materials = wo_bg;
		}

		return materials;
	}

	public static void get_line_plot_data(int row) {
		Float[] gt = new Float[128];
		Float[] reco = new Float[128];
		Float[] trunc = new Float[128];

		for (int i = 0; i < size; i++) {
			int[] idx = { i , row };
			gt[i] = pci.getDark().getValue(idx);
			reco[i] = dark_reco.getValue(idx);
			trunc[i] = trunc_reko.getValue(idx);
		}

		String pathgt = pathtoproject + pathtofile + "Datasheets/line_gt.csv";
		String pathreco = pathtoproject + pathtofile + "Datasheets/line_reco.csv";
		String pathtrunc = pathtoproject + pathtofile + "Datasheets/line_trunc.csv";
		ListInFile.export(Arrays.asList(gt), pathgt, "gt_line_values");
		ListInFile.export(Arrays.asList(reco), pathreco, "reco_line_values");
		ListInFile.export(Arrays.asList(trunc), pathtrunc, "trunc_line_values");
	}

	/**
	 * simulates truncation
	 * 
	 * @param pci_sino - sinograms
	 * @param min      - minimum on coordinate-axes
	 * @param max      - maximum on coordinate-axes
	 * @param axes     - determines the used axe
	 * @param value    - determines the value that will be set in range of min - max
	 *                 on Axes - axe
	 * @param random   - generates a random value that will be set. @param value
	 *                 will be overwritten
	 * @param filter   - filters the image with given filter (gauss, sobel, ....)
	 * @return Phase contrast images containing absorption (phase) and dark-field
	 *         phantoms with artifacts
	 */
//	TODO: filters
	public static PhaseContrastImages fake_truncation(PhaseContrastImages pciSino, int min, int max, int min2, int max2,
			String axes, int value, boolean random, boolean only_dark, String filter) {
		int width = pciSino.getWidth();
		int height = pciSino.getHeight();

		Grid2D amp = (Grid2D) pciSino.getAmp();
		Grid2D dark = (Grid2D) pciSino.getDark();
		Grid2D phase = (Grid2D) pciSino.getPhase();

		Grid2D fake_amp = new Grid2D(200, 360);
		Grid2D fake_dark = new Grid2D(200, 360);
		Grid2D fake_phase = new Grid2D(200, 360);

		for (int k = 0; k < fake_amp.getWidth(); k++) {
			for (int l = 0; l < fake_amp.getHeight(); l++) {
				float val_amp = amp.getAtIndex(k, l);
				float val_dark = dark.getAtIndex(k, l);
				float val_phase = phase.getAtIndex(k, l);

				fake_amp.setAtIndex(k, l, val_amp);
				fake_dark.setAtIndex(k, l, val_dark);
				fake_phase.setAtIndex(k, l, val_phase);

			}
		}

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (axes == "x") {
					if ((i > min && i < max) || (i > min2 && i < max2)) {
						if (random == true) {
							value = (int) (Math.random() * 255);
						}

						if (!only_dark) {
							fake_amp.setAtIndex(i, j, value);
							fake_phase.setAtIndex(i, j, value);
						}
						fake_dark.setAtIndex(i, j, value);
					}
				} else if (axes == "y") {
					if ((j > min && j < max) || (j > min2 && j < max2)) {
						if (random == true) {
							value = (int) (Math.random() * 255);
						}

						if (!only_dark) {
							fake_amp.setAtIndex(i, j, value);
							fake_phase.setAtIndex(i, j, value);
						}
						fake_dark.setAtIndex(i, j, value);
					}
				}

			}
		}

		NumericGridOperator test = new NumericGridOperator();

		PhaseContrastImages pci_sino_fake = new PhaseContrastImages(fake_amp, fake_phase, fake_dark);
		return pci_sino_fake;
	}

	public static NumericGrid fake_truncation_image(Grid2D image, int min, int max, String axes, int value) {

		Grid2D trunc = new Grid2D(image.getWidth(), image.getHeight());
//		int width = image.getWidth();
//		int height = image.getHeight();

		for (int k = 0; k < image.getWidth(); k++) {
			for (int l = 0; l < image.getHeight(); l++) {
				float val = image.getAtIndex(k, l);

				trunc.setAtIndex(k, l, val);

			}
		}

		for (int i = 0; i < 200; i++) {
			for (int j = 0; j < 360; j++) {
				if (axes == "x") {
					if ((i >= min && i <= max)) {

						trunc.setAtIndex(i, j, value);
					}
				} else if (axes == "y") {
					if ((j > min && j < max)) {
						trunc.setAtIndex(i, j, value);
					}
				}

			}
		}

		return trunc;
	}

	public static PhaseContrastImages region_of_interest(PhaseContrastImages sino_original,
			PhaseContrastImages sino_roi, int iter_num) {
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(360, 2 * Math.PI);
		Grid2D roi_amp = new Grid2D(size, size);
		Grid2D roi_phase = new Grid2D(size, size);
		Grid2D roi_dark = new Grid2D(size, size);
		PhaseContrastImages pci_recon = new PhaseContrastImages(roi_amp, roi_phase, roi_dark);

		NumericPointwiseOperators.subtractBy(sino_original.getAmp(), sino_roi.getAmp());
		NumericPointwiseOperators.subtractBy(sino_original.getPhase(), sino_roi.getPhase());
		NumericPointwiseOperators.subtractBy(sino_original.getDark(), sino_roi.getDark());
//		sino_original.show("sino_original");

		PhaseContrastImages pci_reko = p.filtered_backprojection(sino_original, size);
		return pci_reko;
	}

	public static NumericGrid split_dark(NumericGrid trc_dark_sino, NumericGrid mat, NumericGrid[] materials) {
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(nr_of_projections, 2 * Math.PI);
		Grid2D mat_sino = p.project((Grid2D) mat, new Grid2D(200, 360));

		Grid2D splitted_calc_dark = new Grid2D(200, 360);
		NumericPointwiseOperators.copy(splitted_dark, trc_dark_sino);
		NumericPointwiseOperators.copy(splitted_calc_dark, trc_dark_sino);

		for (int i = 0; i < materials.length; i++) {
			if (mat == materials[i]) {

				continue;
			}

			Grid2D material_sino = p.project((Grid2D) materials[i], new Grid2D(200, 360));

			for (int k = 0; k < 200; k++) {
				for (int l = 0; l < 360; l++) {
					int[] idx = { k, l };
					if (material_sino.getAtIndex(k, l) == (float) 0.0) {
						continue;
					} else {
						((Grid2D) splitted_dark).setAtIndex(k, l, (float) 1000);
						splitted_calc_dark.setAtIndex(k, l, (float) 0.0);
					}
				}
			}

		}
		return splitted_calc_dark;
	}

	public static double[][] get_comp_points(NumericGrid abso, NumericGrid dark, NumericGrid thresh_map, boolean thresh,
			int counter) throws IOException {

		if (thresh == true) {
			NumericPointwiseOperators.multiplyBy(dark, thresh_map);
			NumericPointwiseOperators.multiplyBy(abso, thresh_map);
		}

		List<Float> abslist = new ArrayList<Float>();
		List<Float> darklist = new ArrayList<Float>();

		Grid2D abs = (Grid2D) abso;
		Grid2D dar = (Grid2D) dark;

		String pathdark = pathtoproject + pathtofile + "DataSheets/dark.csv";
		String pathabso = pathtoproject + pathtofile + "DataSheets/abso.csv";

		FileWriter fileWriterdark = new FileWriter(pathdark);
		PrintWriter printWriterdark = new PrintWriter(fileWriterdark);

		FileWriter fileWriterabso = new FileWriter(pathabso);
		PrintWriter printWriterabso = new PrintWriter(fileWriterabso);

		for (int i = 0; i < abs.getWidth(); i++) {
			for (int j = 0; j < abs.getHeight(); j++) {

//				 punkt in file reinschreiben
				if (counter == 0) {
					if ((i > xstart && i < xend) || (i > xstart2 && i < xend2)) {
						continue;

					}
				}

				if (((Grid2D) splitted_dark).getAtIndex(i, j) == (float) 1000) {
					continue;
				}

//				if ((0 == (int) abs.getAtIndex(i, j)) && ( 0 == (int) dar.getAtIndex(i, j))) {
//					System.out.println("ist 0 0 ");
//					continue;
//
//				}
//				if(abs.getAtIndex(i, j) != (float) 0.0) {
				printWriterdark.println(dar.getAtIndex(i, j));
				printWriterabso.println(abs.getAtIndex(i, j));
//				}

//				abslist.add(abs.getAtIndex(i, j));
//				darklist.add(dar.getAtIndex(i, j));
			}
		}

		printWriterabso.close();
		printWriterdark.close();

		// get rid of duplicates
		// TODO cut duplicates with new sorting
		// List<Float> newAbs = new ArrayList<Float>();
//        for (Float element : abslist) {  
//            if (!newAbs.contains(element)) { 
//                newAbs.add(element); 
//            } 
//        } 
//		
//      
//        ArrayList<Float> newDark = new ArrayList<Float>(); 
//        for (Float element : darklist) { 
//            if (!newDark.contains(element)) { 
//                newDark.add(element); 
//            } 
//        } 

//		ListInFile.export(darklist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/dark.csv", "dark-values");
//		ListInFile.export(abslist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/abso.csv", "amp-values");

//		String pathdark = pathtoproject + pathtofile + "Datasheets/dark.csv";
//		String pathabso = pathtoproject + pathtofile + "Datasheets/abso.csv";

		BufferedReader csvDarkReader = new BufferedReader(new FileReader(pathdark));
		BufferedReader csvAbsoReader = new BufferedReader(new FileReader(pathabso));

		String[] darkpoints = new String[200 * 360];
		String[] absopoints = new String[200 * 360];

		int darkcounter = 0;
		String darkline;
		while ((darkline = csvDarkReader.readLine()) != null) {
			darkpoints[darkcounter] = darkline;
			darkcounter++;
			// do something with the data
		}

		int absocounter = 0;
		String absoline;
		while ((absoline = csvAbsoReader.readLine()) != null) {
			absopoints[absocounter] = absoline;
			absocounter++;
		}

		csvDarkReader.close();
		csvAbsoReader.close();

		System.out.println("laenge: " + darkpoints[70000]);
		System.out.println("laenge: " + darkpoints[20000]);

		double[][] points = new double[absocounter][2];

		for (int i = 0; i < absocounter; i++) {
			points[i][0] = Double.parseDouble(darkpoints[i]);
			points[i][1] = Double.parseDouble(absopoints[i]);
		}

//		System.out.println(newAbs.size());
//		System.out.println(points.length);
//		for (int i = 0; i < 200*360; i++) {
//	        points[i][0] = darklist.get(i);
//	        points[i][1] = abslist.get(i);
//	    }

		return points;
	}

	public static NumericGrid[] get_masks(NumericGrid[] segmentations) {
		NumericGrid[] masks = new NumericGrid[segmentations.length];
		for (int i = 0; i < segmentations.length; i++) {
			Grid2D temp = new Grid2D(size, size);
			for (int k = 0; k < size; k++) {
				for (int l = 0; l < size; l++) {
					int[] idx = { k, l };
					if (segmentations[i].getValue(idx) == (float) 0.0) {
						temp.setValue(0, idx);
						continue;
					} else {
						temp.setValue(1, idx);
					}
				}
			}
			temp.show("maske temp");
			masks[i] = temp;
		}

		return masks;
	}

	public static double[][] get_lgs_values_x(NumericGrid[] masks_sino, String variant) {
		double[][] x = new double[200 * 360][masks_sino.length];
		if (variant == "amp") {
			for (int i = 0; i < masks_sino.length; i++) {
				int counter = 0;
				for (int k = 0; k < 200; k++) {
					for (int l = 0; l < 360; l++) {
						int[] idx = { k, l };
						x[counter][i] = masks_sino[i].getValue(idx);
						counter++;
					}
				}
			}
		}

		if (variant == "dark") {
			for (int i = 0; i < masks_sino.length; i++) {
				int counter = 0;
				for (int k = 0; k < 200; k++) {
					for (int l = 0; l < 360; l++) {

						if (k < xend || k > xstart2) {
							continue;
						}

						int[] idx = { k, l };
						x[counter][i] = masks_sino[i].getValue(idx);
						counter++;
					}
				}
			}
		}

		return x;
	}

	public static double[] get_lgs_values_y(NumericGrid sinogram, String variant) {
		double[] y = new double[200 * 360];

		if (variant == "amp") {
			int counter = 0;
			for (int k = 0; k < 200; k++) {
				for (int l = 0; l < 360; l++) {
					int[] idx = { k, l };
					y[counter] = sinogram.getValue(idx);
					counter++;
				}
			}
		}

		if (variant == "dark") {
			int counter = 0;
			for (int k = 0; k < 200; k++) {
				for (int l = 0; l < 360; l++) {

					if (k < xend || k > xstart2) {
						continue;
					}

					int[] idx = { k, l };
					y[counter] = sinogram.getValue(idx);
					counter++;
				}
			}
		}

		return y;
	}

	/**
	 * corrects absorption
	 * 
	 * @param dark     - Sinogram of DFI
	 * @param abso     - Sinogram of Absorption
	 * @param thresh   - thresholding value
	 * @param iter_num - number of iterations
	 *
	 * @return dabso - correctet absoption image
	 * @throws IOException
	 */
	public static NumericGrid correct_absorption(NumericGrid dark_orig, NumericGrid dark, NumericGrid abso,
			double thresh, int iter_num) throws IOException {

		Grid2D thresh_map = new Grid2D(size, size);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				thresh_map.setAtIndex(i, j, 1);
			}
		}
//		Grid2D dabso = new Grid2D(size, size);

		// calc inital fabso
		double[][] points = get_comp_points(abso, dark, thresh_map, true, 1);
		PolynomialRegression regression = PolynomialRegression.calc_regression(points, 3);

		// calc thresholding map
		int iter = 1;
		while (iter <= iter_num) {
			System.out.println("correct_absorption iteration: " + iter);
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					int idx[] = { i, j };
					float temp = abso.getValue(idx) - dark.getValue(idx);
					if (regression.beta(2) * Math.pow(temp, 2) + regression.beta(1) * temp
							+ regression.beta(0) < (thresh * NumericPointwiseOperators.max(dark))) {
						thresh_map.setAtIndex(i, j, 1);
					} else {
						thresh_map.setAtIndex(i, j, 0);
					}

				}
			}

			// calc new fabso
			points = get_comp_points(abso, dark, thresh_map, true, 1);
			regression = PolynomialRegression.calc_regression(points, 3);

			iter++;
		}

		Grid2D end = new Grid2D(size, size);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				int idx[] = { i, j };
				float temp = abso.getValue(idx);
				double value = regression.beta(2) * Math.pow(temp, 2) + regression.beta(1) * temp + regression.beta(0);
				end.addAtIndex(i, j, (float) value);
			}
		}

		end.show("end");

		int[] hallo = end.getSize();
		int[] ciao = dark.getSize();
		System.out.println(hallo[0]);
		System.out.println(ciao[0]);

		NumericGrid dabso = NumericPointwiseOperators.subtractedBy(dark_orig, end);
		dabso.show("dabso");
		return dabso;
	}

	public static NumericGrid fill_up_sinogram(NumericGrid sino_dark, NumericGrid sino_abso,
			PolynomialRegression regression) {

		Grid2D abs_global = (Grid2D) sino_abso;
		Grid2D dark_global = (Grid2D) sino_dark;

		Grid2D abs = new Grid2D(200, 360);
		Grid2D dark = new Grid2D(200, 360);

		for (int k = 0; k < abs.getWidth(); k++) {
			for (int l = 0; l < abs.getHeight(); l++) {
				float val_amp = abs_global.getAtIndex(k, l);
				float val_dark = dark_global.getAtIndex(k, l);

				abs.setAtIndex(k, l, val_amp);
				dark.setAtIndex(k, l, val_dark);

			}
		}

		for (int i = 0; i < abs.getWidth(); i++) {
			for (int j = 0; j < abs.getHeight(); j++) {

				float temp = abs.getAtIndex(i, j);
				float val = (float) (regression.beta(3) * Math.pow(temp, 3) + regression.beta(2) * Math.pow(temp, 2)
						+ regression.beta(1) * temp + regression.beta(0)); // (regression.beta(3) * Math.pow(temp, 3) +
																			// (regression.beta(2) * Math.pow(temp, 2)

				if (i < 75 && i < 125) {
					System.out.println("i: " + i + "j: " + j + "val: " + val);
				}

//				if (((i > xstart && i < xend) || (i > xstart2 && i < xend2)) && (temp != (float) 0.0)) {
//					dark.setAtIndex(i, j, val);
//
//				}
				dark.setAtIndex(i, j, val);

			}
		}

		return dark;
	}

	public static NumericGrid calculate_sino_dark(NumericGrid amp_material, NumericGrid sino_dark_trunc,
			NumericGrid[] materials, int counter) throws IOException {
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(nr_of_projections, 2 * Math.PI);
		amp_material.show("ich bin der fehler");
		Grid2D amp_mat = (Grid2D) amp_material;
		Grid2D sino_amp_mat = p.project(amp_mat, new Grid2D(200, 360));
		Grid2D thresh_map = new Grid2D(size, size);
//		pci_sino.getDark().show("Dark");
//	pci_sino.getAmp().show("Amp");

		if (counter == 0) {
			int random = (int) Math.random();
			sino_dark_trunc = split_dark(pci_sino_truncated.getDark(), amp_material, materials);
//			sino_dark_trunc.show("splitted dark");
			if (saveImagesCheck) {
				ImagePlus imp8 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) sino_dark_trunc).createImage());
				String path8 = path + "splitted_dark" + random;
				IJ.saveAs(imp8, "png", path8);
			}

		}

		double[][] points = get_comp_points(sino_amp_mat, sino_dark_trunc, thresh_map, false, counter);

		// Absorption Correction
//		NumericGrid dabso = correct_absorption(pci.getDark(), pci_sino.getDark(), pci.getAmp(), 2, 10);
//		dabso.show();

//		double [][] points2 = {{2 ,3}, {2, 8}, {3, 4} ,{10, 5} ,{14, 3} ,{3, 8}};

		// Calculate regression and put the values to list
		PolynomialRegression regression = PolynomialRegression.calc_regression(points, 3);
		List<Float> reglist = new ArrayList<Float>();
		for (int i = 0; i <= 3; i++) {
			reglist.add((float) regression.beta(i));
		}
		String pathreg = pathtoproject + pathtofile + "Datasheets/reg.csv";
		ListInFile.export(reglist, pathreg, "regression");

		// execute Python script to visualize data
		String command = "py " + pathtoproject + pathtofile + "PythonScripts/polynom.py";
		try {
			Process process = Runtime.getRuntime().exec(command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// fill up unknown dark_sinogram data
		NumericGrid filled_dark_sino = fill_up_sinogram(sino_dark_trunc, sino_amp_mat, regression);
//		filled_dark_sino.show("filled_up");
//
////		// build picture with ones for scaling the backprojection
////		Grid2D ones = new Grid2D(size, size);
////		for (int i = 0; i < size; i++) {
////			for (int j = 0; j < size; j++) {
////				ones.setAtIndex(i, j, 1);
////			}
////		}
////
////		// forward and back projection
//		ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2 * Math.PI);
////		Grid2D sino_ones = o.project(ones, new Grid2D(200, 360));
//////				sino_ones.show("sino_ones");
////		Grid2D ones_reko = o.backprojection_pixel(sino_ones, size);
////		NumericGrid reko_temp = p.backprojection_pixel((Grid2D) filled_dark_sino, size);
////		NumericPointwiseOperators.divideBy(reko_temp, ones_reko);
//
//		NumericGrid reko_fbp = o.filtered_backprojection((Grid2D) filled_dark_sino, size);
//
////		reko_fbp.show("reko_fbp");
//
//		// save image in path
//		ImagePlus imp = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) reko_temp).createImage());
//		String path = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/reko";
//		IJ.saveAs(imp, "png", path);
//
//		ImagePlus imp8 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) reko_fbp).createImage());
//		String path8 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/reko_fbp";
//		IJ.saveAs(imp8, "png", path8);
//
////		reko_temp.show("filled up reko");
//
//		pci_sino2.getDark().show("Dark start");
//		NumericGrid diff = NumericPointwiseOperators.subtractedBy(filled_dark_sino, pci_sino_truncated.getDark());
//		diff.show("Differenz");
//
//		// visualize differents
//		ImagePlus imp3 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) diff).createImage());
//		IJ.saveAs(imp3, "png", "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/difference");
//		
//		float error = NumericPointwiseOperators.mean(diff);
//		errorlist.add(error);
		return filled_dark_sino;
	}

	public static NumericGrid iterative_dark_reconstruction(NumericGrid dark_sino) {
		// build picture with ones for scaling the backprojection
		Grid2D ones = new Grid2D(size, size);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				ones.setAtIndex(i, j, 1);
			}
		}

		// forward and back projection
		ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2 * Math.PI);
		Grid2D sino_ones = o.project(ones, new Grid2D(200, 360));
//		sino_ones.show("sino_ones");
		Grid2D ones_reko = o.backprojection_pixel(sino_ones, size);
//		ones_reko.show("ones_reko");

		// lokalisation algorithm : wie schaut das DFI aus ? braucht man das?

		// fill up sinogram
		int i = 1;
		while (i < iter_num) {

		}
		NumericGrid dark_end = null;
		return dark_end;
	}

	public static PhaseContrastImages iterative_reconstruction(NumericGrid orig_dark, PhaseContrastImages pci_sino,
			int iter_num, int error) {

		// Build picture with ones
		Grid2D ones_amp = new Grid2D(size, size);
		Grid2D ones_phase = new Grid2D(size, size);
		Grid2D ones_dark = new Grid2D(size, size);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				ones_amp.setAtIndex(i, j, 1);
				ones_phase.addAtIndex(i, j, 1);
				ones_dark.setAtIndex(i, j, 1);
			}
		}

		// forward + back projection
		PhaseContrastImages ones = new PhaseContrastImages(ones_amp, ones_phase, ones_dark);
//		ones.show("ones");
		ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2 * Math.PI);
		PhaseContrastImages sino_ones = o.project(ones, new Grid2D(200, 360));
//		sino_ones.show("sino_ones");
		PhaseContrastImages ones_reko = o.backprojection_pixel(sino_ones, size);
//		ones_reko.show("ones_reko");

		// Build empty picture
		Grid2D recon_amp = new Grid2D(size, size);
		Grid2D recon_phase = new Grid2D(size, size);
		Grid2D recon_dark = new Grid2D(size, size);
		PhaseContrastImages pci_recon = new PhaseContrastImages(recon_amp, recon_phase, recon_dark);
//		pci_recon.show("old pci_recon");

		// iteration
		int i = 1;
		while (i <= iter_num || error == 5) {

			System.out.println("Iteration Nr: " + i);

			// Project picture
			ProjectorAndBackprojector p = new ProjectorAndBackprojector(360, 2 * Math.PI);
			PhaseContrastImages sino_recon = p.project(pci_recon, new Grid2D(200, 360));
//			sino_recon.show("sino_recon");

			// !!! Dark bild vergleicht sich mit sich selbst dementsprechend wird es nur
			// immer schlechter !!!

			// Build difference of recon_sino and given sino
			NumericPointwiseOperators.subtractBy(sino_recon.getAmp(), pci_sino.getAmp());
			NumericPointwiseOperators.subtractBy(sino_recon.getPhase(), pci_sino.getPhase());
			NumericPointwiseOperators.subtractBy(sino_recon.getDark(), pci_sino.getDark());
//			sino_recon.show("sino_difference");

			// Todo: refinement and the other stuff ...

			// Backproject to reconstruct
			PhaseContrastImages pci_reko = p.backprojection_pixel(sino_recon, size);
//			pci_reko.show("backprojected");

			// scale/normalise
			NumericPointwiseOperators.divideBy(pci_reko.getAmp(), ones_reko.getAmp());
			NumericPointwiseOperators.divideBy(pci_reko.getPhase(), ones_reko.getPhase());
			NumericPointwiseOperators.divideBy(pci_reko.getDark(), ones_reko.getDark());
//			pci_recon.show("scaled");

			// scale with alpha
			double alpha = 0.25;
			NumericPointwiseOperators.multiplyBy(pci_reko.getAmp(), (float) alpha);
			NumericPointwiseOperators.multiplyBy(pci_reko.getPhase(), (float) alpha);
			NumericPointwiseOperators.multiplyBy(pci_reko.getDark(), (float) alpha);

			// subtrackt on picture
			NumericPointwiseOperators.subtractBy(pci_recon.getAmp(), pci_reko.getAmp());
			NumericPointwiseOperators.subtractBy(pci_recon.getPhase(), pci_reko.getPhase());
			NumericPointwiseOperators.subtractBy(pci_recon.getDark(), pci_reko.getDark());
//			pci_recon.show("recon");

			if (i == 10 || i == 20 || i == 50 || i == 70 || i == 100) {

				pci_recon.show("recon" + i);

				ImagePlus imp = new ImagePlus("Filled",
						ImageUtil.wrapGrid2D((Grid2D) pci_recon.getDark()).createImage());
				String path = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/filled"
						+ Integer.toString(i);
				IJ.saveAs(imp, "png", path);
			}

			// Errorfunction
			NumericGrid orig_dark_copy = orig_dark;
			NumericGrid pci_recon_dark_copy = pci_recon.getDark();

			NumericPointwiseOperators.multiplyBy(orig_dark_copy, orig_dark_copy);
			NumericPointwiseOperators.multiplyBy(pci_recon_dark_copy, pci_recon_dark_copy);
			NumericPointwiseOperators.addBy(orig_dark_copy, pci_recon_dark_copy);
			NumericPointwiseOperators.sqrt(orig_dark_copy);

			float mean = NumericPointwiseOperators.mean(orig_dark_copy);
			errorlist.add(mean);

			i++;
		}

		String patherror = pathtoproject + pathtofile + "Datasheets/error.csv";
		ListInFile.export(errorlist, patherror, "error-values");

		System.out.println("iterative reconstruction done");
		return pci_recon;
	}

	public static void all(String[] args) throws IOException, InterruptedException {
		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- + initializing args values from userinterface frame
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

		simchecked = Boolean.parseBoolean(args[0]);
		if (simchecked) {
			nr_ellipses = Integer.parseInt(args[1]);
		}

		trcchecked = Boolean.parseBoolean(args[2]);
		if (trcchecked) {
			xstart = Integer.parseInt(args[3]);
			xend = Integer.parseInt(args[4]);
			xstart2 = Integer.parseInt(args[5]);
			xend2 = Integer.parseInt(args[6]);
			ystart = Integer.parseInt(args[7]);
			yend = Integer.parseInt(args[8]);
			ystart2 = Integer.parseInt(args[9]);
			yend2 = Integer.parseInt(args[10]);
			only_dark = Boolean.parseBoolean(args[11]);
			value = Integer.parseInt(args[12]);
		}

//		iterchecked = Boolean.parseBoolean(args[13]);

		iter_num = Integer.parseInt(args[13]);
		error_val = Integer.parseInt(args[14]);

		vischecked = Boolean.parseBoolean(args[15]);

		noisechecked = Boolean.parseBoolean(args[16]);
		if (noisechecked) {
			noisetype = args[17];

		}
		singleMaterialCheck = Boolean.parseBoolean(args[18]);
		multipleMaterialCheck = Boolean.parseBoolean(args[19]);
		path = args[20];
		saveImagesCheck = Boolean.parseBoolean(args[21]);
		pathtoproject = args[22];

		// time
		final long timeStart = System.currentTimeMillis();

//		String projectpath = pathtoproject + pathtofile + "pathtoproject.csv";
//		FileWriter csvWriter = new FileWriter(projectpath);
//		csvWriter.append(pathtoproject);
//		csvWriter.flush();
//		csvWriter.close();

		// start ImageJ
		new ImageJ();

		/*
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 * 
		 * + Simulate the data
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */
		ProjectorAndBackprojector p = new ProjectorAndBackprojector(nr_of_projections, 2 * Math.PI);
		if (simchecked) {

			// create phantoms
			pci = simulate_data();
			pci.show("pci");
			ImagePlus imp = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) pci.getDark()).createImage());
			String path = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/groundTruthDark";
			IJ.saveAs(imp, "png", path);

			ImagePlus imp2 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) pci.getAmp()).createImage());
			String path2 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/groundTruthAbsorption";
			IJ.saveAs(imp2, "png", path2);
			if (noisechecked) {
				NumericGrid noisy_amp = add_noise(pci.getAmp(), 0, 0.005, "gaussian");
				NumericGrid noisy_dark = add_noise(pci.getDark(), 0, 0.005, "gaussian");
				pci.getAmp().show("ich bin noisy amp");
				pci.getDark().show("ich bin noisy dark");
			}

//			Grid2D blurred = blur(pci.getAmp(), 10, 49);
//			ImagePlus imp7 = new ImagePlus("Filled", ImageUtil.wrapGrid2D(blurred).createImage());
//			String path7 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/gaussian_blurred";
//			IJ.saveAs(imp7,"png", path7);
//			

			// define geometry
			int detector_width = 200;
			int nr_of_projections = 360;

			pci_sino = p.project(pci, new Grid2D(detector_width, nr_of_projections));

			instreko = p.filtered_backprojection((Grid2D) pci_sino.getDark(), size);
			instreko.show("direkte fbp");

			ImagePlus imp5 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) pci_sino.getAmp()).createImage());
			String path5 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/AbsorptionSinogram";
			IJ.saveAs(imp5, "png", path5);

			ImagePlus imp6 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) pci_sino.getDark()).createImage());
			String path6 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/OrigDarkSino";
			IJ.saveAs(imp6, "png", path6);
			pci_sino.getDark().show("orig dark");

			System.out.println("simulated data with nr_ellipses: " + nr_ellipses);

			/*
			 * -----------------------------------------------------------------------------
			 * -------------------------
			 * 
			 * + Truncate data
			 * -----------------------------------------------------------------------------
			 * -------------------------
			 */

			if (trcchecked) {
				PhaseContrastImages pci_sino_copy = pci_sino;
				pci_sino_truncated = fake_truncation(pci_sino_copy, xstart, xend, xstart2, xend2, "x", value, false,
						true, "pending");
//				pci_sino.show("pci-fake");	
//				truncated_dark_sino = fake_truncation(pci_sino_copy, xstart, xend, xstart2, xend2, "x", value, false,
//						true, "pending").getDark();

				ImagePlus imp3 = new ImagePlus("Filled",
						ImageUtil.wrapGrid2D((Grid2D) pci_sino_truncated.getDark()).createImage());
				String path3 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/TruncatedDfSinogram";
				IJ.saveAs(imp3, "png", path3);

				ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2 * Math.PI);
				trunc_reko = o.backprojection_pixel((Grid2D) pci_sino_truncated.getDark(), size);
				trunc_reko.show("truncated");
				
				ImagePlus imp4 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) trunc_reko).createImage());
				String path4 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/TruncatedDfi";
				IJ.saveAs(imp4, "png", path4);

				System.out.println("truncated data from " + xstart + " to " + xend + " with value " + value);
				System.out.println("truncated data from " + ystart + " to " + yend + " with value " + value);

			}

			// backproject (reconstruct)
//			pci_reko = p.filtered_backprojection(pci_sino, size);
//			pci_reko.show("reconstruction");

			// calculate the difference
//			PhaseContrastImages pci_diff = calculate_PCI(pci, pci_reko, false);
//			pci_diff.show("difference");

		}

		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- + Compute values for the comparison and export them
		 * to csv to do some calculations in python + Calculate the Polynomial
		 * regression function matching to the comparison points
		 * 
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 * 
		 */
		if (multipleMaterialCheck) {

			// copy
			Grid2D trcdark = (Grid2D) pci_sino_truncated.getDark();
			Grid2D sino_dark_trunc = new Grid2D(200, 360);
			for (int k = 0; k < trcdark.getWidth(); k++) {
				for (int l = 0; l < trcdark.getHeight(); l++) {
					float val = trcdark.getAtIndex(k, l);

					sino_dark_trunc.setAtIndex(k, l, val);

				}
			}

			NumericGrid[] amp_materials = get_segmentation(pci.getAmp(), 0.01);
			// sinograms of materials
			NumericGrid[] amp_materials_sino = new NumericGrid[amp_materials.length];
			for (int i = 0; i < amp_materials.length; i++) {
				amp_materials_sino[i] = p.project((Grid2D) amp_materials[i], new Grid2D(200, 360));
				amp_materials_sino[i].show("sino maske");

				ImagePlus imp5 = new ImagePlus("Filled",
						ImageUtil.wrapGrid2D((Grid2D) pci_sino.getAmp()).createImage());
				String path5 = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/masks_sinogramm"
						+ i;
				IJ.saveAs(imp5, "png", path5);
			}

			NumericGrid[] masks = get_masks(amp_materials);

			// sinograms of masks
			NumericGrid[] masks_sino = new NumericGrid[masks.length];
			for (int i = 0; i < masks.length; i++) {
				masks_sino[i] = p.project((Grid2D) masks[i], new Grid2D(200, 360));
				masks_sino[i].show("sino maske");
			}

			double[][] x_amp = get_lgs_values_x(masks_sino, "amp");
			double[][] x_dark = get_lgs_values_x(masks_sino, "dark");
			pci_sino.getAmp().show("AMP sino");
			double[] y_amp = get_lgs_values_y(pci_sino.getAmp(), "amp");
			double[] y_dark = get_lgs_values_y(pci_sino_truncated.getDark(), "dark");

			System.out.println(Arrays.toString(x_amp[0]));

			MultipleLinearRegression regression = new MultipleLinearRegression(x_amp, y_amp);

//			System.out.printf("%.2f + %.2f beta1  (R^2 = %.2f)\n", regression.beta(0), regression.beta(1),
//					regression.R2());

			MultipleLinearRegression regression_dark = new MultipleLinearRegression(x_dark, y_dark);

//			System.out.printf("%.2f + %.2f beta1  (R^2 = %.2f)\n", regression_dark.beta(0), regression_dark.beta(1),
//					regression_dark.R2());

			Grid2D add = new Grid2D(200, 360);
//			NumericPointwiseOperators.copy(add, pci_sino_truncated.getDark());
			NumericGrid[] fill_up = new NumericGrid[masks.length];
			double[] correlation = new double[masks.length];
			for (int i = 0; i < masks.length; i++) {
				Grid2D dark_sino = new Grid2D(200, 360);
				correlation[i] = regression_dark.beta(i) / regression.beta(i);

				for (int k = 0; k < 200; k++) {
					for (int l = 0; l < 360; l++) {
//						if (k >= xend && k <= xstart2) {
//							continue;
//						}
						int[] idx = { k, l };
						float amp_value = amp_materials_sino[i].getValue(idx);
						float val = (float) (amp_value * correlation[i]);
						dark_sino.setAtIndex(k, l, val);
					}
				}
				fill_up[i] = dark_sino;
				NumericPointwiseOperators.addBy(add, dark_sino);
			}

			add.show("added");

			dark_reco = p.filtered_backprojection(add, size);

			dark_reco.show("last reco");

			NumericGrid difference = NumericPointwiseOperators.subtractedBy(pci_sino.getDark(), add);
			double error = Math.abs(NumericPointwiseOperators.mean(difference));
			System.out.print("The overall error is: " + error);

		}

		if (singleMaterialCheck) {
			splitted_dark = new Grid2D(200, 360);
			// copy
			Grid2D trcdark = (Grid2D) pci_sino_truncated.getDark();
			Grid2D sino_dark_trunc = new Grid2D(200, 360);
			for (int k = 0; k < trcdark.getWidth(); k++) {
				for (int l = 0; l < trcdark.getHeight(); l++) {
					float val = trcdark.getAtIndex(k, l);

					sino_dark_trunc.setAtIndex(k, l, val);

				}
			}

			NumericGrid[] amp_materials = get_segmentation(pci.getAmp(), 0.01);
			NumericGrid[] filled_sinos = new NumericGrid[amp_materials.length];
			NumericGrid[] trunc_filled_sinos = new NumericGrid[amp_materials.length];
			Grid2D all_sinos = new Grid2D(200, 360);
			all_sinos.show("all sinos");
			for (int j = 1; j < amp_materials.length; j++) {
				for (int counter = 0; counter < iter_num; counter++) {
					ImagePlus imp = new ImagePlus("Filled",
							ImageUtil.wrapGrid2D((Grid2D) sino_dark_trunc).createImage());
					String path = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/reko" + counter;
					IJ.saveAs(imp, "png", path);
//					pci_sino.getDark().show("reko dark");

					sino_dark_trunc.show("sino_dark_trunc" + counter);

					filled_sinos[j] = calculate_sino_dark(amp_materials[j], sino_dark_trunc, amp_materials, counter);
					filled_sinos[j].show("filled sinos [j]" + counter);

					NumericGrid diff = NumericPointwiseOperators.subtractedBy(pci_sino.getDark(), sino_dark_trunc);
					NumericPointwiseOperators.abs(diff);
					ImagePlus imp3 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) diff).createImage());
					IJ.saveAs(imp3, "png",
							"C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/difference");
					float error = NumericPointwiseOperators.mean(diff);
					errorlist.add(error);
					System.out.println("error betraegt: " + error);

					NumericPointwiseOperators.copy(sino_dark_trunc, filled_sinos[j]);
					sino_dark_trunc.show("new sino");

				}

				System.out.println(Arrays.toString(filled_sinos[j].getSize()));
				trunc_filled_sinos[j] = fake_truncation_image((Grid2D) filled_sinos[j], xend, xstart2, "x", value);   //

				NumericPointwiseOperators.addBy(all_sinos, trunc_filled_sinos[j]);										// filled_sinos -> trunc_filled_sinos
				trunc_filled_sinos[j].show("trunc filled sino " + j);
				all_sinos.show("all_sinos");

			}
			String patherror = pathtoproject + pathtofile + "Datasheets/error.csv";
			ListInFile.export(errorlist, patherror, "error-values");
			NumericGrid trc_all_sinos = fake_truncation_image((Grid2D) all_sinos, xend, xstart2, "x", value);
//			trc_all_sinos.show("trc_all_sinos");
			pci_sino_truncated.getDark().show("lelel");
			NumericPointwiseOperators.addBy(all_sinos, pci_sino_truncated.getDark());
			all_sinos.show("endsinogramm");

			ImagePlus imp3 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) all_sinos).createImage());
			IJ.saveAs(imp3, "png",
					"C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/sinoendergebnis");

			dark_reco = p.filtered_backprojection((Grid2D) all_sinos, size);
			dark_reco.show("end reko");

			NumericGrid difference = NumericPointwiseOperators.subtractedBy(pci_sino.getDark(), all_sinos);
			double error = Math.abs(NumericPointwiseOperators.mean(difference));
			System.out.print("The overall error is: " + error);

			ImagePlus imp4 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D) dark_reco).createImage());
			IJ.saveAs(imp4, "png",
					"C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/fbp_rekonstruction_dark");

		}

//		pci.show("pci");
//		pci_sino.show("pci_sino");
//		pci_sino_truncated.show("pci_sino_truncated");
		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- compute iterative reconstruction
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

//		if (iterchecked) {
//			PhaseContrastImages end = iterative_reconstruction(pci.getDark(), pci_sino, iter_num, 0);
//
////			PhaseContrastImages end2 = iterative_reconstruction(pci.getDark(), pci_sino2, iter_num, 0);
//
//		}

		/*
		 * -----------------------------------------------------------------------------
		 * -------------------------- + execute BA.py to visualize the comparison points
		 * and the calculated polynom
		 * -----------------------------------------------------------------------------
		 * --------------------------
		 */
		if (vischecked) {
			String command1 = "py " + pathtoproject + pathtofile + "PythonScripts/error.py";
			try {
				Process process = Runtime.getRuntime().exec(command1);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- + dont know what to do with this, think its to do
		 * the visualization with the jFreechart
		 * 
		 * This is depracated due to the python solution
		 * 
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

//		double[][] line = new double[128][2];
//		for(int i = 0; i < 128; i ++){
//			double temp = regression.beta(2) * Math.pow(i/64, 2) + regression.beta(1) * (i/64) + regression.beta(0);
//			line[i][0] = i/64;
//			line[i][1] = temp;
//		}
//		LineChartEx.calc_linechart(line);

		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- + Space to add and test stuff
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

		NormalDistribution nd = new NormalDistribution(1000, 100);
		System.out.println("Normaldistrubution: " + nd);

		/*
		 * -----------------------------------------------------------------------------
		 * --------------------------- + console addings
		 * -----------------------------------------------------------------------------
		 * --------------------------
		 */

		final long timeEnd = System.currentTimeMillis();
		System.out.println("Executing Bubeck_Niklas_BA done in : " + (timeEnd - timeStart) + " Millisek.");

		ProjectorAndBackprojector o = new ProjectorAndBackprojector(360, 2 * Math.PI);
		PhaseContrastImages pci_recon = new PhaseContrastImages(pci_sino.getAmp(), pci_sino.getPhase(),
				pci_sino.getDark());
		PhaseContrastImages superposition = o.backprojection_pixel(pci_recon, size);
		superposition.show("superposition");
		
		get_line_plot_data(64);
		String command = "py " + pathtoproject + pathtofile + "PythonScripts/line_plots.py";
		try {
			Process process = Runtime.getRuntime().exec(command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	/**
	 * MAIN
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		Frame1.main(args);

	}

}
