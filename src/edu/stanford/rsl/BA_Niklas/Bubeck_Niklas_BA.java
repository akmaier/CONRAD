package edu.stanford.rsl.BA_Niklas;

import edu.stanford.rsl.BA_Niklas.UserInterface;
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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.io.*;

/**
 * Bachelor thesis of Niklas Bubeck
 * 
 * This class simulates data, forward-projects it and does a
 * filtered-backprojection.
 * 
 * @author Lina Felsner
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
	static PhaseContrastImages pci_sino2;
	static PhaseContrastImages pci_reko;

	/**
	 * creates random ellipses
	 */
	public static void create_ellipse() {
		R = new SimpleMatrix[nr_ellipses];
		t = new double[nr_ellipses][2];
		ab = new double[nr_ellipses][2];
		rho = new double[nr_ellipses];
		df = new double[nr_ellipses];

		for (int i = 0; i < nr_ellipses; i++) {

			// translation between -0.5 and 0.5
			t[i][0] = Math.random() - 0.5; // delta_x
			t[i][1] = Math.random() - 0.5; // delta_y

			// size between 0 and 0.75
			ab[i][0] = Math.random() * 2; // a
			ab[i][1] = Math.random() * 2; // b

			// rotation
			double alpha = Math.random() * 90;
			R[i] = new SimpleMatrix(
					new double[][] { { Math.cos(alpha), -Math.sin(alpha) }, { Math.sin(alpha), Math.cos(alpha) } });

			// values
			rho[i] = Math.random();
			df[i] =  Math.random();

		}
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
	public static PhaseContrastImages fake_truncation(PhaseContrastImages pci_sino, int min, int max, int min2,
			int max2, String axes, int value, boolean random, boolean only_dark, String filter) {
		int width = pci_sino.getWidth();
		int height = pci_sino.getHeight();
		Grid2D fake_amp = (Grid2D) pci_sino.getAmp();
		Grid2D fake_dark = (Grid2D) pci_sino.getDark();
		Grid2D fake_phase = (Grid2D) pci_sino.getPhase();
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

	public static double[][] get_comp_points(NumericGrid abso, NumericGrid dark, NumericGrid thresh_map, boolean thresh)
			throws IOException {

		if (thresh == true) {
			NumericPointwiseOperators.multiplyBy(dark, thresh_map);
			NumericPointwiseOperators.multiplyBy(abso, thresh_map);
		}

		List<Float> abslist = new ArrayList<Float>();
		List<Float> darklist = new ArrayList<Float>();

		Grid2D abs = (Grid2D) abso;
		Grid2D dar = (Grid2D) dark;

		FileWriter fileWriterdark = new FileWriter("C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/dark.csv");
		PrintWriter printWriterdark = new PrintWriter(fileWriterdark);

		FileWriter fileWriterabso = new FileWriter("C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/abso.csv");
		PrintWriter printWriterabso = new PrintWriter(fileWriterabso);

		for (int i = 0; i < abs.getWidth(); i++) {
			for (int j = 0; j < abs.getHeight(); j++) {

				// punkt in file reinschreiben
				if ((i > xstart && i < xend) || (i > xstart2 && i < xend2)) {
					System.out.println("ist 0 0 ");
					continue;

				}
				printWriterdark.println(dar.getAtIndex(i, j));
				printWriterabso.println(abs.getAtIndex(i, j));

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

		BufferedReader csvDarkReader = new BufferedReader(
				new FileReader("C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/dark.csv"));
		BufferedReader csvAbsoReader = new BufferedReader(
				new FileReader("C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/abso.csv"));

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
		double[][] points = get_comp_points(abso, dark, thresh_map, true);
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
			points = get_comp_points(abso, dark, thresh_map, true);
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

		Grid2D abs = (Grid2D) sino_abso;
		Grid2D dark = (Grid2D) sino_dark;

		for (int i = 0; i < abs.getWidth(); i++) {
			for (int j = 0; j < abs.getHeight(); j++) {
				float temp = abs.getAtIndex(i, j);
				float val = (float) (regression.beta(3) * Math.pow(temp, 3) + regression.beta(2) * Math.pow(temp, 2)
						+ regression.beta(1) * temp + regression.beta(0));
				System.out.println("i: " + i + "j: " + j + "val: " + val);
				if ((i > xstart && i < xend) || (i > xstart2 && i < xend2)) {
					dark.setAtIndex(i, j, val);
				}

			}
		}

		return sino_dark;
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

			
			
			// !!! Dark bild vergleicht sich mit sich selbst dementsprechend wird es nur immer schlechter !!!
			
			
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
				
				ImagePlus imp = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D)pci_recon.getDark()).createImage());
				String path = "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/filled" + Integer.toString(i);
				IJ.save(imp, path);
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

		ListInFile.export(errorlist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/error.csv", "error-values");

		System.out.println("iterative reconstruction done");
		return pci_recon;
	}

	/**
	 * MAIN
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		// time
		final long timeStart = System.currentTimeMillis();

		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- + initializing args values from userinterface frame
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

		Boolean simchecked = Boolean.parseBoolean(args[0]);
		if (simchecked) {
			nr_ellipses = Integer.parseInt(args[1]);
		}

		Boolean trcchecked = Boolean.parseBoolean(args[2]);
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

		Boolean iterchecked = Boolean.parseBoolean(args[13]);
		if (iterchecked) {
			iter_num = Integer.parseInt(args[14]);
			error_val = Integer.parseInt(args[15]);
		}

		Boolean vischecked = Boolean.parseBoolean(args[16]);

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

			// define geometry
			int detector_width = 200;
			int nr_of_projections = 360;

			pci_sino = p.project(pci, new Grid2D(detector_width, nr_of_projections));
			pci_sino2 = p.project(pci, new Grid2D(detector_width, nr_of_projections));

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
				pci_sino = fake_truncation(pci_sino, xstart, xend, xstart2, xend2, "x", value, false, true, "pending");
//				pci_sino = fake_truncation(pci_sino, ystart, yend, ystart2, yend2, "y", value, false, true,  "pending");
//				pci_sino.show("pci-fake");				
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
		 */

		Grid2D thresh_map = new Grid2D(size, size);
//			pci_sino.getDark().show("Dark");
//		pci_sino.getAmp().show("Amp");
		double[][] points = get_comp_points(pci_sino.getAmp(), pci_sino.getDark(), thresh_map, false);

//		double [][] points2 = {{2 ,3}, {2, 8}, {3, 4} ,{10, 5} ,{14, 3} ,{3, 8}};
		PolynomialRegression regression = PolynomialRegression.calc_regression(points, 3);
		List<Float> reglist = new ArrayList<Float>();
		for (int i = 0; i <= 3; i++) {
			reglist.add((float) regression.beta(i));
		}
		ListInFile.export(reglist, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/reg.csv", "regression");

		String command = "py C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/BA.py";
		try {
			Process process = Runtime.getRuntime().exec(command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		NumericGrid temp = fill_up_sinogram(pci_sino.getDark(), pci_sino.getAmp(), regression);
		temp.show("filled_up");
		NumericGrid reko_temp = p.filtered_backprojection_phase((Grid2D) temp, size);
		reko_temp.show("filled up reko");

		pci_sino2.getDark().show("Dark start");
		NumericGrid test = NumericPointwiseOperators.subtractedBy(pci_sino2.getDark(), pci_sino.getDark());
		test.show("Differenz");
		
		ImagePlus imp3 = new ImagePlus("Filled", ImageUtil.wrapGrid2D((Grid2D)test).createImage());
		imp3.show("imp3");
		IJ.save(imp3, "C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/test");
		
		/*
		 * -----------------------------------------------------------------------------
		 * ------------------------- compute iterative reconstruction
		 * -----------------------------------------------------------------------------
		 * -------------------------
		 */

		if (iterchecked) {
			PhaseContrastImages end = iterative_reconstruction(pci.getDark(), pci_sino, iter_num, 0);
	
//			PhaseContrastImages end2 = iterative_reconstruction(pci.getDark(), pci_sino2, iter_num, 0);
			
		}

		/*
		 * -----------------------------------------------------------------------------
		 * -------------------------- + execute BA.py to visualize the comparison points
		 * and the calculated polynom
		 * -----------------------------------------------------------------------------
		 * --------------------------
		 */
		if (vischecked) {
			String command1 = "py C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Files/BA.py";
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
		// Absorption Correction
//		NumericGrid dabso = correct_absorption(pci.getDark(), pci_sino.getDark(), pci.getAmp(), 2, 10);
//		dabso.show();

		/*
		 * -----------------------------------------------------------------------------
		 * --------------------------- + console addings
		 * -----------------------------------------------------------------------------
		 * --------------------------
		 */

		final long timeEnd = System.currentTimeMillis();
		System.out.println("Executing Bubeck_Niklas_BA done in : " + (timeEnd - timeStart) + " Millisek.");
	}

}
