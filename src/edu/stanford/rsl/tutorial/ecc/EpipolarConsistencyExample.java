/*
 * Copyright (C) 2015 Martin Berzl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.ecc;

import ij.ImageJ;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;

/**
 * this shows an implementation how to use the Epipolar Consistency class
 * to compute line integrals that fulfill the Epipolar Consistency Conditions
 * the shown data set is a pumpkin with projection views V01 and V07
 *
 */
public class EpipolarConsistencyExample {

	public static void main(String[] args) {
			
		ImageJ ij = new ImageJ();		

		// * directory for the image files (nrrd) *//
		// has to be changed!!
		// for example to: "C:\\Users\\NAME\\Desktop\\Epipolar Consistency\\Daten"
		String directory = "";
		
		
		//* set up two views by creating the class instances *//
		// these views are going to be compared in the following
		// the corresponding xml file is called "ConradNRRD.xml"
		// last index stands for the index of the projection matrices in the xml file
		// ( PMatrixSerialization; 0 is the first projection matrix )
		EpipolarConsistency epi1 = new EpipolarConsistency(directory, "V01.nrrd", "V01_rda.nrrd", "nrrd", 0);
		EpipolarConsistency epi2 = new EpipolarConsistency(directory, "V07.nrrd", "V07_rda.nrrd", "nrrd", 1);
		
		
		// if you have tiff images, you need to use the following:
		// radon transformation is going to be calculated in the constructor
		// the xml file is called "ConradTIFF.xml"
		/*
		String tiffFile = ".tiff"; // has to be changed
		EpipolarConsistency epi1 = new EpipolarConsistency(directory, tiffFile, "", "tiff", 10);
		EpipolarConsistency epi2 = new EpipolarConsistency(directory, tiffFile, "", "tiff", 50);
		*/
		
		
		//* get the mapping matrix to the epipolar plane *//
		SimpleMatrix K = EpipolarConsistency.createMappingToEpipolarPlane(epi1.C, epi2.C);
		// (K is a 4x3 matrix)
		
		//* calculate inverses of projection matrices *//
		SimpleMatrix Pa_Inverse = epi1.P.inverse(InversionType.INVERT_SVD);
		SimpleMatrix Pb_Inverse = epi2.P.inverse(InversionType.INVERT_SVD);
				

		//* go through angles *//
		// we go through a range of [-8°, +8°] in a stepsize of 0.05°
		double angleBorder = 8.0;
		double angleIncrement = 0.05;
		// get number of decimal places of angleIncrement
		String[] split = Double.toString(angleIncrement).split("\\.");
		int decimalPlaces = split[1].length();
		
		int height = (int) (angleBorder * 2 / angleIncrement + 1);
	
		// results are saved in an array in the format [angle,valueView1,valueView2]
		double[][] results = new double[height][3];
		int count = 0;
				
		for (double kappa = -angleBorder; kappa <= angleBorder; kappa += angleIncrement) {
			
			double kappa_RAD = kappa / 180.0 * Math.PI;
			
			//* get values for line integrals that fulfill the epipolar consistency conditions *//
			double[] values = EpipolarConsistency.computeEpipolarLineIntegrals(kappa_RAD, epi1, epi2, K, Pa_Inverse, Pb_Inverse);
			results[count][0] = Math.round(kappa*Math.pow(10, decimalPlaces)) / (Math.pow(10, decimalPlaces) + 0.0);
			results[count][1] = values[0];
			results[count][2] = values[1];
			count++;
		}

		//* show results *//
		for (int i = 0; i < results.length; i++) {
			System.out.println("at angle kappa: " + results[i][0] + " P0: " + results[i][1] + " P1: " + results[i][2]);
		}

	}

}
