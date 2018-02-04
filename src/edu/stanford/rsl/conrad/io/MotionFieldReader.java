/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;

import edu.stanford.rsl.conrad.fitting.Function;
import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.fitting.RANSACFittedFunction;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.utils.interpolation.Interpolator;
import edu.stanford.rsl.conrad.utils.interpolation.LinearInterpolator;

public class MotionFieldReader {

	private String filename = null;
	private static double[][] pos = null;
	private static double[][] mot = null;
	private int xDim = 0;
	private int yDim = 0;

	public MotionFieldReader() throws Exception {

		//this.filename = FileUtil.myFileChoose(".txt", false);
		this.filename = "C:\\Users\\motion.txt";
		if (readOut() == false) {
			System.err.println("Reading the motion field failed. Need format [\"pos\" #rows #cols");
		}
	}

	public boolean readOut() throws IOException {

		BufferedReader br = new BufferedReader(new FileReader(filename));

		//header
		String line = br.readLine();
		String[] parts = line.split(" ");

		if (!parts[0].equals("pos")) {
			br.close();
			return false;
		}
		//read position field
		xDim = Integer.valueOf(parts[1]);
		yDim = Integer.valueOf(parts[2]);
		pos = new double[xDim][yDim];
		mot = new double[xDim][yDim];
		for (int x = 0; x < xDim; x++) {
			line = br.readLine();
			parts = line.split("\t");
			for (int y = 0; y < yDim; y++) {
				pos[x][y] = Double.valueOf(parts[y]);
			}
		}

		for (int y = 0; y < yDim; y++) {
			for (int x = 0; x < xDim; x++) {
				mot[x][y] = pos[x][y] - pos[0][y];
			}
		}

		br.close();
		return true;
	}

	public double[][] getPosition() {
		return pos;
	}

	public double[][] getMotion() {
		return mot;
	}

	public double getMaxMotion() {
		double i1 = 0, i2 = 0;
		double maxMotion = 0;
		for (int j = 0; j < xDim; j++) {
			i2 = Math.abs(mot[j][0]);
			if (i2 > i1) {
				maxMotion = i2;
			}
			i1 = i2;
		}
		return maxMotion;
	}

	public Function getRANSACfittedFunction(double[] po, double[] mo) {
		RANSACFittedFunction ransac = new RANSACFittedFunction(new LinearFunction());
		ransac.setEpsilon(0.3);
		ransac.setNumberOfTries(10000);
		ransac.fitToPoints(po, mo);
		return ransac.getBaseFunction();
	}

	public Function getLinearfittedFunction(int i) {
		LinearFunction linear = new LinearFunction();
		linear.fitToPoints(pos[i], mot[i]);
		VisualizationUtil.createScatterPlot("Linear", pos[i], mot[i], linear).show();
		return linear;
	}

	/**
	 * Method to compute slope. All linear functions with a slope not equal to zero are averaged.
	 * @return slope
	 */
	public double getGlobalCompensationLinearScaling() {

		double res = 0.d;
		int c = 0;
		for (int i = 0; i < xDim; i++) {
			double m;
			if ((m = getRANSACfittedFunction(pos[i], mot[i]).getParametersAsDoubleArray()[0]) != 0.d) {
				res += m;
				c++;
			}
		}
		res /= c;
		return res;
	}

	/**
	 * Method to compute simple slope, based on maximum measured diaphragm motion and the point furthest on top
	 * @return slope
	 **/
	public double getGlobalCompensationLinearScalingSimple() {

		double m = (mot[xDim / 2][10] - mot[xDim / 2][0]) / (pos[xDim / 2][10] - pos[xDim / 2][0]);

		return m;
	}

	/**
	 * This method assumes a linear decrease from diaPos to maxPos
	 * @param diaMotion
	 * @param diaPos 
	 * @param maxPos
	 * @return slope
	 */
	public double getGlobalCompensationLinearMinMax(double diaMotion, double diaPos, double maxPos) {
		double m = (diaMotion) / (maxPos - diaPos);

		return m;
	}

	public double[] getBilinearInterpolationDataMotion(double diaphragmaMotion) {
		double diaMotFloor = 0;
		double diaMotCeil = 0;
		int index1 = 0;

		for (int i = 0; i < xDim; i++) {
			if (diaphragmaMotion <= mot[i][0]) {
				index1 = i;
				break;
			}
		}
		//if diaphragmPosition was larger than the maximum in the measurements, floor it to the maximum
		if (index1 == 0 && diaphragmaMotion > mot[0][0]) {
			double i1 = 0, i2 = 0;

			for (int j = 0; j < xDim; j++) {
				i2 = Math.abs(mot[j][0]);
				if (i2 > i1) {
					index1 = j;

					i1 = i2;
				}
			}
		}
		if (index1 > 0)
			diaMotFloor = mot[index1 - 1][0];
		else
			diaMotFloor = mot[index1][0];
		diaMotCeil = mot[index1][0];
		Interpolator inter = new LinearInterpolator();
		double result[] = new double[yDim];
		for (int i = 0; i < yDim; i++) {
			inter.setXPoints(diaMotFloor, diaMotCeil);
			if (index1 > 0) {
				inter.setYPoints(mot[index1 - 1][i], mot[index1][i]);
			} else {
				result[i] = mot[0][i];
				continue;
			}
			if ((diaMotCeil - diaMotFloor) != 0) {
				result[i] = (float) inter.InterpolateYValue(diaphragmaMotion);
			} else {
				result[i] = (float) ((mot[index1 - 1][i] + mot[index1][i]) / 2);

			}
		}
		return result;
	}

	public double[] getBilinearInterpolationDataPosition(double diaphragmaMotion) {
		double diaMotFloor = 0;
		double diaMotCeil = 0;
		int index1 = 0;

		for (int i = 0; i < xDim; i++) {
			if (diaphragmaMotion <= mot[i][0]) {
				index1 = i;
				break;
			}
		}
		//if diaphragmPosition was larger than the maximum in the measurements, floor it to the maximum
		if (index1 == 0 && diaphragmaMotion > mot[0][0]) {
			double i1 = 0, i2 = 0;

			for (int j = 0; j < xDim; j++) {
				i2 = mot[j][0];
				if (i2 <= i1) {
					index1 = j;
					i1 = i2;
				}
			}
		}
		if (index1 > 0)
			diaMotFloor = mot[index1 - 1][0];
		else
			diaMotFloor = mot[index1][0];
		diaMotCeil = mot[index1][0];
		Interpolator inter = new LinearInterpolator();
		double result[] = new double[yDim];
		for (int i = 0; i < yDim; i++) {
			inter.setXPoints(diaMotFloor, diaMotCeil);
			if (index1 == 0) {
				result[i] = pos[0][i];
			} else {
				inter.setYPoints(pos[index1 - 1][i], pos[index1][i]);
				if ((diaMotCeil - diaMotFloor) != 0) {
					result[i] = (float) inter.InterpolateYValue(diaphragmaMotion);
				} else {
					result[i] = (float) ((pos[index1 - 1][i] + pos[index1][i]) / 2);

				}
			}
		}
		return result;
	}

	public double bilinearInterpolation(double diaphragmaMotion, double positionZ) {
		Interpolator inter = new LinearInterpolator();

		double motFloor1 = 0;
		double motCeil1 = 0;
		double motFloor2 = 0;
		double motCeil2 = 0;
		double posFloor1 = 0;
		double posCeil1 = 0;
		double posFloor2 = 0;
		double posCeil2 = 0;
		double diaMotFloor = 0;
		double diaMotCeil = 0;
		int index1 = 0;

		for (int i = 0; i < xDim; i++) {
			if (diaphragmaMotion <= mot[i][0]) {
				index1 = i;
				break;
			}
		}
		//if diaphragmPosition was larger than the maximum in the measurements, floor it to the maximum
		if (index1 == 0) {
			double i1 = 0, i2 = 0;

			for (int j = 0; j < xDim; j++) {
				i2 = mot[j][0];
				if (i2 <= i1) {
					index1 = j;
					diaphragmaMotion = mot[j][0];
					break;
				}
				i1 = i2;
			}
		}
		diaMotFloor = mot[index1 - 1][0];
		diaMotCeil = mot[index1][0];
		int index2 = 0;

		for (int i = 0; i < yDim; i++) {
			if (positionZ <= pos[index1 - 1][i]) {
				index2 = i;
				break;
			}
		}
		if (index2 == 0) {
			index2 = yDim - 1;
			positionZ = pos[index1 - 1][index2];
		}
		motFloor1 = mot[index1 - 1][index2 - 1];
		motCeil1 = mot[index1 - 1][index2];
		posFloor1 = pos[index1 - 1][index2 - 1];
		posCeil1 = pos[index1 - 1][index2];
		int index3 = 0;
		for (int i = 0; i < yDim; i++) {
			if (positionZ <= pos[index1][i]) {
				index3 = i;
				break;
			}
		}
		if (index3 == 0) {
			index3 = yDim - 1;
			positionZ = pos[index1][index3];
		}
		motFloor2 = mot[index1][index3 - 1];
		motCeil2 = mot[index1][index3];
		posFloor2 = pos[index1][index3 - 1];
		posCeil2 = pos[index1][index3];

		//bilinear interpolation
		inter.setXPoints(posFloor1, posCeil1);
		inter.setYPoints(motFloor1, motCeil1);
		double interp1;
		if ((posCeil1 - posFloor1) != 0) {
			interp1 = inter.InterpolateYValue(positionZ);
		} else {
			interp1 = (motFloor1 + motCeil1) / 2;
		}
		inter.setXPoints(posFloor2, posCeil2);
		inter.setYPoints(motFloor2, motCeil2);
		double interp2;
		if ((posCeil2 - posFloor2) != 0) {
			interp2 = inter.InterpolateYValue(positionZ);
		} else {
			interp2 = (motFloor2 + motCeil2) / 2;
		}

		inter.setXPoints(diaMotFloor, diaMotCeil);
		inter.setYPoints(interp1, interp2);
		if ((diaMotCeil - diaMotFloor) != 0) {
			return inter.InterpolateYValue(diaphragmaMotion);
		} else {
			return (interp1 + interp2) / 2;
		}

	}

	public double getInterpolatedGlobalCompensationLinearScaling(double diaMot) {

		double mot[] = getBilinearInterpolationDataMotion(diaMot);
		double pos[] = getBilinearInterpolationDataPosition(diaMot);
		double m = getRANSACfittedFunction(pos, mot).getParametersAsDoubleArray()[0];

		return m;
	}

	public static void main(String args[]) throws Exception {
		MotionFieldReader motion = new MotionFieldReader();
		for (int i = 0; i < 11; i++) {
			Function f = motion.getRANSACfittedFunction(pos[i], mot[i]);
			VisualizationUtil.createScatterPlot(pos[i], mot[i], f).show();
			System.out.println(f.getParametersAsDoubleArray()[0] + " " + f.getParametersAsDoubleArray()[1]);
		}
		System.out.println(motion.getGlobalCompensationLinearScaling());
		System.out.println(motion.getGlobalCompensationLinearScalingSimple());
		//	double r = motion.bilinearInterpolation(50, 176);
		//	System.out.println(r);
	}
}
