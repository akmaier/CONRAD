/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.util.ArrayList;

import org.apache.commons.math3.util.FastMath;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.jpop.utils.UserUtil;
import weka.core.Attribute;
import weka.core.Instances;

public class HistogramFeatureExtractor extends GridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8201678340342264383L;
	int numberOfBins = 64;
	int patchSize = 2;	//25 Pixel neighborhood
	// Atmung 0-14484,	Rotation 0-502199
	float min = 0;
	float max = 14484;
	int HistoFeatures = 6;

	public Instances getInstances() {
		return instances;
	}

	protected Attribute generateClassAttribute() {
		Attribute classAttribute = null;
		if (classValues == null) {
			classAttribute = new Attribute("Class");
		} else {
			classAttribute = new Attribute(className, classValues);
		}
		classAttribute = new Attribute(className);
		return classAttribute;
	}

	// compute histogram based features: mean grey value, standard deviation,
	// entropy; coefficient of variance, skewness and kurtosis
	public double[] extractHistogramFeatures(double[] hist) {

		double histSum = 0.0;
		double AveGreyVal, Variance, StandardDev, Entropy = 0.0;
		double tmp;
		double accum = 0.0;
		double accum2 = 0.0;
		double accum3 = 0.0;
		double accum4 = 0.0;
		double len = hist.length;
		double[] feature = new double[7];

		// calculate mean grey value of histogram
		for (int i = 0; i < len; i++) {
			histSum += hist[i];
		}
		AveGreyVal = histSum / len;
		// write mean grey value in feature array
		feature[0] = AveGreyVal;

		// Normalize histogram, that is makes the sum of all bins equal to 1.
		double[] normalizedHist = new double[hist.length];
		for (int i = 0; i < hist.length; i++) {
			normalizedHist[i] = hist[i] / histSum;
		}

		// calculate variance of histogram
		for (int i = 0; i < len; i++) {
			tmp = normalizedHist[i] - AveGreyVal;
			accum += tmp * tmp;
			accum2 += tmp;
		}
		Variance = (accum - (accum2 * accum2 / len)) / (len - 1);
		StandardDev = Math.sqrt(Variance);
		// write standard deviation in feature array
		feature[1] = StandardDev;

		// calculate entropy of histogram
		for (int i = 0; i < len; i++) {
			if (normalizedHist[i] > 0) {
				tmp = normalizedHist[i] / histSum;
				Entropy -= tmp * (Math.log(tmp) / Math.log(2));
			}
		}
		feature[2] = Entropy;

		// calculate coefficient of variance of histogram and write it in
		// feature array
		feature[3] = (StandardDev / AveGreyVal) * 100.0;

		// calculate skewness of histogram
		for (int i = 0; i < len; i++) {
			accum3 += FastMath.pow((normalizedHist[i] - AveGreyVal), 3.0);

		}
		accum3 /= Variance * Math.sqrt(Variance);
		// write skewness in feature array
		feature[4] = (len / ((len - 1.0) * (len - 2))) * accum3;

		// calculate kurtosis of histogram
		for (int i = 0; i < len; i++) {
			accum4 += FastMath.pow((normalizedHist[i] - AveGreyVal), 4.0);
		}
		accum4 /= FastMath.pow(StandardDev, 4.0);

		double coeffs = (len * (len + 1)) / ((len - 1) * (len - 2) * (len - 3));
		double termTwo = (3 * FastMath.pow((len - 1), 2.0))
				/ ((len - 2) * (len - 3));
		// write kurtosis in feature array
		feature[5] = (coeffs * accum4) - termTwo;

		return feature;
	}

	@Override
	public void configure() throws Exception {


		patchSize = UserUtil
				.queryInt("Set histogram patch size", patchSize);
		numberOfBins = UserUtil.queryInt("Select number of histogram bins",
				numberOfBins);
		min = (float) UserUtil.queryDouble("Minimal histogram value", min);
		max = (float) UserUtil.queryDouble("Maximal histogram value ", max);
		configured = true;

		String[] nameString = { "mean grey value", "standard deviation",
				"entropy", "coefficient of variance", "skewness", "kurtosis" };

		if (this.dataGrid instanceof MultiChannelGrid2D) {
			MultiChannelGrid2D multiChannelGrid = (MultiChannelGrid2D) this.dataGrid;

			attribs = new ArrayList<Attribute>(HistoFeatures
					* multiChannelGrid.getNumberOfChannels() + 1);

			for (int i = 0; i < multiChannelGrid.getNumberOfChannels(); i++) {
				for (int j = 0; j < HistoFeatures; j++) {
					String name = "Slice " + i + ": Histogram " + nameString[j];
					attribs.add(new weka.core.Attribute(name));
				}
			}

		} else {

			attribs = new ArrayList<Attribute>(HistoFeatures + 1);
			for (int j = 0; j < HistoFeatures; j++) {
				String name = ": Histogram " + nameString[j];
				attribs.add(new weka.core.Attribute(name));
			}

		}
		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);
		// leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);

	}

	@Override
	public double[] extractFeatureAtIndex(int x, int y) {

		double[] result;

		if (this.dataGrid instanceof MultiChannelGrid2D) {
			MultiChannelGrid2D multiChannelGrid = (MultiChannelGrid2D) this.dataGrid;
			result = new double[(HistoFeatures
					* multiChannelGrid.getNumberOfChannels() + 1)];

			for (int c = 0; c < multiChannelGrid.getNumberOfChannels(); c++) {

				double[] histo = new double[numberOfBins + 1];

				for (int j = y - patchSize; j <= y + patchSize; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x - patchSize; i <= x + patchSize; i++) {
						if (i < 0)
							continue;
						if (i >= this.dataGrid.getSize()[0])
							continue;
						float cellLength = (max - min) / (numberOfBins);
						// float value = dataGrid.getAtIndex(i, j);
						double value = multiChannelGrid.getChannel(c)
								.getAtIndex(i, j);
						int coord = (int) Math
								.floor((value - min) / cellLength);
						if (coord > numberOfBins) {
							coord = numberOfBins;
						}
						if (coord < 0)
							coord = 0;
						histo[coord]++;
					}
				}

				// histo[numberOfBins] = labelGrid.getAtIndex(x, y);
				double[] singleVector = extractHistogramFeatures(histo);

				for (int k = 0; k < HistoFeatures; k++) {
					result[(c * HistoFeatures) + k] = singleVector[k];
				}
			}
		} else {

			result = new double[(HistoFeatures + 1)];
			double[] histo = new double[numberOfBins + 1];

			for (int j = y - patchSize; j <= y + patchSize; j++) {
				if (j < 0)
					continue;
				if (j >= this.dataGrid.getSize()[1])
					continue;
				for (int i = x - patchSize; i <= x + patchSize; i++) {
					if (i < 0)
						continue;
					if (i >= this.dataGrid.getSize()[0])
						continue;
					float cellLength = (max - min) / (numberOfBins);
					float value = dataGrid.getAtIndex(i, j);
					int coord = (int) Math.floor((value - min) / cellLength);
					if (coord > numberOfBins) {
						coord = numberOfBins;
					}
					if (coord < 0)
						coord = 0;
					histo[coord]++;
				}
			}
			result = extractHistogramFeatures(histo);

		}
		result[result.length - 1] = labelGrid.getAtIndex(x, y);
		return result;
	}

	/**
	 * @return the numberOfBins
	 */
	public int getNumberOfBins() {
		return numberOfBins;
	}

	/**
	 * @param numberOfBins
	 *            the numberOfBins to set
	 */
	public void setNumberOfBins(int numberOfBins) {
		this.numberOfBins = numberOfBins;
	}

	/**
	 * @return the patchSize
	 */
	public int getPatchSize() {
		return patchSize;
	}

	/**
	 * @param patchSize
	 *            the patchSize to set
	 */
	public void setPatchSize(int patchSize) {
		this.patchSize = patchSize;
	}

	/**
	 * @return the min
	 */
	public float getMin() {
		return min;
	}

	/**
	 * @param min
	 *            the min to set
	 */
	public void setMin(float min) {
		this.min = min;
	}

	/**
	 * @return the max
	 */
	public float getMax() {
		return max;
	}

	/**
	 * @param max
	 *            the max to set
	 */
	public void setMax(float max) {
		this.max = max;
	}

	@Override
	public String getName() {
		return "Patch-based Histogram Features";
	}

}
