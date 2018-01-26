/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.Instances;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * best patchSize and imagelevels for the different classifeirs: patchSize
 * always 5 (121 pixel) Linear: 16 imagelevels Bagging: 16 imagelevels REPTree:
 * 16 imagelevels MLP: 16 imagelevels
 * 
 */

public class GLCMFeatureExtractor extends GridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6689796010159048008L;
	int patchSize = 2;
	int imagelevels = 16;
	int GLCMnumFeatures = 9;

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


	@Override
	public void configure() throws Exception {

		patchSize = UserUtil.queryInt("Set GLCM patch size", patchSize);
		imagelevels = UserUtil.queryInt(
				"Set number of different grey levels", imagelevels);
		configured = true;
		String[] nameString = { "angular second moment", "contrast",
				"correlation", "variance", "linear difference moment",
				"entropy", "energy", "homogenity", "inertia" };

		if (this.dataGrid instanceof MultiChannelGrid2D) {
			MultiChannelGrid2D multiChannelGrid = (MultiChannelGrid2D) this.dataGrid;
			attribs = new ArrayList<Attribute>(GLCMnumFeatures
					* multiChannelGrid.getNumberOfChannels() + 1);

			for (int i = 0; i < multiChannelGrid.getNumberOfChannels(); i++) {
				for (int j = 0; j < GLCMnumFeatures; j++) {
					String name = "Slice " + i + ": GLCM " + nameString[j];
					attribs.add(new weka.core.Attribute(name));
				}
			}
		} else {
			attribs = new ArrayList<Attribute>(GLCMnumFeatures + 1);
			for (int j = 0; j < GLCMnumFeatures; j++) {
				String name = "Slice " + j + ": GLCM " + nameString[j];
				attribs.add(new weka.core.Attribute(name));
			}
		}

		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);
		// leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);


	}

	/*
	 * computes a quantized matrix from the float image matrix
	 */
	public static int[][] computeLeveled(float[][] matrix, int NumOfLevels,
			float min, float max) {

		int[][] _leveled = new int[matrix.length][matrix[0].length];

		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {

				float quant = (max - min) / NumOfLevels;
				int level = (int) (Math.floor((matrix[i][j] - min) / quant));
				if (level < 0) {
					level = 0;
				} else if (level >= NumOfLevels) {
					level = NumOfLevels - 1;
				}
				_leveled[i][j] = level;
			}
		}
		return _leveled;
	}

	/*
	 * find minimum of 2d array
	 */
	public float min(float[][] matrix) {
		float min = matrix[0][0];
		for (int col = 0; col < matrix.length; col++) {
			for (int row = 0; row < matrix[col].length; row++) {
				if (min > matrix[col][row]) {
					min = matrix[col][row];
				}
			}
		}
		return min;
	}

	/*
	 * find maxium of 2d array
	 */
	public float max(float[][] matrix) {
		float max = matrix[0][0];
		for (int col = 0; col < matrix.length; col++) {
			for (int row = 0; row < matrix[col].length; row++) {
				if (max < matrix[col][row]) {
					max = matrix[col][row];
				}
			}
		}
		return max;
	}

	/*
	 * create Grey Level Co-occurrence Matrix with given angle phi
	 */
	public double[][] createGLCM(int[][] leveledMatrix, int phi) {

		int value = 0;
		int dvalue = 0;
		int pixelcount = 0;
		int offsetX = 1;
		int offsetY = 0;
		double[][] GLCM = new double[imagelevels][imagelevels];

		// set offsets based on the selected angle
		if (phi == 0) {
			offsetX = 1;
			offsetY = 0;
		} else if (phi == 45) {
			offsetX = 1;
			offsetY = -1;
		} else if (phi == 90) {
			offsetX = 0;
			offsetY = -1;
		} else if (phi == 135) {
			offsetX = -1;
			offsetY = -1;
		}

		// loop through the pixels in the ROI bounding rectangle
		for (int j = 0; j < leveledMatrix.length; j++) {
			for (int i = 0; i < leveledMatrix[0].length; i++) {
				int dy = j + offsetY;
				int dx = i + offsetX;
				if (((dx >= 0) && (dx < leveledMatrix[0].length))
						&& ((dy >= 0) && (dy < (leveledMatrix.length)))) {
					value = leveledMatrix[i][j];
					dvalue = leveledMatrix[dx][dy];
					GLCM[value][dvalue]++;
					pixelcount++;
				}
			}
		}

		// convert the GLCM from absolute counts to probabilities
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				GLCM[i][j] = (GLCM[i][j]) / (pixelcount);
			}
		}
		return GLCM;
	}

	@Override
	public double[] extractFeatureAtIndex(int x, int y) {

		int lengthX = 0;
		int lengthY = 0;
		double[] result;

		if (x < patchSize && y > (this.dataGrid.getSize()[1] - patchSize)) {
			lengthX = (x + patchSize) - x;
			lengthY = y - (y - patchSize);
		}

		if (y < patchSize && x > (this.dataGrid.getSize()[0] - patchSize)) {
			lengthY = (y + patchSize) - y;
			lengthX = x - (x - patchSize);
		}

		if (y > (this.dataGrid.getSize()[1] - patchSize)
				&& x > (this.dataGrid.getSize()[0] - patchSize)) {
			lengthY = y - (y - patchSize);
			lengthX = x - (x - patchSize);
		}

		if (x < patchSize && y < patchSize) {
			lengthX = x - (x - patchSize);
			lengthY = (y + patchSize) - y;
		}

		lengthY = (y + patchSize) - (y - patchSize);
		lengthX = (x + patchSize) - (x - patchSize);
		float[][] pixels = new float[lengthX][lengthY];

		if (this.dataGrid instanceof MultiChannelGrid2D) {
			MultiChannelGrid2D multiChannelGrid = (MultiChannelGrid2D) this.dataGrid;
			result = new double[(GLCMnumFeatures
					* multiChannelGrid.getNumberOfChannels() + 1)];

			for (int c = 0; c < multiChannelGrid.getNumberOfChannels(); c++) {

				// writes pixelvalues in a 2d array
				if ((x >= 10) && (y >= 10)
						&& (x < (dataGrid.getWidth() - patchSize))
						&& (y < (dataGrid.getHeight() - patchSize))) {
					for (int j = y - patchSize; j < y + patchSize; j++) {
						if (j < 0)
							continue;
						if (j >= this.dataGrid.getSize()[1])
							continue;
						for (int i = x - patchSize; i < x + patchSize; i++) {
							pixels[i - (x - patchSize)][j - (y - patchSize)] = multiChannelGrid
									.getChannel(c).getAtIndex(i, j);
						}
					}
				}

				// writes pixelvalues in a 2d array x < and y >
				if (x < patchSize
						&& y > (this.dataGrid.getSize()[1] - patchSize)) {
					for (int j = y - patchSize; j < y; j++) {
						if (j < 0)
							continue;
						if (j >= this.dataGrid.getSize()[1])
							continue;
						for (int i = x; i < x + patchSize; i++) {
							pixels[i - x][j - (y - patchSize)] = multiChannelGrid
									.getChannel(c).getAtIndex(i, j);
							// dataGrid.getAtIndex(i,
							// j);
						}
					}
				}

				// writes pixelvalues in a 2d array x > and y <
				if (y < patchSize
						&& x > (this.dataGrid.getSize()[0] - patchSize)) {
					for (int j = y; j < y; j++) {
						if (j < 0)
							continue;
						if (j >= this.dataGrid.getSize()[1])
							continue;
						for (int i = x - patchSize; i < x; i++) {
							pixels[i - (x - patchSize)][j - y] = multiChannelGrid
									.getChannel(c).getAtIndex(i, j);
						}
					}
				}

				// writes pixelvalues in a 2d array x > and y >
				if (y > (this.dataGrid.getSize()[1] - patchSize)
						&& x > (this.dataGrid.getSize()[0] - patchSize)) {
					for (int j = y - patchSize; j < y; j++) {
						if (j < 0)
							continue;
						if (j >= this.dataGrid.getSize()[1])
							continue;
						for (int i = x - patchSize; i < x; i++) {
							pixels[i - (x - patchSize)][j - (y - patchSize)] = multiChannelGrid
									.getChannel(c).getAtIndex(i, j);
						}
					}
				}

				// writes pixelvalues in a 2d array x < and y <
				if (x < patchSize && y < patchSize) {
					for (int j = y; j < y + patchSize; j++) {
						if (j < 0)
							continue;
						if (j >= this.dataGrid.getSize()[1])
							continue;
						for (int i = x; i < x + patchSize; i++) {
							pixels[i - x][j - y] = multiChannelGrid.getChannel(
									c).getAtIndex(i, j);
						}
					}
				}

				// get min and max pixelvalue and compute leveled matrix
				float max = max(pixels);
				float min = min(pixels);
				int[][] leveledMatrix = computeLeveled(pixels, imagelevels,
						min, max);

				// feature array which will be returned
				double[] singleVector = new double[10];

				// get GLCM for all four angels
				double[][] GLCM_0 = createGLCM(leveledMatrix, 0);
				double[][] GLCM_45 = createGLCM(leveledMatrix, 45);
				double[][] GLCM_90 = createGLCM(leveledMatrix, 90);
				double[][] GLCM_135 = createGLCM(leveledMatrix, 135);

				// extract GLCm features for every angle
				double[] features_0 = extractGLCMFeatures(GLCM_0);
				double[] features_45 = extractGLCMFeatures(GLCM_45);
				double[] features_90 = extractGLCMFeatures(GLCM_90);
				double[] features_135 = extractGLCMFeatures(GLCM_135);

				// take the average of the features
				for (int i = 0; i < singleVector.length - 1; i++) {
					singleVector[i] = (features_0[i] + features_45[i]
							+ features_90[i] + features_135[i]) / 4;
				}

				for (int k = 0; k < GLCMnumFeatures; k++) {
					result[(c * GLCMnumFeatures) + k] = singleVector[k];
				}
			}

		} else {
			result = new double[(GLCMnumFeatures + 1)];

			// writes pixelvalues in a 2d array
			if ((x >= patchSize) && (y >= patchSize)
					&& (x < (dataGrid.getWidth() - patchSize))
					&& (y < (dataGrid.getHeight() - patchSize))) {
				for (int j = y - patchSize; j < y + patchSize; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x - patchSize; i < x + patchSize; i++) {
						pixels[i - (x - patchSize)][j - (y - patchSize)] = dataGrid
								.getAtIndex(i, j);
					}
				}
			}

			// writes pixelvalues in a 2d array x < and y >
			if (x < patchSize && y > (this.dataGrid.getSize()[1] - patchSize)) {
				for (int j = y - patchSize; j < y; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x; i < x + patchSize; i++) {
						pixels[i - x][j - (y - patchSize)] = dataGrid
								.getAtIndex(i, j);

					}
				}
			}

			// writes pixelvalues in a 2d array x > and y <
			if (y < patchSize && x > (this.dataGrid.getSize()[0] - patchSize)) {
				for (int j = y; j < y; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x - patchSize; i < x; i++) {
						pixels[i - (x - patchSize)][j - y] = dataGrid
								.getAtIndex(i, j);
					}
				}
			}

			// writes pixelvalues in a 2d array x > and y >
			if (y > (this.dataGrid.getSize()[1] - patchSize)
					&& x > (this.dataGrid.getSize()[0] - patchSize)) {
				for (int j = y - patchSize; j < y; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x - patchSize; i < x; i++) {
						pixels[i - (x - patchSize)][j - (y - patchSize)] = dataGrid
								.getAtIndex(i, j);
					}
				}
			}

			// writes pixelvalues in a 2d array x < and y <
			if (x < patchSize && y < patchSize) {
				for (int j = y; j < y + patchSize; j++) {
					if (j < 0)
						continue;
					if (j >= this.dataGrid.getSize()[1])
						continue;
					for (int i = x; i < x + patchSize; i++) {
						pixels[i - x][j - y] = dataGrid.getAtIndex(i, j);
					}
				}
			}

			// get min and max pixelvalue and compute leveled matrix
			float max = max(pixels);
			float min = min(pixels);
			int[][] leveledMatrix = computeLeveled(pixels, imagelevels, min,
					max);

			// feature array which will be returned
			// double[] singleVector = new double[10];

			// get GLCM for all four angels
			double[][] GLCM_0 = createGLCM(leveledMatrix, 0);
			double[][] GLCM_45 = createGLCM(leveledMatrix, 45);
			double[][] GLCM_90 = createGLCM(leveledMatrix, 90);
			double[][] GLCM_135 = createGLCM(leveledMatrix, 135);

			// extract GLCm features for every angle
			double[] features_0 = extractGLCMFeatures(GLCM_0);
			double[] features_45 = extractGLCMFeatures(GLCM_45);
			double[] features_90 = extractGLCMFeatures(GLCM_90);
			double[] features_135 = extractGLCMFeatures(GLCM_135);

			// take the average of the features
			for (int i = 0; i < result.length - 1; i++) {
				result[i] = (features_0[i] + features_45[i] + features_90[i] + features_135[i]) / 4;
			}
		}
		result[result.length - 1] = labelGrid.getAtIndex(x, y);
		return result;
	}

	// compute grey level co-ocurrence matrix features:
	// contrast, correlation, angular second moment, homogenity, entropy,
	// inverse difference moment, variance, inertia
	public double[] extractGLCMFeatures(double[][] GLCM) {

		double[] feature = new double[9];

		// calculate meanX, meanY, stdX and stdY for the GLCM
		double[] px = new double[imagelevels];
		double[] py = new double[imagelevels];
		double meanX = 0.0;
		double meanY = 0.0;
		double stdX = 0.0;
		double stdY = 0.0;

		// Px(i) and Py(j) are the marginal-probability matrix; sum rows (px) or
		// columns (py)
		// First, initialize the arrays to 0
		for (int i = 0; i < imagelevels; i++) {
			px[i] = 0.0;
			py[i] = 0.0;
		}

		// sum the GLCM rows to Px(i)
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				px[i] += GLCM[i][j];
			}
		}

		// sum the GLCM rows to Py(j)
		for (int j = 0; j < imagelevels; j++) {
			for (int i = 0; i < imagelevels; i++) {
				py[j] += GLCM[i][j];
			}
		}

		// calculate meanx and meany
		for (int i = 0; i < imagelevels; i++) {
			meanX += (i * px[i]);
			meanY += (i * py[i]);
		}

		// calculate stdevx and stdevy
		for (int i = 0; i < imagelevels; i++) {
			stdX += ((Math.pow((i - meanX), 2)) * px[i]);
			stdY += ((Math.pow((i - meanY), 2)) * py[i]);
		}

		// calculate the angular second moment (asm)

		double asm = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				asm += (GLCM[i][j] * GLCM[i][j]);
			}
		}
		feature[0] = asm;

		// calculate the contrast (Haralick, et al. 1973)
		double contrast = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				contrast += Math.pow(Math.abs(i - j), 2) * (GLCM[i][j]);
			}
		}
		feature[1] = contrast;

		// calculate the correlation (Haralick et al., 1973)
		double corr = 0.0;
		for (int x = 0; x < imagelevels; x++) {
			for (int y = 0; y < imagelevels; y++) {
				double num = (x - meanX) * (y - meanY) * GLCM[x][y];
				double denum = stdX * stdY;
				corr += num / denum;
			}
		}
		feature[2] = corr;

		// calculate the variance ("variance" in Walker 1995;
		// "Sum of Squares: Variance" in Haralick 1973)
		double variance = 0.0;
		double mean = 0.0;

		mean = (meanX + meanY) / 2;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				variance += (Math.pow((i - mean), 2) * GLCM[i][j]);
			}
		}
		feature[3] = variance;

		// calculate the inverse difference moment (idm) (Walker, et al. 1995)
		double idm = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				idm += (GLCM[i][j] / (1 + (Math.pow(i - j, 2))));
			}
		}
		feature[4] = idm;

		// calculate the entropy (Haralick et al., 1973; Walker, et al., 1995)
		double entropy = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				if (GLCM[i][j] != 0) {
					entropy -= (GLCM[i][j] * (Math.log(GLCM[i][j])));
				}
			}
		}
		feature[5] = entropy;

		// calculate the energy
		double energy = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				energy += Math.pow(GLCM[i][j], 2);
			}
		}
		feature[6] = energy;

		// calculate the homogenity (Parker)
		double homogeneity = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				homogeneity += GLCM[i][j] / (1.0 + Math.abs(i - j));
			}
		}
		feature[7] = homogeneity;

		// calculate the inertia (Walker, et al., 1995; Connors, et al. 1984)
		double inertia = 0.0;
		for (int i = 0; i < imagelevels; i++) {
			for (int j = 0; j < imagelevels; j++) {
				if (GLCM[i][j] != 0) {
					inertia += (Math.pow((i - j), 2) * GLCM[i][j]);
				}
			}
		}
		feature[8] = inertia;

		return feature;
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
	 * @return the imagelevels
	 */
	public int getImagelevels() {
		return imagelevels;
	}

	/**
	 * @param imagelevels
	 *            the imagelevels to set
	 */
	public void setImagelevels(int imagelevels) {
		this.imagelevels = imagelevels;
	}

	@Override
	public String getName() {
		return "Patch-based Gray Level Cooccurence Matrix Features";
	}

}
