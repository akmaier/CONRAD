/*
 * Copyright (C) 2016 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class BackgroundSeparationTool extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9047172880007368459L;
	/**
	 * 
	 */
	boolean subtract = true;
	private String operation = null;
	public static final String MIN = " min ";
	public static final String MAX = " max ";
	public static final String MEAN = " mean ";
	public static final String MEDIAN = " median ";

	@Override
	public void configure() throws Exception {
		subtract = UserUtil.queryBoolean("Subtract Static Layer (for online subtraction)?");
		String [] operations = {MIN, MAX, MEAN, MEDIAN};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		// Nothing to do here
	}

	@Override
	protected void cleanup() {
		int numberOfImages= inputQueue.size();
		Grid2D referenceImage = (Grid2D) new Grid2D(inputQueue.get(0));
		for (int j=0; j<referenceImage.getHeight(); j++){
			for (int i=0; i<referenceImage.getWidth(); i++){
				double [] values = new double[numberOfImages];
				for (int k = 0; k < numberOfImages; k++){
					values[k]=inputQueue.get(k).getAtIndex(i, j);
				}
				double val = 0;
				if (operation.equals(MIN)){
					val = DoubleArrayUtil.minOfArray(values);
				}
				if (operation.equals(MAX)){
					val = DoubleArrayUtil.maxOfArray(values);
				}
				if (operation.equals(MEAN)){
					val = DoubleArrayUtil.computeMean(values);
				}
				if (operation.equals(MEDIAN)){
					val = DoubleArrayUtil.computeMedian(values);
				}
				referenceImage.putPixelValue(i, j, val);
			}
		}
		if (subtract) {
			for (int i = 0; i < numberOfImages; i++){
				inputQueue.get(i).getGridOperator().subtractBy(inputQueue.get(i), referenceImage);;
				try {
					sink.process(inputQueue.get(i), i);
					inputQueue.remove(i);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		} else {
			try {
				sink.process(referenceImage, 0);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		super.cleanup();
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		String nameString = "Background Separation Tool";
		if (configured) nameString += " (subtract " + operation + subtract +")";
		return nameString;
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

}
