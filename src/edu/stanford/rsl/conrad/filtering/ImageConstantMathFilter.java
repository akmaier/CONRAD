package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.process.FloatProcessor;

public class ImageConstantMathFilter extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1576269350353505544L;
	private double operand = 0;
	/**
	 * @return the operand
	 */
	public double getOperand() {
		return operand;
	}

	/**
	 * @param operand the operand to set
	 */
	public void setOperand(double operand) {
		this.operand = operand;
	}

	/**
	 * @param deviceDependent the deviceDependent to set
	 */
	public void setDeviceDependent(boolean deviceDependent) {
		this.deviceDependent = deviceDependent;
	}

	private String operation = null;
	private boolean deviceDependent = false;
	public static final String ADD = " add ";
	public static final String SUBTRACT = " subtract ";
	public static final String MULTIPLY = " multiply ";
	public static final String DIVIDE = " divide ";
	public static final String LOGARITHM = " logarithm ";
	public static final String SQUARE = " square ";
	public static final String SQUAREROOT = " squareroot ";

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
	throws Exception {
		FloatProcessor revan = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		revan.setPixels(imageProcessor.getBuffer());
		if (operation.equals(MULTIPLY)) {
			revan.multiply(operand);
		}
		if (operation.equals(DIVIDE)) {
			revan.multiply(1.0 / operand);
		}
		if (operation.equals(SUBTRACT)) {
			revan.add(-operand);
		}
		if (operation.equals(ADD)) {
			revan.add(operand);
		}
		if (operation.equals(LOGARITHM)) {
			revan.log();
		}
		if (operation.equals(SQUARE)) {
			revan.sqr();
		}
		if (operation.equals(SQUAREROOT)) {
			revan.sqrt();
		}
		return imageProcessor;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		ImageConstantMathFilter clone = new ImageConstantMathFilter();
		clone.operand = operand;
		clone.operation = operation;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		if ((operation != null) ) {
			return "Image Constant Math Filter" + operation + " " + operand;
		} else {
			return "Image Constant Math Filter";
		}
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	public String getOperation() {
		return operation;
	}

	public void setOperation(String operation) {
		this.operation = operation;
	}

	@Override
	public void configure() throws Exception {
		String [] operations = {ADD, SUBTRACT, MULTIPLY, DIVIDE, LOGARITHM, SQUARE, SQUAREROOT};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		operand = UserUtil.queryDouble("Enter operand value", operand);
		if ((operation != null)){
			deviceDependent = UserUtil.queryBoolean("Does this filter model device / hardware dependent behaviour?");
		}
		configured=true;
	}
	
	/**
	 * The use may differ. Hence device dependency can be set. Is set during configuration.
	 */
	@Override
	public boolean isDeviceDependent() {
		return deviceDependent;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
