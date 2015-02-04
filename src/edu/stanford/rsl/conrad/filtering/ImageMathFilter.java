package edu.stanford.rsl.conrad.filtering;

import java.io.File;

import edu.stanford.rsl.apps.gui.RawDataOpener;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.ImagePlus;
import ij.io.Opener;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class ImageMathFilter extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1161045076404535185L;
	private String operandFileName = null;
	private ImagePlus operand;
	private String operation = null;
	private boolean deviceDependent = false;
	public static final String ADD = " add ";
	public static final String SUBTRACT = " subtract ";
	public static final String MULTIPLY = " multiply ";
	public static final String DIVIDE = " divide ";

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		operand = null;
		configured = false;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D image)
	throws Exception {
		FloatProcessor imageProcessor = new FloatProcessor(image.getWidth(), image.getHeight());
		imageProcessor.setPixels(image.getBuffer());
		ImageProcessor currentOperand = null;
		if(operand == null){
			open();
		}
		if (operand.getStackSize() > 1) {
			currentOperand =  operand.getStack().getProcessor(imageIndex + 1);
		} else {
			currentOperand = operand.getChannelProcessor();
		}
		ImageProcessor revan = null;
		if (operation.equals(MULTIPLY)) {
			revan = ImageUtil.multiplyImages(imageProcessor, currentOperand);
		}
		if (operation.equals(DIVIDE)) {
			revan = ImageUtil.divideImages(imageProcessor, currentOperand);
		}
		if (operation.equals(SUBTRACT)) {
			currentOperand = currentOperand.duplicate();
			currentOperand.multiply(-1);
			ImageUtil.addProcessors(currentOperand, imageProcessor);
			revan = currentOperand;
		}
		if (operation.equals(ADD)) {
			currentOperand = currentOperand.duplicate();
			ImageUtil.addProcessors(currentOperand, imageProcessor);
			revan = currentOperand;
		}
		Grid2D result = new Grid2D((float[])revan.getPixels(), image.getWidth(), image.getHeight());
		return result;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		ImageMathFilter clone = new ImageMathFilter();
		clone.operand = operand;
		clone.operation = operation;
		clone.configured = configured;
		clone.operandFileName = operandFileName;
		return clone;
	}

	@Override
	public String getToolName() {
		if ((operation != null) && (operandFileName != null)) {
			return "Image Math Filter" + operation + operandFileName;
		} else {
			return "Image Math Filter";
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
		operandFileName = UserUtil.queryString("Enter name of Image: ", operandFileName);
		String [] operations = {ADD, SUBTRACT, MULTIPLY, DIVIDE};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		if ((operation != null) && (operandFileName != null)){
			deviceDependent = UserUtil.queryBoolean("Does this filter model device / hardware dependent behaviour?");
			open();
		}
	}

	private synchronized void open(){
		Opener opener = new Opener();
		operand = opener.openImage(operandFileName);
		if (operand == null){
			RawDataOpener raw = RawDataOpener.getRawDataOpener();
			operand = raw.openImage(new File(operandFileName), raw.getFileInfo());
		}
		configured = true;
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
