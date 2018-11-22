package edu.stanford.rsl.conrad.angio.preprocessing.background;


import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Locale;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public abstract class Inpainting{

	protected boolean[][][] mask = null;

	protected Number[] parameters = null;

	protected String name = "Inpainting";

	protected String[] parameterNames = null;

	public Inpainting() {
		mask = null;
	}

	public Inpainting(boolean[][][] mask) {
		this.mask = mask;
	}

	public void setParameters(Number... parameters){
		this.parameters = parameters;
	}

	public void setParameters(double... parameters){
		this.parameters = new Number[parameters.length];
		for (int i = 0; i < parameters.length; i++) {
			this.parameters[i] = new Double(parameters[i]);
		}
	}

	public Number[] getParameters(){
		return parameters;
	}

	public void setMask(boolean[][][] mask){
		this.mask = mask;
	}

	public boolean[][][] getMask(){
		return mask;
	}

	public abstract Grid3D applyToGrid(Grid3D input);

	public void configure(){
		this.setName();
	}

	public String getParameterString(){
		String out = "";
		DecimalFormat dfDouble = new DecimalFormat("00.0000");
		dfDouble.setRoundingMode(RoundingMode.HALF_UP);
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		dfDouble.setDecimalFormatSymbols(otherSymbols);

		if (parameterNames != null && parameters != null){
			for (int i = 0; i < parameterNames.length; i++) {
				out += parameterNames[i];
				out += "_";
				out += dfDouble.format(parameters[i]);
				if (i < parameterNames.length-1)
					out += "___";
			}
		}
		return out;
	}

	public abstract void setName();

	public abstract String[] getNames();

	public void blankMarkersAndCreateMask(Grid3D image, ArrayList<ArrayList<double[]>> twoDPosReal, double radius, boolean blankOut){
		if (image!=null && mask == null){
			mask = new boolean[image.getSize()[2]][image.getSize()[1]][image.getSize()[0]];
			//remove markers
			for (int i = 0; i < twoDPosReal.size(); i++) {
				for (int j = 0; j < twoDPosReal.get(i).size(); j++) {
					double uv[] = twoDPosReal.get(i).get(j);
					if ((int)uv[2] >= image.getSize()[2])
						continue;
					int blankRadius = (int)Math.ceil(radius);
					for (int u = (int)Math.floor(uv[0])-blankRadius; u < (int)Math.ceil(uv[0])+blankRadius; ++u){
						for (int v = (int)Math.floor(uv[1])-blankRadius; v < (int)Math.ceil(uv[1])+blankRadius; ++v){
							if (u >= 0 && v >= 0 && u < image.getSize()[0] && v < image.getSize()[1]){
								if ((new PointND(uv[0],uv[1])).euclideanDistance(new PointND(u,v)) < radius){
									if (blankOut)
										image.setAtIndex(u, v, (int)uv[2], 0f);
									mask[(int)uv[2]][v][u] = true;
								}
							}
						}
					}
				}
			}
		}
	}
	
	public Grid3D getMaskAsStack(){
		if (mask != null){
			int noSlices = mask.length;
			int height = mask[0].length;
			int width = mask[0][0].length;
			
			Grid3D out = new Grid3D(width,height,noSlices);
			for (int i = 0; i < mask.length; i++) {
				for (int j = 0; j < mask[i].length; j++) {
					for (int k = 0; k < mask[i][j].length; k++) {
						out.setAtIndex(k, j, i, mask[i][j][k] ? 0.f : 1.f);
					}
				}
			}
			return out;
		}
		return null;
	}

	@Override
	public String toString() {
		if (this.getNames() != null)
			return this.getNames()[0] +"___" + this.getParameterString();
		else
			return "Inpainting Method";
	}


}
