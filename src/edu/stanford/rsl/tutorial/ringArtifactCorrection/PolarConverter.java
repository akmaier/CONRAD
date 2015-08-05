package edu.stanford.rsl.tutorial.ringArtifactCorrection;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;

public class PolarConverter {
	int widthCartesian, heightCartesian, widthPolar, heightPolar;
	double centerX, centerY;
	
	
	public Grid2D convertToPolar(Grid2D cartesian) {
		this.setPolarGridSize(cartesian);
		Grid2D polar = new Grid2D(widthPolar, heightPolar);
		for (int x=0; x<widthPolar; x++) {
			for (int y=0; y<heightPolar; y++) {
				double radius = x;
				double theta = y/360.0 * Math.PI * 2;
				double realX = getXFromPolar(radius, theta) + centerX;
				double realY = getYFromPolar(radius, theta) + centerY;
				float value = InterpolationOperators.interpolateLinear(cartesian, realX, realY);
				polar.putPixelValue(x, y, value);
				
			}
		}
		return polar;
	}
	
	public Grid2D convertToCartesian(Grid2D polar) {
		this.setCartesianGridSize(polar);
		Grid2D cartesian = new Grid2D(widthCartesian, heightCartesian);
		for (int x=0; x<widthCartesian; x++) {
			for (int y=0; y<heightCartesian; y++) {
				double realX = x - centerX;
				double realY = y - centerY;
				double radius = this.getRadiusFromCartesian(realX, realY);
				double theta = this.getAngleFromCartesian(realX, realY);
				realX = radius;
				realY = theta * (heightPolar/360.0);
				float value = InterpolationOperators.interpolateLinear(polar, realX, realY);
				cartesian.putPixelValue(x, y, value);
			}
		}
		return cartesian;
	}

	private void setPolarGridSize(Grid2D cartesian) {
		widthCartesian = cartesian.getWidth();
		heightCartesian = cartesian.getHeight();
		centerX = widthCartesian / 2.0f;
		centerY = heightCartesian / 2.0f;
		heightPolar = 360; // use y-axis for angles
		widthPolar = widthCartesian;
	}
	
	private void setCartesianGridSize(Grid2D polar) {
		widthPolar = polar.getWidth();
		heightPolar = polar.getHeight();
		widthCartesian = widthPolar;
		heightCartesian = widthPolar;
		centerX = widthCartesian / 2.0f;
		centerY = heightCartesian / 2.0f;
	}
	
	private double getXFromPolar(double r, double theta) {
		return r*Math.cos(theta);
	}
	
	private double getYFromPolar(double r, double theta) {
		return r*Math.sin(theta);
	}
	
	private double getRadiusFromCartesian(double x, double y) {
		return Math.sqrt((x*x)+(y*y));
	}
	
	private double getAngleFromCartesian(double x, double y) {
		double angle = Math.toDegrees(Math.atan2(y, x));
		if (angle < 0) {
			angle += 359; //correction to give angle between 0 and 360 deg
		}
		return angle;
	}
	

}
/*
 * Copyright (C) 2010-2015 Florian Gabsteiger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
