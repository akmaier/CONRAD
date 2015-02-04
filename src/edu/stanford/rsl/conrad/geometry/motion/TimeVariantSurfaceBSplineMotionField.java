package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

/**
 * ParzenWindowMotionField that uses the points from a TimeVarianSurfaceBSpline to create the motion information.
 * @author akmaier
 *
 */
public class TimeVariantSurfaceBSplineMotionField extends
ParzenWindowMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 138799495833604748L;
	private TimeVariantSurfaceBSpline variant = null;

	
	/**
	 * Contructor
	 * @param inputSurface the input surface
	 * @param sigma the sigma [mm]
	 */
	public TimeVariantSurfaceBSplineMotionField(TimeVariantSurfaceBSpline inputSurface, double sigma) {
		super(sigma);
		variant = inputSurface;
	}

	PointND [] getRasterPoints (double time){
		PointND[] result = timePointMap.get(time);
		if (result == null){
			result = variant.getRasterPoints(TessellationUtil.getSamplingU(variant), TessellationUtil.getSamplingV(variant), time);
			timePointMap.put(time, result);
		}
		return result;
	}


	/**
	 * @return the variant
	 */
	public TimeVariantSurfaceBSpline getVariant() {
		return variant;
	}


	/**
	 * @param variant the variant to set
	 */
	public void setVariant(TimeVariantSurfaceBSpline variant) {
		this.variant = variant;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/