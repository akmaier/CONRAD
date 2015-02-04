package edu.stanford.rsl.conrad.geometry.splines;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Implementation after Ruijters, Romeny & Suetens.
 * "Efficient GPU-Based Textutre Interpolation using Uniform B-Splines". 2009
 * 
 * @author akmaier; modifications for closed cubic b-spline curve by jrausch
 *
 */
public class ClosedCubicBSpline extends BSpline {

	public static double oneOverSix = 1.0 / 6.0;
	public static double twoOverThree = 2.0 / 3.0;

	/**
	 * 
	 * 
	 */
	private static final long serialVersionUID = 8467214013384105086L;

	public ClosedCubicBSpline(ArrayList<PointND> controlPoints,
			double[] uVector) {
		super(controlPoints, uVector);		
	}

	public ClosedCubicBSpline(ArrayList<PointND> controlPoints,
			SimpleVector knotVector) {
		super(controlPoints, knotVector);
	}

	public static double [] getWeights(double t){
		double [] revan = new double [4];
		double index = Math.floor(t);
		double alpha = t - index;
		double oneAlpha = 1.0 - alpha;
		double oneAlphaSquare = oneAlpha * oneAlpha;
		double alphaSquare = alpha * alpha;
		revan[0] = oneOverSix * oneAlphaSquare * oneAlpha;
		revan[1] = twoOverThree - 0.5 * alphaSquare *(2.0-alpha);
		revan[2] = twoOverThree - 0.5 * oneAlphaSquare * (2.0-oneAlpha);
		revan[3] = oneOverSix * alphaSquare * alpha;
		return revan;
	}


	public double [] evalFast(double t) {
		if(t == 0.5)
			System.out.println("   ");
		double [] p = new double [getControlPoints().get(0).getDimension()];
		double internal = ((t * getKnots().length));
		int numPts = getControlPoints().size();
		double step = internal- Math.floor(internal);

		//System.out.println(t + " " + internal);
		double [] weights = getWeights(step);
		for (int i=0;i<4;i++){
			double [] loc; 
				loc = getControlPoint((int) (internal+i)).getCoordinates();
			for (int j = 0; j < loc.length; j++) {
				p[j] += (loc[j] * weights[i]);
				//p[j] = loc[j];
			}
		}
		return p;
	}
	


}
