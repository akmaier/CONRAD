package edu.stanford.rsl.conrad.geometry.motion;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

public class MixedSurfaceBSplineMotionField extends ParzenWindowMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4288495612235276488L;

	public MixedSurfaceBSplineMotionField(ArrayList<TimeVariantSurfaceBSpline> variants, ArrayList<SurfaceBSpline> invariants, double sigma) {
		super(sigma);
		this.invariants = invariants;
		this.variants = variants;
	}

	private ArrayList<TimeVariantSurfaceBSpline> variants;
	private ArrayList<SurfaceBSpline> invariants;
	
	@Override
	PointND[] getRasterPoints(double time) {
		PointND[] result = timePointMap.get(time);
		if (result == null){
			ArrayList<PointND[]> allPoints= new ArrayList<PointND[]>();
			int sizes = 0;
			for(int i=0; i< variants.size(); i++){
				PointND [] current = variants.get(i).getRasterPoints(TessellationUtil.getSamplingU(variants.get(i)), TessellationUtil.getSamplingV(variants.get(i)), time);
				allPoints.add(current);
				sizes += current.length;
			}
			for(int i=0; i< invariants.size(); i++){
				PointND [] current = invariants.get(i).getRasterPoints(TessellationUtil.getSamplingU(invariants.get(i)), TessellationUtil.getSamplingV(invariants.get(i)));
				allPoints.add(current);
				sizes += current.length;
			}
			result = new PointND[sizes];
			sizes = 0;
			for(int i=0; i< allPoints.size(); i++){
				PointND[] current = allPoints.get(i);
				for (int j=0;j<current.length;j++){
					result[j+sizes] = current[j];
				}
				sizes += current.length;
			}
			timePointMap.put(time, result);
		}
		return result;
	}

	/**
	 * @param invariants the invariants to set
	 */
	public void setInvariants(ArrayList<SurfaceBSpline> invariants) {
		this.invariants = invariants;
	}

	/**
	 * @return the invariants
	 */
	public ArrayList<SurfaceBSpline> getInvariants() {
		return invariants;
	}

	/**
	 * @param variants the variants to set
	 */
	public void setVariants(ArrayList<TimeVariantSurfaceBSpline> variants) {
		this.variants = variants;
	}

	/**
	 * @return the variants
	 */
	public ArrayList<TimeVariantSurfaceBSpline> getVariants() {
		return variants;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/