package edu.stanford.rsl.conrad.geometry.motion;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.xcat.CombinedBreathingHeartScene;
import edu.stanford.rsl.conrad.utils.TessellationUtil;

public class CombinedBreathingHeartMotionField extends ParzenWindowMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4787248458841471038L;
	private CombinedBreathingHeartScene breathingHeartScene;
	
	public CombinedBreathingHeartMotionField(CombinedBreathingHeartScene scene, double sigma){
		super(sigma);
		this.breathingHeartScene = scene;
	}
	
	@Override
	PointND[] getRasterPoints(double time) {
		PointND[] result = timePointMap.get(time);
		if (result == null){
			ArrayList<PointND[]> allPoints= new ArrayList<PointND[]>();
			int sizes = 0;
			// Deformation caused by the breathing splines
			ArrayList<TimeVariantSurfaceBSpline> variants = breathingHeartScene.getBreathing().getVariants();
			for(int i=0; i< variants.size(); i++){
				PointND [] current = variants.get(i).getRasterPoints(
						TessellationUtil.getSamplingU(variants.get(i)), 
						TessellationUtil.getSamplingV(variants.get(i)), 
						breathingHeartScene.getBreathing().getTimeWarper().warpTime(time));
				allPoints.add(current);
				sizes += current.length;
			}
			// fixed points that are time invariant in the breathing scene
			ArrayList<SurfaceBSpline> invariants = breathingHeartScene.getBreathing().getSplines();
			for(int i=0; i< invariants.size(); i++){
				PointND [] current = invariants.get(i).getRasterPoints(
						TessellationUtil.getSamplingU(invariants.get(i)), 
						TessellationUtil.getSamplingV(invariants.get(i)));
				allPoints.add(current);
				sizes += current.length;
			}
			
			SimpleVector diaphragmMotion =(breathingHeartScene.getBreathing().getDiaphragmMotionVector(0, time));
			// Deformation caused by the motion of the heart.
			variants = breathingHeartScene.getHeart().getVariants();
			for(int i=0; i< variants.size(); i++){
				PointND [] current = variants.get(i).getRasterPoints(
						TessellationUtil.getSamplingU(variants.get(i)), 
						TessellationUtil.getSamplingV(variants.get(i)), 
						breathingHeartScene.getHeart().getTimeWarper().warpTime(time));
				for (int j =0;j<current.length;j++){
					current[j].applyTransform(breathingHeartScene.getHeartTranslation());
					current[j].applyTransform(new Translation(diaphragmMotion));
				}
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
	 * @param breathingHeartScene the breathingHeartScene to set
	 */
	public void setBreathingHeartScene(CombinedBreathingHeartScene breathingHeartScene) {
		this.breathingHeartScene = breathingHeartScene;
	}

	/**
	 * @return the breathingHeartScene
	 */
	public CombinedBreathingHeartScene getBreathingHeartScene() {
		return breathingHeartScene;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/