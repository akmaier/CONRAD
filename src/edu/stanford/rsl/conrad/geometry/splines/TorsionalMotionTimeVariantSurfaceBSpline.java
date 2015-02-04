/*
 * Copyright (C) 2014 Katrin Mentl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.splines;


import java.awt.Color;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/*long axis of the heart is the z-axis */
public class TorsionalMotionTimeVariantSurfaceBSpline extends 
TimeVariantSurfaceBSpline{

	private double torsionAngleBaseMax = 0.0;
	private double torsionAngleApexMax = 0.0;
	private double shorteningRatioMax = 0.0;
	private double zApexPos = 0.0;
	private double zBasePos = 0.0;
	ArrayList<PointND> transformationCenters = new ArrayList<PointND>();
	PointND transformationCenter = new PointND(0,0,0);
	Translation back = new Translation();
	Translation toCenter = new Translation();
	
	//rotation axis equals z axis (-> index 2)
	SimpleVector rotationAxis = new SimpleVector(0,0,0);
	
	public TorsionalMotionTimeVariantSurfaceBSpline(ArrayList<SurfaceBSpline> spline, SimpleVector rotationAx, double torsionAngleBMax, double torsionAngleAMax, double shorteningRatio, double[] bPos, double[] aPos, PointND firstTrafoCenter, ArrayList<PointND> trafoCenters){
		//timeVariantShapes = splines; in super(splines)
		super(spline);
		this.rotationAxis = rotationAx;
		this.torsionAngleBaseMax = torsionAngleBMax;
		//the apex rotates counterclockwise so negate the angle!
		this.torsionAngleApexMax = torsionAngleAMax;
		this.shorteningRatioMax = shorteningRatio;
		this.zApexPos = aPos[0];
		this.zBasePos = bPos[0];
		this.transformationCenter = firstTrafoCenter;
		this.transformationCenters = trafoCenters;
		this.back = new Translation(transformationCenter.getAbstractVector());
		this.toCenter = back.inverse();
		
		/*for(int i = 0; i < bPos.length; i++){
			System.out.println("Basepos at heartstate " + i + " is " + bPos[i]);
			System.out.println("Apexpos at heartstate " + i + " is " + aPos[i]);
		}*/
		
		//base rotates clockwise (as seen from apex)
		//Transform rotationOfBase = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new Axis(rotationAxis), torsionAngleBaseMax));
		//apex rotates counterclockwise (as seen from apex)
		//Transform rotationOfApex = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new Axis(rotationAxis), torsionAngleApexMax));

		double factor = 0;
		
		//for PointCloudViewer
		ArrayList<PointND> points = new ArrayList<PointND>();
		ArrayList<Color> colors = new ArrayList<Color>();

		//27 different heartstates from XCat file
		for(int k = 0; k < timeVariantShapes.size(); k ++){
			//set sum and means back to zero if they are calculated for every heartstate 
			
			
			double torsionAngleBase = 0.0;
			double torsionAngleApex = 0.0;
		
			//System.out.println("Factor at Heartphase  " + k + " is " + factor);
			
			//assume the first heartstate refers to diastole?
			//if k < 14  (14 first heartstates) increase angle and shorteningRatio
			double size = timeVariantShapes.size();
			if(k < (timeVariantShapes.size() - 1)/2){
				torsionAngleBase = (factor/size) * torsionAngleBaseMax;
				torsionAngleApex = (factor/size) * torsionAngleApexMax;
				shorteningRatio = (factor/size) * shorteningRatioMax;
				factor = factor + (size-1)/12;//angleFactor should have reached 26 now
				//System.out.println("AngleFactor: " + angleFactor);
			}
			//if k == 13 (mean)
			else if(k == (timeVariantShapes.size() - 1)/2){
				torsionAngleBase = torsionAngleBaseMax;
				torsionAngleApex = torsionAngleApexMax;	
				shorteningRatio = shorteningRatioMax;
				factor = factor - (size-1)/12;
			} else{
				//if k > 13 decrease angles and shorteningRatio 
				torsionAngleBase = (factor/size) * torsionAngleBaseMax;
				torsionAngleApex = (factor/size) * torsionAngleApexMax;
				shorteningRatio = (factor/size) * shorteningRatioMax;
				factor = factor - (size-1)/12;//decrease angleFactor
				
				
			}
			
			
			

			
			
			SurfaceBSpline currentSplineHeartStateK = (SurfaceBSpline)timeVariantShapes.get(k);
			ArrayList<PointND> controlPoints = currentSplineHeartStateK.getControlPoints();

			
			//fit the transformation Centers to each heart state
			back = new Translation(transformationCenters.get(k).getAbstractVector());
			toCenter = back.inverse();
			
			//doesn't work using different base and apex positions???
			double zBasePosCurrent = bPos[k];
			double zApexPosCurrent = aPos[k];
			//System.out.println("zBasePosCurrentHeartstate: " + zBasePosCurrent);
			//System.out.println("zApexPosCurrentHeartstate: " + zApexPosCurrent);
			
			for(int c = 0; c < controlPoints.size(); c ++){
				//current controlPoint 
				PointND controlPointCurrent = controlPoints.get(c);
				
				//use z value
				double zCurrent = controlPointCurrent.get(2);
				
				
				
				//interpolation formula to calculate rotationAngle at current z Position of control Point
				//do not take the maximum angles but the ones fitted to the different heartstates
				//double rotAngleCurrent = ((zBasePos - zCurrent)/(zBasePos - zApexPos)) * torsionAngleApex + ((zCurrent - zApexPos)/(zBasePos - zApexPos))* torsionAngleBase;
				double rotAngleCurrent = ((zBasePosCurrent - zCurrent)/(zBasePosCurrent - zApexPosCurrent)) * torsionAngleApex + ((zCurrent - zApexPosCurrent)/(zBasePosCurrent - zApexPosCurrent))* torsionAngleBase;
				
				
				if(zBasePosCurrent < zCurrent){
					rotAngleCurrent = torsionAngleBaseMax;
				}
				
				if(zApexPosCurrent > zCurrent){
					rotAngleCurrent = torsionAngleApexMax;
				}
				
				Transform rotationCurrent = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new Axis(rotationAxis), rotAngleCurrent));
				
				PointND controlPointRotated = back.transform(rotationCurrent.transform(toCenter.transform(controlPointCurrent)));
				
				SimpleVector newControlPointVector = new SimpleVector(controlPointRotated.get(0), controlPointRotated.get(1), controlPointRotated.get(2));
				
				//overwrite coordinates of controlPoint with coordinates of rotated ControlPoint
				controlPointCurrent.setCoordinates(newControlPointVector);
				
				//for PointCloudViewer
				if (c%2 == 0) {
					points.add(controlPointCurrent);
					colors.add(new Color(128,128,(int)(255-(c*5%255))));
				}
				//not needed anymore?
				//points.add(controlPointCurrent);
				//colors.add(new Color(128, 128, (int) (255 - (((double)(c))/controlPoints.size()))));
				//.add(new Color(128, 128, (int) (255 - (((double(c))/controlPoints.size()))));
			
			}
			
			
		}
		//uncomment to use pointCloudViewer
		/*PointCloudViewer pcv = new PointCloudViewer(this.getTitle(), points);
		pcv.setColors(colors);
		pcv.setVisible(true);*/
		
	}
	
}

