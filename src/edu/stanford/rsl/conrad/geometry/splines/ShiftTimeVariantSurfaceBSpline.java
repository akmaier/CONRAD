package edu.stanford.rsl.conrad.geometry.splines;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ShiftTimeVariantSurfaceBSpline extends TimeVariantSurfaceBSpline {
	
	public ShiftTimeVariantSurfaceBSpline(ArrayList<SurfaceBSpline> splines){
		super(splines);
		
		
		for(int k = 0; k < timeVariantShapes.size(); k ++){
			SurfaceBSpline currentSplineHeartStateK = (SurfaceBSpline)timeVariantShapes.get(k);
			ArrayList<PointND> controlPoints = currentSplineHeartStateK.getControlPoints();
			for(int c = 0; c < controlPoints.size(); c ++){
				PointND controlPoint = controlPoints.get(c);
				SimpleVector newControlPointVector = new SimpleVector(0,0,0);
				//if(k < 12){
					double[] coordinates = controlPoint.getCoordinates();
					coordinates[0] = coordinates[0]/1;
					coordinates[1] = coordinates[1]/1 + 100;
					coordinates[2] = coordinates[2]/1;
					newControlPointVector = new SimpleVector(coordinates[0], coordinates[1], coordinates[2]);
				//	}
				controlPoint.setCoordinates(newControlPointVector);
			}
		}
	}
}
