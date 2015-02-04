package edu.stanford.rsl.conrad.phantom.forbild;

import java.util.regex.Pattern;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SimpleSurface;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildBox;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildCone;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildCylinder;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildEllipsoid;
import edu.stanford.rsl.conrad.phantom.forbild.shapes.ForbildSphere;


/**
 * <p>This class creates a forbild surface given an appropriate definition.</p>
 * 
 * @author Rotimi .X. Ojo
 *
 */
public class ForbildShapeFactory {

	/**
	 * Determines AbstractShape specified by given bound expressions
	 * @param objBounds forbild bound definition e.g [ Box: x=0;  y=9;  z=2.5;  dx=2.5;  dy=0.8;  dz=25;]
	 * @return shape
	 */
	public static AbstractShape getShape(String objBounds) {
		objBounds = objBounds.substring(2,objBounds.indexOf(']'));
		SimpleSurface shape = getBaseShape(objBounds);	
		return shape;
	}

	public static SimpleSurface getBaseShape(String objBounds){
		String lower = objBounds.toLowerCase();

		// Forbild uses degrees instead of radians, thus, we convert them to a special degree function (cosd,sind,tand).
		objBounds = objBounds.replaceAll("(?i)\\bcos\\b", "cosd").
				replaceAll("(?i)\\bsin\\b", "sind").
				replaceAll("(?i)\\btan\\b", "tand");

		// Forbild uses cm instead of mm as units, thus we apply a scaling by 10 to all Forbild shapes
		Transform cmTOmm = new ScaleRotate(SimpleMatrix.I_3.multipliedBy(10));


		SimpleSurface result = null;
		// debugging
		//if (lower.contains("ellipt_cyl") && lower.contains("axis(1,0,0)")){
			//System.out.println(objBounds);
			if(lower.contains("cyl")){
				result = new ForbildCylinder(objBounds);
			}else if(lower.contains("sphere")){
				result = new ForbildSphere(objBounds);
			}else if(lower.contains("ellipsoid")){
				result = new ForbildEllipsoid(objBounds);
			}else if(lower.contains("box")){
				result = new ForbildBox(objBounds);
			}else if(lower.contains("cone")){
				result = new ForbildCone(objBounds);
			}
			result.applyTransform(cmTOmm);
		//}
		return result;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */