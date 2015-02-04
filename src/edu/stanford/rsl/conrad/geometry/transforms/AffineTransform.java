package edu.stanford.rsl.conrad.geometry.transforms;

import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Affine Transform Class where y = T(x)= A(x)+ b and x = T^-1(y) = A^-1(y - b)
 * @author Rotimi X Ojo 
 */
public class AffineTransform extends ComboTransform  {

	private static final long serialVersionUID = -6509394061746644677L;
	
	/**
	 * Creates a transform that scales the data from (point1Before, point2Before) to (point1After, point2After). No rotation is considered. Internally every dimension is
	 * scaled individually:<br><br>
	 * 
	 * point1After = transform(point1Before)
	 * <br><br>
	 * and
	 * <br><br>
	 * point2After = transform(point2Before)
	 * 
	 * @param point1Before the minimum before the transform
	 * @param point2Before the maximum before the transform
	 * @param point1After the minimum after the transform
	 * @param point2After the maximum after the transfrom
	 */
	public AffineTransform(PointND point1Before, PointND point2Before, PointND point1After, PointND point2After){
		LinearFunction funcx =  new LinearFunction(point1Before.get(0), point2Before.get(0), point1After.get(0), point2After.get(0));
		LinearFunction funcy =  new LinearFunction(point1Before.get(1), point2Before.get(1), point1After.get(1), point2After.get(1));
		LinearFunction funcz =  new LinearFunction(point1Before.get(2), point2Before.get(2), point1After.get(2), point2After.get(2));
		SimpleMatrix scaleRotate = new SimpleMatrix(3, 3);
		scaleRotate.setElementValue(0, 0, funcx.getM());
		scaleRotate.setElementValue(1, 1, funcy.getM());
		scaleRotate.setElementValue(2, 2, funcz.getM());
		SimpleVector translatorVec = new SimpleVector(funcx.getT(),funcy.getT(),funcz.getT());
		super.init(new ScaleRotate(scaleRotate), new Translation(translatorVec));
	}
	
	
	/**
	 * Initialize a new Affine transform;
	 * @param scaleRotate is the transformation matrix
	 * @param translatorVec is the translation vector
	 */
	public AffineTransform(SimpleMatrix scaleRotate, SimpleVector translatorVec){
		if(scaleRotate == null || translatorVec == null){
			throw new RuntimeException("Null inputs are not supported");
		}
		super.init(new ScaleRotate(scaleRotate), new Translation(translatorVec));
	}
	
	/**
	 * Initialize a new Affine transform
	 * @param transform0 is the first transformation
	 * @param transform1 is the second transformation
	 */
	public AffineTransform(Transform transform0, Transform transform1) {
		if(transform0 == null || transform1 == null){
			throw new RuntimeException("Null inputs are not supported");
		}
		super.init(transform0,transform1);
	}
	
	@Override
	public AffineTransform inverse(){	
		Transform t0 = transforms.get(1).inverse();
		Transform t1 = transforms.get(0).inverse();
		return new AffineTransform(t0,t1);
	}
	
	public AffineTransform clone(){
		return new AffineTransform(transforms.get(0).clone(), transforms.get(1).clone());
	}

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/