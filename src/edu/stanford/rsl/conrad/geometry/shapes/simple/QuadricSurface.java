package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * <p>Class to model an arbitrary quadric surface like cylinders, cones, and spheres.
 * <p>For more information on quadric surfaces check <a href="http://en.wikipedia.org/wiki/Quadric"> Quadric Surfaces</a>
 * <p>The main function of this class is to provide intersection and membership verification support to all derived quadric surfaces.
 * 
 * @author Rotimi X Ojo
 */
public abstract class QuadricSurface extends SimpleSurface{

	private static final long serialVersionUID = 2995274520995183991L;
	protected SimpleMatrix constMatrix;
	protected double constant;
	protected PointND origin = new PointND(0,0,0);
	private ArrayList<PointND> dummy = new ArrayList<PointND>();
	
	public QuadricSurface(){
		super();
	}
	
	public QuadricSurface(QuadricSurface qs){
		super(qs);
		constMatrix = (qs.constMatrix!= null) ? qs.constMatrix.clone() : null;
		constant = qs.constant;
		origin = (qs.origin != null) ? qs.origin.clone() : null;
		
		if (qs.dummy != null){
			Iterator<PointND> it = qs.dummy.iterator();
			dummy = new ArrayList<PointND>();
			while (it.hasNext()) {
				PointND p = it.next();
				dummy.add((p!=null) ? p.clone() : null);
			}
		}
		else{
			dummy = null;
		}
	}


	
	public ArrayList<PointND> getHits(AbstractCurve other) {
		SimpleVector originVec = origin.getAbstractVector();
		StraightLine line = (StraightLine)other;
		SimpleVector lineOrigin = line.getPoint().getAbstractVector();
		SimpleVector direction = line.getDirection();
		
		if (constMatrix == null) {
			throw new RuntimeException("Please initialize constMatrix");
		}
		ArrayList<PointND> hits = new ArrayList<PointND>();			
		
		double a = SimpleOperators.multiplyInnerProd(direction, SimpleOperators.multiply(constMatrix, direction));		
		double b = 2*SimpleOperators.multiplyInnerProd(direction,	SimpleOperators.multiply(constMatrix, SimpleOperators.add(originVec,lineOrigin)));
		double c1 = SimpleOperators.multiplyInnerProd(lineOrigin,SimpleOperators.multiply(constMatrix, lineOrigin))- constant;
		double c2 = SimpleOperators.multiplyInnerProd(originVec, SimpleOperators.multiply(constMatrix, originVec));
		double c3 = 2*SimpleOperators.multiplyInnerProd(originVec, SimpleOperators.multiply(constMatrix, lineOrigin));
		double c = c1 + c2 + c3;
		
		double buff = b * b - 4 * a * c ;
		if (buff > -CONRAD.FLOAT_EPSILON) {
			if(buff < 0) buff =0;
			double leftVal = -b/(2*a);
			double rightVal = Math.sqrt(buff)/(2*a);

			SimpleVector buffvec = direction.multipliedBy(rightVal + leftVal);
			buffvec.add(lineOrigin);
			hits.add(new PointND(buffvec));
			buffvec = direction.multipliedBy(leftVal - rightVal);
			buffvec.add(lineOrigin);
			hits.add(new PointND(buffvec));			
		}

		return hits;
	}

	public abstract  boolean isBounded();

	public int getDimension(){
		return 3;
	}

	@Override
	public boolean isMember(PointND point, boolean pointTransformed) {
		if (!pointTransformed)
			point= transform.transform(point);
		SimpleVector vec = point.getAbstractVector();
		double eval = SimpleOperators.multiplyInnerProd(vec,SimpleOperators.multiply(constMatrix, vec));
		if(eval <= constant){
			return true;
		}
		return false;
	}
	

	@Override
	public ArrayList<PointND> getHitsOnBoundingBox(AbstractCurve other){
		if(dummy.size() == 0){
			dummy.add(origin);
		}
		return dummy;
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		// TODO Auto-generated method stub
		return null;
	}



	@Override
	public PointND evaluate(double u, double v) {
		// TODO Auto-generated method stub
		return null;
	}	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/