package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class StraightLine extends AbstractCurve {

	private static final long serialVersionUID = 4615305861839196846L;
	protected SimpleVector direction;
	protected PointND point;
	
	/**
	 * Creates a StraightLine from point with direction dir 
	 * @param point the point
	 * @param dir the direction
	 */
	public StraightLine(PointND point, SimpleVector dir){
		this.direction = dir;
		this.point = point;
	}
	
	public StraightLine(StraightLine sl){
		super(sl);
		direction = (sl.direction != null) ? sl.direction.clone() : null;
		point = (sl.point != null) ? sl.point.clone() : null;
	}
	
	public void init(PointND point, SimpleVector dir){
		this.direction = dir;
		this.point = point;
	}

	/**
	 * Creates a new Straight line passing from point to point2
	 * @param point the base point
	 * @param point2 the other point
	 */
	public StraightLine(PointND point, PointND point2){
		this.direction = SimpleOperators.subtract(point2.getAbstractVector(), point.getAbstractVector());
		this.point = point;
	}
	
	public void init(PointND point, PointND point2){
		this.direction = SimpleOperators.subtract(point2.getAbstractVector(), point.getAbstractVector());
		this.point = point;
	}

	public void normalize(){
		direction.normalizeL2();
	}
	
	@Override
	public PointND evaluate(double u) {
		return new PointND(SimpleOperators.add(point.getAbstractVector(), direction.multipliedBy(u)));
	}

	@Override
	public int getDimension() {
		return point.getDimension();
	}

	protected SimpleVector computeIntersectionCoefficients(StraightLine l2){
		SimpleVector b = SimpleOperators.subtract(l2.point.getAbstractVector(), point.getAbstractVector());
		SimpleMatrix A = new SimpleMatrix(3, 2);
		A.setColValue(0, direction);
		A.setColValue(1, l2.direction);
		return Solvers.solveLinearLeastSquares(A, b);
	}
	
	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		if (other instanceof StraightLine) {
			StraightLine l2 = (StraightLine) other;
			ArrayList<PointND> list = new ArrayList<PointND>();
			PointND i = intersect(l2);
			if (i != null)
				list.add(i);
			
			return list;

		} else {
			throw new RuntimeException("This curve is not supported yet");
		}
	}
	
	public PointND intersect(StraightLine line){
		return intersect(line, CONRAD.SMALL_VALUE*1000);
	}
	
	public PointND intersect(StraightLine line, double threshold){
		SimpleVector lambda = computeIntersectionCoefficients(line);
		PointND p1 = evaluate(lambda.getElement(0));
		PointND p2 = line.evaluate(-lambda.getElement(1));
		if (p1 != null) {
			if (p1.euclideanDistance(p2) < threshold){
				return p1;
			} 	
		}
		return null;
	}

	/**
	 * @return the direction
	 */
	public SimpleVector getDirection() {
		return direction;
	}

	/**
	 * @param direction the direction to set
	 */
	public void setDirection(SimpleVector direction) {
		this.direction = direction;
	}

	/**
	 * @return the point
	 */
	public PointND getPoint() {
		return new PointND(point);
	}

	/**
	 * @param point the point to set
	 */
	public void setPoint(PointND point) {
		this.point = point;
	}

	@Override
	public boolean isBounded() {
		return false;
	}

	/**
	 * Computes the closest distance between the line and the point p.
	 * @param p the point p
	 * @return the distance
	 */
	public double computeDistanceTo(PointND p) {
		SimpleVector vector = SimpleOperators.subtract(point.getAbstractVector(), p.getAbstractVector());
		double projectionValue = SimpleOperators.multiplyInnerProd(vector, direction);
		return p.euclideanDistance(evaluate(projectionValue));
	}

	@Override
	public void applyTransform(Transform t) {
		point = t.transform(point);
		direction = t.transform(direction);
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		return null;
	}
	
	@Override
	public String toString(){
		return point + " + n * " + direction;
	}

	@Override
	public AbstractShape clone() {
		return new StraightLine(this);
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/