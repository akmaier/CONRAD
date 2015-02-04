package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;




import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * There are 3 representations for a plane:
 * <ul>
 * <li>The parametric representation defines the plane using a point P and two non-colinear vectors u and v, so that the plane is defined by {@latex.inline $\\mathbf P + \\alpha \\cdot \\mathbf u + \\beta \\cdot \\mathbf v, \\quad \\alpha, \\beta \\in \\mathbb R$}.
 * <li>The normal form uses a unit normal vector n and an offset value d, defining the plane as the set {@latex.inline $\\{ \\mathbf x : \\mathbf{n}^T \\mathbf x = d \\}$}.
 * <li>The mixed form consists of a point P in the plane and the unit vector n normal to the plane so that the plane is {@latex.inline $\\{ \\mathbf x : \\mathbf{n}^T (\\mathbf x - \\mathbf P) = 0 \\}$}.
 * </ul>
 * All these representations are stored internally and can be requested as soon as the plane has been created (using any form of constructor).
 * @author Andreas Keil
 */
public class Plane3D extends AbstractSurface {

	private static final long serialVersionUID = -6304736161052003537L;

	protected SimpleVector dirU;
	protected SimpleVector dirV;
	protected PointND pointP;
	protected SimpleVector normalN;
	protected double offsetD;

	/**
	 * Creates a plane from the given parametric representation {@latex.inline $\\mathbf P + \\alpha \\cdot \\mathbf u + \\beta \\cdot \\mathbf v, \\quad \\alpha, \\beta \\in \\mathbb R$}.
	 * @param p1  An arbitrary point in the plane.
	 * @param dirU  A first direction vector in the plane.
	 * @param dirV  A second direction vector in the plane which is not colinear with the first vector.
	 */
	public Plane3D(PointND p1, SimpleVector dirU, SimpleVector dirV) {
		// input checks
		if (dirU.normL2() < Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("First given direction vector must not be null!");
		if (dirV.normL2() < Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Second given direction vector must not be null!");

		// assign point parameter
		this.pointP = p1.clone();
		this.dirU = dirU;
		this.dirV = dirV;
		// init the other representations
		initFromParametricRepresentation();
	}

	/**
	 * Creates a plane from the given normal form {@latex.inline $\\{ \\mathbf x : \\mathbf{n}^T \\mathbf x = d \\}$}.
	 * Note that the given normal vector is normalized and stored as a unit vector.
	 * @param normal  A vector that is normal to the plane.
	 * @param offset  The offset from the coordinate system's origin to this plane in the normal direction.
	 *        This offset value is negative if the normal does not point from the origin to the plane but in the opposite direction.
	 */
	public Plane3D(SimpleVector normal, double offset) {
		// input check
		final double normalMagnitude = normal.normL2();
		if (normalMagnitude < Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Given normal vector is a null vector!");

		// store parameters
		this.normalN = normal.dividedBy(normalMagnitude);
		this.offsetD = offset / normalMagnitude;

		// init other representations
		initFromNormalForm();
	}

	/**
	 * Creates a plane from a given point in the plane and a vector normal to the plane so that {@latex.inline $\\{ \\mathbf x : \\mathbf{n}^T (\\mathbf x - \\mathbf P) = 0 \\}$}.
	 * This representation is called <em>mixed form</em> here.
	 * Note that the given normal vector is normalized and stored as a unit vector.
	 * @param point  Any point in the plane.
	 * @param normal  A vector normal to the plane.
	 */
	public Plane3D(PointND point, SimpleVector normal) {
		// input check
		final double normalMagnitude = normal.normL2();
		if (normalMagnitude < Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Given normal vector is a null vector!");

		// store parameters
		this.normalN = normal.dividedBy(normalMagnitude);
		this.pointP = point.clone();

		// init other representations
		initFromMixedForm();
	}

	/**
	 * Initializes the plane to the one with minimum sum of squared distances from all given points.
	 * @param points  The array or comma-separated list of points this plane should be fitted to.
	 * 	 
	 */
	public Plane3D(PointND... points) {
		// linear model for plane: z = a0*x + a1*y + a2

		final int noPoints = points.length;
		final int noParameters = 3;
		assert noPoints >= 3 : new IllegalArgumentException("At least 3 points are needed for determining a plane in 3D!");

		// add a line (point^T(0..1), 1) for each point to assemble the matrix for the target function ||Ax-b||
		SimpleMatrix A = new SimpleMatrix(noPoints, noParameters);
		for (int i = 0; i < noPoints; ++i) {
			SimpleVector pointVec = points[i].getAbstractVector();
			assert pointVec.getLen() == 3 : new IllegalArgumentException("Points have to be in 3D for fitting a plane!");
			A.setSubRowValue(i, 0, pointVec);
			A.setElementValue(i, 2, 1.0);
		}
		
		// create right hand side b, filled with z-components and solve minimization problem min||Ax-b||
		SimpleVector b = new SimpleVector(noPoints);
		for (int i = 0; i < noPoints; ++i) {
			SimpleVector pointVec = points[i].getAbstractVector();
			assert pointVec.getLen() == 3 : new IllegalArgumentException("Points have to be in 3D for fitting a plane!");
			b.setElementValue(i, pointVec.getElement(2));
		}
		SimpleVector x = Solvers.solveLinearLeastSquares(A, b);
		assert x.getLen() == 3 : new RuntimeException("Unexpected dimension for plane fitting result!");
		
		// assign solution to plane parameters
		this.normalN = new SimpleVector(3);
		this.normalN.setElementValue(0, x.getElement(0));
		this.normalN.setElementValue(1, x.getElement(1));
		this.normalN.setElementValue(2, -1.0);
		// calculate L2-Norm of current normal vector and normalize normal vector and offset
		final double normalMagnitude = normalN.normL2();
		this.normalN.divideBy(normalMagnitude);
		this.offsetD = -x.getElement(2)/normalMagnitude;

		// update parametric representation from normal representation
		initFromNormalForm();
}



	public Plane3D(Plane3D plane3d) {
		super(plane3d);
		dirU = (plane3d.dirU!=null) ? new SimpleVector(plane3d.dirU) : null;
		dirV = (plane3d.dirV!=null) ? new SimpleVector(plane3d.dirV) : null;
		pointP = (plane3d.pointP!=null) ? new PointND(plane3d.pointP) : null;
		normalN = (plane3d.normalN!=null) ? new SimpleVector(plane3d.normalN) : null;
		offsetD = plane3d.offsetD;
	}

	@Override
	public int getDimension() {
		//		return point.getLen();
		return 3; //TODO: Hyperplanes in higher/lower dimension may be defined in the future, requiring some refactoring for splitting up the 3D and the n-D related stuff.
	}

	@Override
	public PointND evaluate(double u, double v) {
		return new PointND(SimpleOperators.add(pointP.getAbstractVector(), dirU.multipliedBy(u), dirV.multipliedBy(v)));
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		if (other instanceof StraightLine) {
			ArrayList<PointND> list = new ArrayList<PointND>();
			list.add(intersect((StraightLine) other));
			return list;
		} else {
			throw new RuntimeException("Not implemented yet!");
		}
	}

	public PointND intersect(StraightLine l) {
		// compute line parameter from inserting the equation for a point on the line into the plane's normal form
		double denominator = (SimpleOperators.multiplyInnerProd(this.normalN, l.getDirection()));
		if (denominator == 0) throw new RuntimeException("Line is parallel to plane");
		double numerator = this.offsetD - SimpleOperators.multiplyInnerProd(this.normalN, l.getPoint().getAbstractVector());
		return l.evaluate(numerator/denominator);
	}

	@Override
	public boolean isBounded() {
		return false;
	}
	
	public double getOffset() {
		
		return this.offsetD;
	}
	
	/**
	 * Orient the normal either from the origin to the plane or in the opposite direction.
	 * If the normal points from the direction to the plane, then the offset parameter d will be positive.
	 * Otherwise it will be negative.
	 * @param fromOriginToPlane  specifies the direction in which to orient the plane.
	 */
	public void orientNormal(boolean fromOriginToPlane) {
		if ((fromOriginToPlane && this.offsetD < 0.0) || (!fromOriginToPlane && this.offsetD > 0.0)) { // normalize so that offset is positive
			this.offsetD *= -1.0;
			this.normalN.negate();
		}
	}

	/**
	 * Computes the distance of a point to this plane.
	 * @param givenPoint  The point whose distance is to be computed and whose closest neighbor on the plane is to be determined.
	 * @return  The signed distance of the given point to this plane.
	 *          The plane's unit normal vector has to be multiplied with this signed distance to get from the closest point to the given point, i.e.
	 *          {@latex.ilb \\[ \\mathbf{G} = \\mathbf{C} + d \\cdot \\mathbf{n} \\]}
	 *          where G is the given point, C the closest point on the plane, and n the plane's normal.
	 */
	public double computeDistance(PointND givenPoint) {
		// input check
		SimpleVector givenPointAsVec = givenPoint.getAbstractVector();
		final double distance = SimpleOperators.multiplyInnerProd(this.normalN, givenPointAsVec) - this.offsetD;
		return distance;
	}

	/**
	 * Computes the distance of a point to this plane and returns the closest point to the given point on the plane.
	 * @param givenPoint  The point whose distance is to be computed and whose closest neighbor on the plane is to be determined.
	 * @param closestPoint  The closest point to the given one on this plane is returned here.
	 * @return  The signed distance of the given point to this plane.
	 *          The plane's unit normal vector has to be multiplied with this signed distance to get from the closest point to the given point, i.e.
	 *          {@latex.ilb \\[ \\mathbf{G} = \\mathbf{C} + d \\cdot \\mathbf{n} \\]}
	 *          where G is the given point, C the closest point on the plane, and n the plane's normal.
	 */
	public double computeDistance(PointND givenPoint, PointND closestPoint) {
		double distance = computeDistance(givenPoint);
		closestPoint.getAbstractVector().init(SimpleOperators.subtract(givenPoint.getAbstractVector(), this.normalN.multipliedBy(distance)));
		return distance;
	}



	private void initFromParametricRepresentation() {
		// init normal form
		this.normalN = General.crossProduct(this.dirU.normalizedL2(), this.dirV.normalizedL2());

		// internal check
		double normN = this.normalN.normL2();
		if (normN < Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("The given direction vectors are colinear!");

		this.normalN.divideBy(normN);
		this.offsetD = SimpleOperators.multiplyInnerProd(this.normalN, this.pointP.getAbstractVector());
	}

	/**
	 * flip the normal of the plane
	 */
	public void flipNormal(){
		normalN.negate();
		offsetD *= -1;
	}

	private void initFromMixedForm() {
		// init normal form
		this.offsetD = SimpleOperators.multiplyInnerProd(this.normalN, this.pointP.getAbstractVector());

		// init parametric representation
		initDirectionalVectorsFromNormal();
	}

	private void initFromNormalForm() {
		// init mixed form
		this.pointP = new PointND(this.normalN.multipliedBy(this.offsetD)); // choose closest point to origin

		// init parametric representation
		initDirectionalVectorsFromNormal();
	}

	private void initDirectionalVectorsFromNormal() {
		SimpleVector someVec = new SimpleVector(3);
		someVec.setElementValue(0, 1.0);
		this.dirU = General.crossProduct(this.normalN, someVec);
		if (this.dirU.normL2() < Math.sqrt(CONRAD.DOUBLE_EPSILON)) {
			someVec.setElementValue(0, 0.0);
			someVec.setElementValue(1, 1.0);
			this.dirU = General.crossProduct(this.normalN, someVec);
		}
		this.dirU.normalizeL2();
		this.dirV = General.crossProduct(this.normalN, this.dirU);
		this.dirV.normalizeL2();
	}

	@Override
	public void applyTransform(Transform t) {
		SimpleVector buff = t.transform(normalN);
		normalN = buff.dividedBy(buff.normL2());
		pointP = t.transform(pointP);
		initFromMixedForm();
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		return null;
	}

	public SimpleVector getNormal() {
		return normalN;
	}

	public PointND getPoint() {
		return new PointND(pointP);
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape clone() {
		return new Plane3D(this);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/