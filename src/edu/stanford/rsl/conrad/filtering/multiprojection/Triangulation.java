package edu.stanford.rsl.conrad.filtering.multiprojection;

import javax.swing.JOptionPane;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.motion.CompressionMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.DualPhasePeriodicTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * This class triangulates the 3D position from subsequent projections.
 * <BR><BR>
 * The offset and the correction between the projections has to be set during configuration.
 * The result is stored in the Configuration and used in the subsequent reconstruction.
 * <BR><BR>
 * The image data is not altered. The filter is merely used to extract information.
 * 
 * @author Marco Bögel
 *
 */

public class Triangulation extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8039698341980560155L;
	private static int offset = 90;
	private static int totalProjs;
	private static SimpleVector pt3D;
	private static double[] motionfield;
	private static double[] positionfield;
	private static double maxMotion = 0;
	double correction = 1.0;

	private static double[] twomot;
	private static double[] posx, posy, posz;

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {

		//init
		if (projectionNumber == 0) {
			if (offset < 0)
				offset *= -1;
			//	offset /= Configuration.getGlobalConfiguration().getGeometry().getAverageAngularIncrement();
			setContext(offset + 1);
			totalProjs = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize();
			motionfield = new double[totalProjs];
			positionfield = new double[3 * totalProjs];
			twomot = new double[totalProjs];
			posx = new double[totalProjs];
			posy = new double[totalProjs];
			posz = new double[totalProjs];
			correction = Configuration.getGlobalConfiguration().getCorrectionFactor();
		}

		if (offset > totalProjs / 2)
			throw new Exception("The offset for depth triangulation is larger than the size of the projectionstack!");

		// This part of the code is used to select the triangulation algorithm
		// TODO: Move this into the configuration part.

		//		SimpleVector sol = triangulationSimple(projectionNumber,1);
		//		SimpleVector sol = triangulationIter(projectionNumber,1);
		//		SimpleVector sol = triangulationTrajectory(projectionNumber);
		SimpleVector sol = triangulationIterRectif(projectionNumber, 1);

		Grid2D imp = inputQueue.get(projectionNumber);

		motionfield[projectionNumber] = correction * (-pt3D.getElement(2) + sol.getElement(2));
		if (Math.abs(motionfield[projectionNumber]) > maxMotion)
			maxMotion = Math.abs(motionfield[projectionNumber]);
		positionfield[3 * projectionNumber] = sol.getElement(0);
		positionfield[3 * projectionNumber + 1] = sol.getElement(1);
		positionfield[3 * projectionNumber + 2] = sol.getElement(2);
		posx[projectionNumber] = sol.getElement(0);
		posy[projectionNumber] = sol.getElement(1);
		posz[projectionNumber] = sol.getElement(2);
		SimpleVector p1 = new SimpleVector(
				Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(projectionNumber));
		twomot[projectionNumber] = p1.getElement(1);

		if (projectionNumber == totalProjs - 1) {
			//			VisualizationUtil.createPlot(motionfield, "motionfieldbef", "time", "z").show();
			//			motionfield = DoubleArrayUtil.gaussianFilter(motionfield, 10 );

			Configuration.getGlobalConfiguration().setDiaphragmPositionField(positionfield);
			Configuration.getGlobalConfiguration().setRespiratoryMotionField(motionfield);

			Configuration.getGlobalConfiguration().setMaxMotion(maxMotion);
			VisualizationUtil.createPlot(motionfield, "motionfield", "frame", "z").show();
			VisualizationUtil.createPlot(positionfield, "positionfield", "frame", "z").show();
			VisualizationUtil.createPlot(twomot, "2d", "frame", "z").show();
			VisualizationUtil.createPlot(posx).show();
			VisualizationUtil.createPlot(posy).show();
			VisualizationUtil.createPlot(posz).show();

		}

		sink.process(imp, projectionNumber);

	}

	/**
	 * Simple Linear-Eigen Triangulation using SVD to solve A*x -> min
	 * @param image1 index of first projection
	 * @param image2 index of second projection
	 * @param compensate factor to compensate breathing motion between the two images 
	 * @return 4-D Vector of the 3-dimensional position
	 */
	private SimpleVector triangulation(int image1, int image2, double compensate) {
		SimpleVector p1 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image1));
		SimpleVector p2 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image2));

		if (compensate > 0) {
			double translation = p1.getElement(1) - p2.getElement(1);
			translation *= compensate;
			p2.addToElement(1, translation);
		}

		SimpleMatrix P1 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1).computeP();
		SimpleMatrix P2 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image2).computeP();

		//Linear-Eigen solution to the triangulation problem by Hartley & Sturm
		//start Triangulation
		//setup Linear Equations x*p3^T = p1^T      y*p3^T = p2^T 
		SimpleMatrix A = new SimpleMatrix(4, 4);
		//first image
		SimpleVector row1 = P1.getRow(2).multipliedBy(p1.getElement(0));
		row1.subtract(P1.getRow(0));
		SimpleVector row2 = P1.getRow(2).multipliedBy(p1.getElement(1));
		row2.subtract(P1.getRow(1));
		//second image
		SimpleVector row3 = P2.getRow(2).multipliedBy(p2.getElement(0));
		row3.subtract(P2.getRow(0));
		SimpleVector row4 = P2.getRow(2).multipliedBy(p2.getElement(1));
		row4.subtract(P2.getRow(1));
		A.setRowValue(0, row1);
		A.setRowValue(1, row2);
		A.setRowValue(2, row3);
		A.setRowValue(3, row4);

		//solve with SVD
		Matrix B = new Matrix(A.copyAsDoubleArray());
		SingularValueDecomposition svd = new SingularValueDecomposition(B);
		Matrix V = svd.getV();

		//solution is nullspace of A -> last column of V
		SimpleVector sol = new SimpleVector(V.getColumnDimension());
		int lastrow = V.getRowDimension() - 1;

		for (int i = 0; i < sol.getLen(); i++) {
			sol.setElementValue(i, V.get(i, lastrow));
		}
		//dehomogenize
		sol.divideBy(sol.getElement(3));
		//end Triangulation

		return sol;
	}

	/**
	 * Iterative Triangulation Method
	 * @param image1 index of first projection
	 * @param image2 index of second projection
	 * @param compensate factor to compensate for breathingmotion between the two images
	 * @return 4-D Vector of the 3-dimensional position
	 */
	private SimpleVector iterativeTriangulation(int image1, int image2, double compensate) {

		SimpleVector sol = null;

		SimpleVector p1 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image1));
		SimpleVector p2 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image2));

		if (compensate > 0) {
			p2.setElementValue(1, p1.getElement(1));
		}

		SimpleMatrix P1 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1).computeP();
		SimpleMatrix P2 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image2).computeP();

		SimpleVector row1 = P1.getRow(2).multipliedBy(p1.getElement(0));
		row1.subtract(P1.getRow(0));
		SimpleVector row2 = P1.getRow(2).multipliedBy(p1.getElement(1));
		row2.subtract(P1.getRow(1));
		SimpleVector row3 = P2.getRow(2).multipliedBy(p2.getElement(0));
		row3.subtract(P2.getRow(0));
		SimpleVector row4 = P2.getRow(2).multipliedBy(p2.getElement(1));
		row4.subtract(P2.getRow(1));

		double weight1 = 1;
		double weight2 = 1;
		double oldWeight1 = 2;
		double oldWeight2 = 2;
		int iterations = 0;

		while ((Math.abs(1 / weight1 - 1 / oldWeight1) > 0.1 || Math.abs(1 / weight2 - 1 / oldWeight2) > 0.1)
				&& iterations < 10) {

			//Linear-Eigen solution to the triangulation problem by Hartley & Sturm
			//start Triangulation
			//setup Linear Equations x*p3^T = p1^T      y*p3^T = p2^T 
			SimpleMatrix A = new SimpleMatrix(4, 4);
			//first image
			row1.divideBy(weight1);
			row2.divideBy(weight1);
			//second image
			row3.divideBy(weight2);
			row4.divideBy(weight2);

			A.setRowValue(0, row1);
			A.setRowValue(1, row2);
			A.setRowValue(2, row3);
			A.setRowValue(3, row4);

			//solve with SVD
			Matrix B = new Matrix(A.copyAsDoubleArray());
			SingularValueDecomposition svd = new SingularValueDecomposition(B);
			Matrix V = svd.getV();

			//solution is nullspace of A -> last column of V
			sol = new SimpleVector(V.getColumnDimension());
			int lastrow = V.getRowDimension() - 1;

			for (int i = 0; i < sol.getLen(); i++) {
				sol.setElementValue(i, V.get(i, lastrow));
			}
			//dehomogenize
			sol.divideBy(sol.getElement(3));

			//adjust weights
			oldWeight1 = weight1;
			oldWeight2 = weight2;
			weight1 = SimpleOperators.multiplyInnerProd(P1.getRow(2), sol);
			weight2 = SimpleOperators.multiplyInnerProd(P2.getRow(2), sol);
			//			System.out.println(image1 + "iter: "+iterations +"  weights: "+ weight1);

			iterations++;
		}
		//end Triangulation

		return sol;
	}

	/**
	 * This method first rectifies two specified images. Afterwards iterative triangulation is applied
	 * @param image1 index of first image
	 * @param image2 index of second image
	 * @param compensate motion compensation ( 0 for no comp, >0 for compensation)
	 * @return 4-D vector of the 3-dimensional position
	 */
	private SimpleVector rectifiedIterativeTriangulation(int image1, int image2, double compensate) {

		SimpleVector sol = null;

		SimpleVector p1 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image1));
		SimpleVector p2 = new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(image2));

		SimpleMatrix P1 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1).computeP();
		SimpleMatrix P2 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image2).computeP();

		SimpleVector c1 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1)
				.computeCameraCenter();
		SimpleVector c2 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image2)
				.computeCameraCenter();

		SimpleVector v1 = c1.clone();
		v1.subtract(c2);

		SimpleVector Rrow = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1).getR()
				.getRow(2);

		SimpleVector v2 = crossProduct(Rrow, v1);
		SimpleVector v3 = crossProduct(v1, v2);

		v1.normalizeL2();
		v2.normalizeL2();
		v3.normalizeL2();

		SimpleMatrix R = new SimpleMatrix(3, v1.getLen());
		R.setRowValue(0, v1);
		R.setRowValue(1, v2);
		R.setRowValue(2, v3);
		//intrinsic
		SimpleMatrix K1 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image1).getK();
		SimpleMatrix K2 = Configuration.getGlobalConfiguration().getGeometry().getProjectionMatrix(image2).getK();

		K1.add(K2);
		K1.divideBy(2);
		K1.setElementValue(0, 1, 0);

		SimpleMatrix Pn1 = new SimpleMatrix(3, 4);
		SimpleMatrix Pn2 = new SimpleMatrix(3, 4);

		Pn1.setSubMatrixValue(0, 0, R);
		Pn2.setSubMatrixValue(0, 0, R);
		Pn1.setColValue(3, SimpleOperators.multiply(R, c1).multipliedBy(-1));
		Pn2.setColValue(3, SimpleOperators.multiply(R, c2).multipliedBy(-1));

		Pn1 = SimpleOperators.multiplyMatrixProd(K1, Pn1);
		Pn2 = SimpleOperators.multiplyMatrixProd(K1, Pn2);

		SimpleMatrix Rect1 = SimpleOperators.multiplyMatrixProd(Pn1.getSubMatrix(0, 0, 3, 3),
				P1.getSubMatrix(0, 0, 3, 3).inverse(InversionType.INVERT_QR));
		SimpleMatrix Rect2 = SimpleOperators.multiplyMatrixProd(Pn2.getSubMatrix(0, 0, 3, 3),
				P2.getSubMatrix(0, 0, 3, 3).inverse(InversionType.INVERT_QR));

		p1 = SimpleOperators.concatenateVertically(p1, new SimpleVector(1.0));
		p2 = SimpleOperators.concatenateVertically(p2, new SimpleVector(1.0));
		//		System.out.println(p1  + "    " + p2);
		p1 = SimpleOperators.multiply(Rect1, p1);
		p2 = SimpleOperators.multiply(Rect2, p2);
		p1.divideBy(p1.getElement(2));
		p2.divideBy(p2.getElement(2));
		//		System.out.println(p1  + "    " + p2);

		P1 = Pn1;
		P2 = Pn2;
		if (compensate > 0) {
			p2.setElementValue(1, p1.getElement(1));
		}

		SimpleVector row1 = P1.getRow(2).multipliedBy(p1.getElement(0));
		row1.subtract(P1.getRow(0));
		SimpleVector row2 = P1.getRow(2).multipliedBy(p1.getElement(1));
		row2.subtract(P1.getRow(1));
		SimpleVector row3 = P2.getRow(2).multipliedBy(p2.getElement(0));
		row3.subtract(P2.getRow(0));
		SimpleVector row4 = P2.getRow(2).multipliedBy(p2.getElement(1));
		row4.subtract(P2.getRow(1));

		double weight1 = 1;
		double weight2 = 1;
		double oldWeight1 = 2;
		double oldWeight2 = 2;
		int iterations = 0;

		while ((Math.abs(1 / weight1 - 1 / oldWeight1) > 0.1 || Math.abs(1 / weight2 - 1 / oldWeight2) > 0.1)
				&& iterations < 10) {

			//Linear-Eigen solution to the triangulation problem by Hartley & Sturm
			//start Triangulation
			//setup Linear Equations x*p3^T = p1^T      y*p3^T = p2^T 
			SimpleMatrix A = new SimpleMatrix(4, 4);
			//first image
			row1.divideBy(weight1);
			row2.divideBy(weight1);
			//second image
			row3.divideBy(weight2);
			row4.divideBy(weight2);

			A.setRowValue(0, row1);
			A.setRowValue(1, row2);
			A.setRowValue(2, row3);
			A.setRowValue(3, row4);

			//solve with SVD
			Matrix B = new Matrix(A.copyAsDoubleArray());
			SingularValueDecomposition svd = new SingularValueDecomposition(B);
			Matrix V = svd.getV();

			//solution is nullspace of A -> last column of V
			sol = new SimpleVector(V.getColumnDimension());
			int lastrow = V.getRowDimension() - 1;

			for (int i = 0; i < sol.getLen(); i++) {
				sol.setElementValue(i, V.get(i, lastrow));
			}
			//dehomogenize
			sol.divideBy(sol.getElement(3));

			//adjust weights
			oldWeight1 = weight1;
			oldWeight2 = weight2;
			weight1 = SimpleOperators.multiplyInnerProd(P1.getRow(2), sol);
			weight2 = SimpleOperators.multiplyInnerProd(P2.getRow(2), sol);
			//			System.out.println(image1 + "iter: "+iterations +"  weights: "+ weight1);

			iterations++;
		}
		//end Triangulation

		return sol;
	}

	/**
	 * Wrapper for simple Triangulation
	 * @param projectionNumber index of projection
	 * @param compensate factor to compensate breathing motion
	 * @return 4-D Vector of 3-dimensional position
	 */
	private SimpleVector triangulationSimple(int projectionNumber, double compensate) {

		if (projectionNumber + offset > totalProjs - 1)
			offset *= -1;

		SimpleVector sol = triangulation(projectionNumber, projectionNumber + offset, compensate);
		if (projectionNumber == 0)
			pt3D = sol;
		return sol;

	}

	/**
	 * Wrapper for rectified iterative Triangulation
	 * @param projectionNumber index of projection
	 * @param compensate factor to compensate breathing motion
	 * @return 4-D Vector of 3-dimensional position
	 */
	private SimpleVector triangulationIterRectif(int projectionNumber, double compensate) {

		if (projectionNumber + offset > totalProjs - 1)
			offset *= -1;

		SimpleVector sol = rectifiedIterativeTriangulation(projectionNumber, projectionNumber + offset, compensate);
		if (projectionNumber == 0)
			pt3D = sol;
		return sol;

	}

	/**
	 * Wrapper for iterative Triangulation
	 * @param projectionNumber index of projection
	 * @param compensate factor to compensate breathing motion
	 * @return 4-D Vector of 3-dimensional position
	 */
	private SimpleVector triangulationIter(int projectionNumber, double compensate) {

		if (projectionNumber + offset > totalProjs - 1)
			offset *= -1;

		SimpleVector sol = iterativeTriangulation(projectionNumber, projectionNumber + offset, compensate);
		if (projectionNumber == 0)
			pt3D = sol;
		return sol;

	}

	/**
	 * This triangulation-method assumes that the point only moves along the superior-inferiors axis 
	 * and the other 2 coordinates remain constant. So only the height of the point at the given depth needs to be evaluated.
	 * The point in space is estimated as the average of triangulated points spread over the rotation
	 * @param projectionNumber index of projection
	 * @return 4-D Vector of 3-dimensional position
	 */
	private SimpleVector triangulationTrajectory(int projectionNumber) {

		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		SimpleVector sol = new SimpleVector(0.d, 0.d, 0.d, 0.d);

		//estimation of x,y-coordinates
		if (projectionNumber == 0) {
			int size = geom.getNumProjectionMatrices();
			int n = (int) (size / 50.0);
			int fac = size / (n - 1);
			for (int i = 0; i < size; i += fac) {
				sol.add(triangulationIterRectif(i, 1));
			}
			sol.divideBy(n);
			pt3D = sol.getSubVec(0, 3);
		}

		SimpleVector n = new SimpleVector(new double[] { 0, 0, 1 });
		SimpleVector q = geom.getProjectionMatrix(projectionNumber).computeCameraCenter();
		SimpleVector v = geom.getProjectionMatrix(projectionNumber).computeRayDirection(
				new SimpleVector(Configuration.getGlobalConfiguration().getDiaphragmCoordsEntry(projectionNumber)));
		SimpleVector w = crossProduct(n, v);
		SimpleVector z = q.clone();
		q.subtract(pt3D);
		SimpleVector a = crossProduct(q, v);
		double factor1 = SimpleOperators.multiplyInnerProd(a, w) / SimpleOperators.multiplyInnerProd(w, w);
		SimpleVector y = pt3D.clone();
		y.add(n.multipliedBy(factor1));
		SimpleVector b = crossProduct(q, n);
		double factor2 = SimpleOperators.multiplyInnerProd(b, w) / SimpleOperators.multiplyInnerProd(w, w);
		z.add(v.multipliedBy(factor2));

		sol = new SimpleVector(4);
		sol.setSubVecValue(0, pt3D);
		sol.setElementValue(2, z.getElement(2));
		sol.setElementValue(3, 1.0);

		return sol;
	}

	private SimpleVector crossProduct(SimpleVector v1, SimpleVector v2) {
		SimpleVector sol = new SimpleVector(v1.getLen());

		sol.setElementValue(0, v1.getElement(1) * v2.getElement(2) - v1.getElement(2) * v2.getElement(1));
		sol.setElementValue(1, v1.getElement(2) * v2.getElement(0) - v1.getElement(0) * v2.getElement(2));
		sol.setElementValue(2, v1.getElement(0) * v2.getElement(1) - v1.getElement(1) * v2.getElement(0));

		return sol;
	}

	@Override
	public ImageFilteringTool clone() {
		Triangulation clone = new Triangulation();
		clone.setContext(offset + 1);
		clone.setConfigured(configured);
		return clone;
	}

	@Override
	public String getToolName() {
		return "Triangulation";
	}

	@Override
	public boolean isDeviceDependent() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void configure() throws Exception {

		setContext(offset + 1);
		//setContext(200);
		double corr = (double) Double
				.parseDouble(JOptionPane.showInputDialog("Enter correction factor diaphragm <-> heart motion:", 1.0));
		Configuration.getGlobalConfiguration().setCorrectionFactor(corr);

		setConfigured(true);

	}

	/*
	 * 	A Master's thesis.
		Required fields: author, title, school, year
		Optional fields: type, address, month, note, key
	
	 */

	@Override
	public String getBibtexCitation() {
		return "@article{Boegel13-RMC,\n" + "  number={1},\n"
				+ "  author={Marco B{\"o}gel and Hannes Hofmann and Joachim Hornegger and Rebecca Fahrig and Stefan Britzen and Andreas Maier},\n"
				+ "  keywords={cardiac reconstruction; c-arm ct; motion compensation; diaphragm tracking},\n"
				+ "  url={http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2013/Boegel13-RMC.pdf},\n"
				+ "  doi={10.1155/2013/520540},\n" + "  journal={International Journal of Biomedical Imaging},\n"
				+ "  volume={2013},\n"
				+ "  title={{Respiratory Motion Compensation Using Diaphragm Tracking for Cone-Beam C-Arm CT: A Simulation and a Phantom Study}},\n"
				+ "  year={2013},\n" + "  pages={1--10}\n" + "}";
	}

	@Override
	public String getMedlineCitation() {
		return "Bögel M, Hofmann H, Hornegger J, Fahrig R, Britzen S, Maier A. Respiratory Motion Compensation Using Diaphragm Tracking for Cone-Beam C-Arm CT: A Simulation and a Phantom Study. International Journal of Biomedical Imaging, vol. 2013, no. 1, pp. 1-10, 2013 ";
	}

}
/*
 * Copyright (C) 2010-2014 - Marco Bögel 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/