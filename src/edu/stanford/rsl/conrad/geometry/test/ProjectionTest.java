package edu.stanford.rsl.conrad.geometry.test;

import static org.junit.Assert.*;

//import org.junit.After;
//import org.junit.AfterClass;
//import org.junit.Before;
//import org.junit.BeforeClass;
import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.Rotations.BasicAxis;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import static edu.stanford.rsl.conrad.utils.TestingTools.*;

public class ProjectionTest {
	
	// scaling parameter
	private static final double fs1 = randPositive();
	private static final double fs2 = randPositive();
	
	// intrinsic parameters
	private static final double ffocalU1 = rand(500.0, 1500.0); // px
	private static final double ffocalU2 = rand(500.0, 1500.0); // px
	private static final double ffocalV1 = rand(500.0, 1500.0); // px
	private static final double ffocalV2 = rand(500.0, 1500.0); // px
	private static final double fcameraToImage1 = rand(500.0, 1500.0); // mm
	private static final double fcameraToImage2 = rand(500.0, 1500.0); // mm
	private static final double fspacingU1 = fcameraToImage1 / ffocalU1; // mm/px
	private static final double fspacingU2 = fcameraToImage2 / ffocalU2; // mm/px
	private static final double fspacingV1 = fcameraToImage1 / ffocalV1; // mm/px
	private static final double fspacingV2 = fcameraToImage2 / ffocalV2; // mm/px
	private static final SimpleVector fspacing1 = new SimpleVector(fspacingU1, fspacingV1);
	private static final SimpleVector fspacing2 = new SimpleVector(fspacingU2, fspacingV2);
	private static final double fskew1 = rand(-0.5, 0.5); // px
	private static final double fskew2 = rand(-0.5, 0.5); // px
	private static final double fppU1 = rand(200.0, 600.0); // px
	private static final double fppU2 = rand(200.0, 600.0); // px
	private static final double fppV1 = rand(200.0, 600.0); // px
	private static final double fppV2 = rand(200.0, 600.0); // px
	private static final SimpleVector fpp1 = new SimpleVector(fppU1, fppV1);
	private static final SimpleVector fpp2 = new SimpleVector(fppU2, fppV2);
	private static final double fsizeU1 = rand(400.0, 1200.0); // px
	private static final double fsizeU2 = rand(400.0, 1200.0); // px
	private static final double fsizeV1 = rand(400.0, 1200.0); // px
	private static final double fsizeV2 = rand(400.0, 1200.0); // px
	private static final SimpleVector fsize1 = new SimpleVector(fsizeU1, fsizeV1);
	private static final SimpleVector fsize2 = new SimpleVector(fsizeU2, fsizeV2);
	private static final SimpleVector foffset1 = SimpleOperators.subtract(fpp1, fsize1.multipliedBy(0.5));
	private static final SimpleVector foffset2 = SimpleOperators.subtract(fpp2, fsize2.multipliedBy(0.5));
	private static final double fdir1 = randPmOne(); // +/-z_C
	private static final double fdir2 = randPmOne(); // +/-z_C
	private static final SimpleMatrix fK1 = new SimpleMatrix(new double[][] {
			{ffocalU1, fskew1,   fppU1*fdir1},
			{0.0,      ffocalV1, fppV1*fdir1},
			{0.0,      0.0,            fdir1}
		});
	private static final SimpleMatrix fK2 = new SimpleMatrix(new double[][] {
			{ffocalU2, fskew2,   fppU2*fdir2},
			{0.0,      ffocalV2, fppV2*fdir2},
			{0.0,      0.0,            fdir2}
		});
	
	// extrinsic parameters
	private static final double fcameraToOrigin1 = rand(0.0, 1.0) * fcameraToImage1; // [0, 1]*fcameraToImage
	private static final double fcameraToOrigin2 = rand(0.0, 1.0) * fcameraToImage2; // [0, 1]*fcameraToImage
	private static final SimpleVector ft1 = new SimpleVector(rand(-50.0, 50.0), rand(-50.0, 50.0), fdir1 * fcameraToOrigin1);
	private static final SimpleVector ft2 = new SimpleVector(rand(-50.0, 50.0), rand(-50.0, 50.0), fdir2 * fcameraToOrigin2);
	private static final SimpleMatrix tmpRx1 = Rotations.createBasicRotationMatrix(BasicAxis.X_AXIS, randAng());
	private static final SimpleMatrix tmpRx2 = Rotations.createBasicRotationMatrix(BasicAxis.X_AXIS, randAng());
	private static final SimpleMatrix tmpRy1 = Rotations.createBasicRotationMatrix(BasicAxis.Y_AXIS, randAng());
	private static final SimpleMatrix tmpRy2 = Rotations.createBasicRotationMatrix(BasicAxis.Y_AXIS, randAng());
	private static final SimpleMatrix tmpRz1 = Rotations.createBasicRotationMatrix(BasicAxis.Z_AXIS, randAng());
	private static final SimpleMatrix tmpRz2 = Rotations.createBasicRotationMatrix(BasicAxis.Z_AXIS, randAng());
	private static final SimpleMatrix fR1 = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(tmpRx1, tmpRy1), tmpRz1);
	private static final SimpleMatrix fR2 = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(tmpRx2, tmpRy2), tmpRz2);
	private static final SimpleMatrix fRt1 = General.createHomAffineMotionMatrix(fR1, ft1);
	private static final SimpleMatrix fRt2 = General.createHomAffineMotionMatrix(fR2, ft2);
	
	// complete matrix
	private static final SimpleMatrix fP1 = (SimpleOperators.multiplyMatrixProd(fK1, fRt1.getSubMatrix(0, 0, 3, 4))).multipliedBy(fs1);
	private static final SimpleMatrix fP2 = (SimpleOperators.multiplyMatrixProd(fK2, fRt2.getSubMatrix(0, 0, 3, 4))).multipliedBy(fs2);
	
	// other
	private static final double additionalScaling = 10.0*randPositive();
	
//	@BeforeClass
//	public static void setUpBeforeClass() throws Exception {
//	}

//	@AfterClass
//	public static void tearDownAfterClass() throws Exception {
//	}

//	@Before
//	public void setUp() throws Exception {
//	}

//	@After
//	public void tearDown() throws Exception {
//	}

	@Test
	public void testProjection() {
		Projection proj = new Projection();
		assertEquals(proj.getS(), 1.0, 0.0);
		final SimpleVector v0 = new SimpleVector(3);
		assertEqualElementWise(proj.getK(), SimpleMatrix.I_3, 0.0);
		assertEqualElementWise(proj.getR(), SimpleMatrix.I_3, 0.0);
		assertEqualElementWise(proj.getT(), v0, 0.0);
	}

	@Test
	public void testProjectionAbstractMatrix() {
		Projection proj = new Projection(fP1);
		assertEqualElementWise(proj.computeP(), fP1, DELTA);
	}

	@Test
	public void testProjectionDoubleArrayDoubleArrayIntArray() {
		Projection projtmp = new Projection(fP1);
		int[] glViewport = new int[] {(int)rand(-100.0, 100.0), (int)rand(-100.0, 100.0), (int)fsizeU1, (int)fsizeV1};
		double[] glProjectionGlVec = new double[16];
		double[] glModelviewGlVec = new double[16];
		projtmp.computeGLMatrices(glViewport[0], glViewport[1], glViewport[2], glViewport[3], rand(100.0, 400.0), rand(800.0, 1200.0), glProjectionGlVec, glModelviewGlVec); // s is not transferred to the OpenGL representation
		Projection proj = new Projection(glProjectionGlVec, glModelviewGlVec, glViewport);
		assertEquals(proj.getS(), 1.0, DELTA); // s cannot be recovered since it's not been transferred to the OpenGL representation
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test
	public void testSetToExampleCamera() {
		Projection proj = new Projection();
		proj.initToExampleCamera();
		final SimpleMatrix Ktmp = new SimpleMatrix(new double[][] {
				{1000.0,   0.0,  500.0},
				{   0.0, 1000.0, 500.0},
				{   0.0,    0.0,   1.0}
		});
		final SimpleVector ttmp = new SimpleVector(0.0, 0.0, 500.0);
		assertEquals(proj.getS(), 1.0, 0.0);
		assertEqualElementWise(proj.getK(), Ktmp, 0.0);
		assertEqualElementWise(proj.getR(), SimpleMatrix.I_3, 0.0);
		assertEqualElementWise(proj.getT(), ttmp, 0.0);
	}

	@Test
	public void testSetFromP() {
		Projection proj = new Projection();
		
		// test with valid matrix
		proj.initFromP(fP1.multipliedBy(additionalScaling));
		assertEquals(proj.getS(), fs1*additionalScaling, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
		proj.initFromP(fP2.multipliedBy(additionalScaling));
		assertEquals(proj.getS(), fs2*additionalScaling, DELTA);
		assertEqualElementWise(proj.getK(), fK2, DELTA);
		assertEqualElementWise(proj.getR(), fR2, DELTA);
		assertEqualElementWise(proj.getT(), ft2, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetFromPSingularMatrix() {
		Projection proj = new Projection();
		final SimpleMatrix fP_fail_sing = fP1.clone();
		fP_fail_sing.setSubRowValue(1, 0, fP_fail_sing.getSubRow(0, 0, 3));
		proj.initFromP(fP_fail_sing);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetFromPWrongSizeMatrix() {
		Projection proj = new Projection();
		final SimpleMatrix fP_fail_size = fP1.getSubMatrix(0, 0, 3, 3);
		proj.initFromP(fP_fail_size);
	}

	@Test
	public void testSetFromGL() {
		Projection projtmp = new Projection(fP2);
		int[] glViewport = new int[] {(int)rand(-100.0, 100.0), (int)rand(-100.0, 100.0), (int)fsizeU2, (int)fsizeV2};
		double[] glProjectionGlVec = new double[16];
		double[] glModelviewGlVec = new double[16];
		projtmp.computeGLMatrices(glViewport[0], glViewport[1], glViewport[2], glViewport[3], rand(100.0, 400.0), rand(800.0, 1200.0), glProjectionGlVec, glModelviewGlVec); // s is not transferred to the OpenGL representation
		Projection proj = new Projection();
		proj.initFromGL(glProjectionGlVec, glModelviewGlVec, glViewport);
		assertEquals(proj.getS(), 1.0, DELTA); // s cannot be recovered since it's not been transferred to the OpenGL representation
		assertEqualElementWise(proj.getK(), fK2, DELTA);
		assertEqualElementWise(proj.getR(), fR2, DELTA);
		assertEqualElementWise(proj.getT(), ft2, DELTA);
	}

	@Test
	public void testSetSKRT() {
		Projection proj = new Projection();
		proj.initFromSKRT(fs1*additionalScaling, fK1, fR1, ft1);
		assertEqualElementWise(proj.computeP(), fP1.multipliedBy(additionalScaling), DELTA);
		proj.initFromSKRT(fs2*additionalScaling, fK2, fR2, ft2);
		assertEqualElementWise(proj.computeP(), fP2.multipliedBy(additionalScaling), DELTA);
	}

	@Test
	public void testSetS() {
		Projection proj = new Projection(fP1);
		proj.setSValue(fs2);
		assertEquals(proj.getS(), fs2, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetSNonPositive() {
		Projection proj = new Projection(fP1);
		proj.setSValue(randNonPositive());
	}
	
	@Test
	public void testSetK() {
		Projection proj = new Projection(fP1);
		proj.setKValue(fK2);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK2, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetKWrongSize() {
		Projection proj = new Projection();
		SimpleMatrix K_wrongsize = fK1.getSubMatrix(0, 0, 3, 2);
		proj.setKValue(K_wrongsize);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKNonTriangular10() {
		Projection proj = new Projection();
		SimpleMatrix K_nontriang = fK1.clone();
		K_nontriang.setElementValue(1, 0, randNonZero());
		proj.setKValue(K_nontriang);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKNonTriangular20() {
		Projection proj = new Projection();
		SimpleMatrix K_nontriang = fK1.clone();
		K_nontriang.setElementValue(2, 0, randNonZero());
		proj.setKValue(K_nontriang);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKNonTriangular21() {
		Projection proj = new Projection();
		SimpleMatrix K_nontriang = fK1.clone();
		K_nontriang.setElementValue(2, 1, randNonZero());
		proj.setKValue(K_nontriang);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKWrongDiag0() {
		Projection proj = new Projection();
		SimpleMatrix K_wrongdiag = fK1.clone();
		K_wrongdiag.setElementValue(0, 0, randNonPositive());
		proj.setKValue(K_wrongdiag);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKWrongDiag1() {
		Projection proj = new Projection();
		SimpleMatrix K_wrongdiag = fK1.clone();
		K_wrongdiag.setElementValue(1, 1, randNonPositive());
		proj.setKValue(K_wrongdiag);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKWrongDiag2() {
		Projection proj = new Projection();
		SimpleMatrix K_wrongdiag = fK1.clone();
		K_wrongdiag.setElementValue(2, 2, randNotPmOne());
		proj.setKValue(K_wrongdiag);
	}
	
	@Test
	public void testSetR() {
		Projection proj = new Projection(fP1);
		proj.setRValue(fR2);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR2, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRWrongSize() {
		Projection proj = new Projection();
		SimpleMatrix R_wrongsize = fR1.getSubMatrix(0, 0, 3, 2);
		proj.setRValue(R_wrongsize);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRNonSOButO() {
		Projection proj = new Projection();
		SimpleMatrix R_nonSO = fR1.negated();
		proj.setRValue(R_nonSO);
	}

	@Test
	public void testSetT() {
		Projection proj = new Projection(fP1);
		proj.setTVector(ft2);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft2, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetTWrongSize() {
		Projection proj = new Projection();
		SimpleVector t_wrongsize = ft1.getSubVec(0, 2);
		proj.setTVector(t_wrongsize);
	}

	@Test
	public void testSetRt() {
		Projection proj = new Projection(fP1);
		proj.setRtValue(fRt2);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR2, DELTA);
		assertEqualElementWise(proj.getT(), ft2, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtNonSOButO() {
		Projection proj = new Projection();
		SimpleMatrix Rt_nonso = fRt1.clone();
		Rt_nonso.setSubMatrixValue(0, 0, Rt_nonso.getSubMatrix(0, 0, 3, 3).negated());
		proj.setRtValue(Rt_nonso);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetRtWrongLastRow0() {
		Projection proj = new Projection();
		SimpleMatrix Rt_wronglastrow = fRt1.clone();
		Rt_wronglastrow.setElementValue(3, 0, randNonZero());
		proj.setRtValue(Rt_wronglastrow);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetRtWrongLastRow1() {
		Projection proj = new Projection();
		SimpleMatrix Rt_wronglastrow = fRt1.clone();
		Rt_wronglastrow.setElementValue(3, 1, randNonZero());
		proj.setRtValue(Rt_wronglastrow);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetRtWrongLastRow2() {
		Projection proj = new Projection();
		SimpleMatrix Rt_wronglastrow = fRt1.clone();
		Rt_wronglastrow.setElementValue(3, 2, randNonZero());
		proj.setRtValue(Rt_wronglastrow);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetRtWrongLastRow3() {
		Projection proj = new Projection();
		SimpleMatrix Rt_wronglastrow = fRt1.clone();
		Rt_wronglastrow.setElementValue(3, 3, randNonZero() + 1.0);
		proj.setRtValue(Rt_wronglastrow);
	}
	
	@Test
	public void testSetPrincipalPoint() {
		Projection proj = new Projection(fP1);
		proj.setPrincipalPointValue(fpp2);
		assertEqualElementWise(proj.getPrincipalPoint(), fpp2, DELTA);
		proj.setPrincipalPointValue(fpp1);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetPrincipalPointWrongSize() {
		Projection proj = new Projection();
		SimpleVector pp_wrongsize = fpp1.getSubVec(0, 1);
		proj.setTVector(pp_wrongsize);
	}

	@Test
	public void testSetViewingDirection() {
		Projection proj = new Projection(fP1);
		proj.setViewingDirectionValue(fdir2);
		assertEquals(proj.getViewingDirection(), fdir2, DELTA);
		proj.setViewingDirectionValue(fdir1);
		assertEquals(proj.getS(), fs1, DELTA);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetViewingDirectionNotPmOne() {
		Projection proj = new Projection();
		proj.setViewingDirectionValue(randNotPmOne());
	}

	@Test
	public void testGetRt() {
		Projection proj = new Projection();
		proj.setRtValue(fRt1);
		assertEqualElementWise(proj.getRt(), fRt1, DELTA);
	}

	@Test
	public void testGetPrincipalPoint() {
		Projection proj = new Projection();
		proj.setPrincipalPointValue(fpp1);
		assertEqualElementWise(proj.getPrincipalPoint(), fpp1, DELTA);
	}

	@Test
	public void testGetViewingDirection() {
		Projection proj = new Projection();
		proj.setViewingDirectionValue(fdir1);
		assertEquals(proj.getViewingDirection(), fdir1, DELTA);
	}

	@Test
	public void testComputeP() {
		Projection proj = new Projection();
		proj.initFromSKRT(fs1, fK1, fR1, ft1);
		assertEqualElementWise(proj.computeP(), fP1, DELTA);
	}

	@Test
	public void testComputeCameraCenter() {
		Projection proj = new Projection(fP1.clone());
		SimpleVector C_hom = General.augmentToHomgeneous(proj.computeCameraCenter());
		SimpleVector zero4 = new SimpleVector(3);
		assertEqualElementWise(SimpleOperators.multiply(fP1, C_hom), zero4, DELTA);
	}

	@Test
	public void testComputePrincipalAxis() {
		Projection proj = new Projection(fP1.clone());
		SimpleVector pa = proj.computePrincipalAxis();
		SimpleVector C = proj.computeCameraCenter();
		double lambda = randPositive();
		assertTrue(
			General.areColinear(
				SimpleOperators.multiply(
					fP1, General.augmentToHomgeneous(SimpleOperators.add(C, pa.multipliedBy(lambda)))
				),
				General.augmentToHomgeneous(fpp1),
				100.0*DELTA
			)
		);
	}

	@Test
	public void testComputeRayDirection() {
		Projection proj = new Projection(fP1.clone());
		SimpleVector pixel = new SimpleVector(rand(0.0, fsizeU1), rand(0.0, fsizeU2));
		SimpleVector raydir = proj.computeRayDirection(pixel);
		SimpleVector C = proj.computeCameraCenter();
		SimpleVector X = SimpleOperators.add(C, raydir.multipliedBy(100.0 * randPositive()));
		SimpleVector X_hom = General.augmentToHomgeneous(X);
		SimpleVector lhs = SimpleOperators.multiply(fP1.clone(), X_hom).normalizedL2();
		SimpleVector rhs = General.augmentToHomgeneous(pixel).normalizedL2();
		assertEqualElementWise(lhs.normalizedL2(), rhs.normalizedL2(), DELTA);
	}

	@Test
	public void testComputeGLMatricesIntIntIntIntDoubleDoubleDoubleArrayDoubleArray() {
		Projection projtmp = new Projection(fP1);
		int[] glViewport = new int[] {(int)rand(-100.0, 100.0), (int)rand(-100.0, 100.0), (int)fsizeU1, (int)fsizeV1};
		double[] glProjectionGlVec = new double[16];
		double[] glModelviewGlVec = new double[16];
		projtmp.computeGLMatrices(glViewport[0], glViewport[1], glViewport[2], glViewport[3], rand(100.0, 400.0), rand(800.0, 1200.0), glProjectionGlVec, glModelviewGlVec); // s is not transferred to the OpenGL representation
		Projection proj = new Projection(glProjectionGlVec, glModelviewGlVec, glViewport);
		assertEquals(proj.getS(), 1.0, DELTA); // s cannot be recovered since it's not been transferred to the OpenGL representation
		assertEqualElementWise(proj.getK(), fK1, DELTA);
		assertEqualElementWise(proj.getR(), fR1, DELTA);
		assertEqualElementWise(proj.getT(), ft1, DELTA);
	}

	@Test
	public void testComputeGLMatricesIntIntIntIntAbstractVectorAbstractVectorDoubleArrayDoubleArray() {
		Projection projtmp = new Projection(fP2);
		int[] glViewport = new int[] {(int)rand(-100.0, 100.0), (int)rand(-100.0, 100.0), (int)fsizeU1, (int)fsizeV1};
		double[] glProjectionGlVec = new double[16];
		double[] glModelviewGlVec = new double[16];
		SimpleVector cubmin = new SimpleVector(rand(-200.0, 0.0), rand(-200.0, 0.0), rand(-200.0, 0.0));
		SimpleVector cubmax = new SimpleVector(rand(0.0, 200.0), rand(0.0, 200.0), rand(0.0, 200.0));
		projtmp.computeGLMatrices(glViewport[0], glViewport[1], glViewport[2], glViewport[3], cubmin, cubmax, glProjectionGlVec, glModelviewGlVec); // s is not transferred to the OpenGL representation
		Projection proj = new Projection(glProjectionGlVec, glModelviewGlVec, glViewport);
		assertEquals(proj.getS(), 1.0, DELTA); // s cannot be recovered since it's not been transferred to the OpenGL representation
		assertEqualElementWise(proj.getK(), fK2, DELTA);
		assertEqualElementWise(proj.getR(), fR2, DELTA);
		assertEqualElementWise(proj.getT(), ft2, DELTA);
	}

	@Test
	public void testComputeDepth() {
		Projection proj = new Projection(fP1);
		SimpleVector C = proj.computeCameraCenter();
		SimpleVector pa = proj.computePrincipalAxis();
		double dist = fcameraToImage1 * randPositive();
		SimpleVector v = SimpleOperators.add(C, pa.multipliedBy(dist));
		assertEquals(proj.computeDepth(v), dist, DELTA);
		v = SimpleOperators.add(C, pa.multipliedBy(-dist));
		assertEquals(proj.computeDepth(v), -dist, DELTA);
	}

	@Test
	public void testProject() {
		// initialize projection and test variables
		final Projection proj = new Projection(fP1);
		final SimpleVector pixel = new SimpleVector(rand(0.0, fsizeU1), rand(0.0, fsizeV1));
		final SimpleVector ray = proj.computeRayDirection(pixel);

		// construct a visible and an invisible point
		final SimpleVector C = proj.computeCameraCenter();
		final SimpleVector Xvis = SimpleOperators.add(C, ray.multipliedBy(fcameraToImage1*randPositive()));
		final SimpleVector Xnonvis = SimpleOperators.add(C, ray.multipliedBy(-fcameraToImage1*randPositive()));

		// test projecting both, the visible and the invisible, points
		final SimpleVector pixel_proj = new SimpleVector(2);
		double depth;
		depth = proj.project(Xvis, pixel_proj);
		assertEqualElementWise(pixel_proj, pixel, DELTA);
		assertTrue(depth > 0.0);
		assertEquals(proj.project(Xvis, pixel_proj), proj.computeDepth(Xvis), DELTA);
		depth = proj.project(Xnonvis, pixel_proj);
		assertEqualElementWise(pixel_proj, pixel, DELTA);
		assertTrue(depth < 0.0);
		assertEquals(proj.project(Xnonvis, pixel_proj), proj.computeDepth(Xnonvis), DELTA);
	}

	@Test
	public void testIntersectRayWithCuboid() {
		Projection proj = new Projection(fP1);
		final SimpleVector cubmin = new SimpleVector(rand(-200.0, 0.0), rand(-200.0, 0.0), rand(-200.0, 0.0));
		final SimpleVector cubmax = new SimpleVector(rand(0.0, 200.0), rand(0.0, 200.0), rand(0.0, 200.0));
		final SimpleVector p = new SimpleVector(rand(0.0, 1000.0), rand(0.0, 1000.0));
		final double[] tntf = new double[2];
		final SimpleVector C = new SimpleVector(3);
		final SimpleVector d = new SimpleVector(3);
		boolean cubeHit = proj.intersectRayWithCuboid(p, cubmin, cubmax, tntf, C, d);
		if (cubeHit) {
			final SimpleVector nearPoint = SimpleOperators.add(C, d.multipliedBy(tntf[0]));
			final SimpleVector pixelNear = new SimpleVector(2);
			double projectDepthNear = proj.project(nearPoint, pixelNear);
			final SimpleVector farPoint = SimpleOperators.add(C, d.multipliedBy(tntf[1]));
			final SimpleVector pixelFar = new SimpleVector(2);
			double projectDepthFar = proj.project(farPoint, pixelFar);
			assertEqualElementWise(p, pixelNear, DELTA);
			assertEqualElementWise(p, pixelFar, DELTA);
			assertEquals(tntf[0], projectDepthNear, DELTA);
			assertEquals(tntf[1], projectDepthFar, DELTA);
			assertTrue(tntf[1] > 0.0);
		}
	}

	@Test
	public void testSetKFromDistancesSpacingsSizeOffset() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, fsize1, foffset1, fdir1, fskew1);
		assertEqualElementWise(proj.getK(), fK1, DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetNonPosDist() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(randNonPositive()*fcameraToImage1, fspacing1, fsize1, foffset1, fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetWrongSize1() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, new SimpleVector(3), fsize1, foffset1, fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetNonPosSpacingU() {
		Projection proj = new Projection();
		SimpleVector fspacing1_fail = fspacing1.clone();
		fspacing1_fail.setElementValue(0, randNonPositive()*fspacing1.getElement(0));
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1_fail, fsize1, foffset1, fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetNonPosSpacingV() {
		Projection proj = new Projection();
		SimpleVector fspacing1_fail = fspacing1.clone();
		fspacing1_fail.setElementValue(1, randNonPositive()*fspacing1.getElement(1));
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1_fail, fsize1, foffset1, fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetWrongSize2() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, new SimpleVector(3), foffset1, fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetWrongSize3() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, fsize1, new SimpleVector(3), fdir1, fskew1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testSetKFromDistancesSpacingsSizeOffsetDirNotPmOne() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, fsize1, foffset1, randNotPmOne(), fskew1);
	}
	
	@Test
	public void testSetRtFromCircularTrajectory1() {
		// initialize parameters randomly
		final SimpleVector rotationCenter = new SimpleVector(rand(-100.0, 100.0), rand(-100.0, 100.0), rand(-100.0, 100.0));
		final SimpleVector rotationAxis = new SimpleVector(randNonZero(), randNonZero(), randNonZero());
		final double sourceToAxisDistance = 100.0*randPositive();

		// create centerToCameraAtZeroAngle vector and orthogonalize it w.r.t. the rotation axis
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(rand(-100.0, 100.0), rand(-100.0, 100.0), rand(-100.0, 100.0));
		SimpleVector rotationAxis_norm = rotationAxis.normalizedL2(); 
		centerToCameraAtZeroAngle = SimpleOperators.subtract(centerToCameraAtZeroAngle, rotationAxis_norm.multipliedBy(SimpleOperators.multiplyInnerProd(centerToCameraAtZeroAngle, rotationAxis_norm)));

		// initialize some more parameters randomly
		// this axis definition yields a +z viewing direction
		final CameraAxisDirection uDirection = CameraAxisDirection.DETECTORMOTION_PLUS;
		final CameraAxisDirection vDirection = CameraAxisDirection.ROTATIONAXIS_PLUS;
		final double rotationAngle = randAng();

		// set up the projection
		final Projection proj = new Projection();
		final double dir = proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
		proj.setKFromDistancesSpacingsSizeOffset(rand(1.5, 2.5)*sourceToAxisDistance, fspacing1, fsize1, foffset1, dir, fskew1);

		// test projecting the center point
		final SimpleVector pixel = new SimpleVector(2);
		double depth = proj.project(rotationCenter, pixel);
		assertEqualElementWise(pixel, fpp1, 10.0*DELTA);
		assertTrue(depth > 0.0);
		assertTrue(Math.abs(depth - sourceToAxisDistance) < DELTA);
		
		// test projecting a point on the detector (at its current angulation)
		SimpleVector v = centerToCameraAtZeroAngle.normalizedL2().negated().multipliedBy(sourceToAxisDistance);
		SimpleMatrix rot = Rotations.createRotationMatrixAboutAxis(rotationAxis, rotationAngle);
		v = SimpleOperators.add(SimpleOperators.multiply(rot, v), rotationCenter);
		depth = proj.project(v, pixel);
		assertEqualElementWise(pixel, fpp1, 10.0*DELTA);
		assertTrue(depth > 0.0);
		assertTrue(Math.abs(depth - 2.0*sourceToAxisDistance) < DELTA);
		
		// test projecting a point behind the camera
		v = centerToCameraAtZeroAngle.normalizedL2().multipliedBy(2.0*sourceToAxisDistance);
		rot = Rotations.createRotationMatrixAboutAxis(rotationAxis, rotationAngle);
		v = SimpleOperators.add(SimpleOperators.multiply(rot, v), rotationCenter);
		depth = proj.project(v, pixel);
		assertEqualElementWise(pixel, fpp1, 10.0*DELTA);
		assertTrue(depth < 0.0);
		assertTrue(Math.abs(depth + sourceToAxisDistance) < DELTA);
		
		// test projecting a point next to the camera center, displaced in +u direction
		final SimpleVector rotationAxisDirection = rotationAxis.normalizedL2();
		final SimpleVector detectorMotionDirection = General.crossProduct(centerToCameraAtZeroAngle.normalizedL2(), rotationAxisDirection);
		v = SimpleOperators.multiply(rot, detectorMotionDirection);
		v.add(rotationCenter);
		depth = proj.project(v, pixel);
		assertTrue(depth > 0.0);
		assertTrue(pixel.getElement(0) > fpp1.getElement(0));
		assertTrue(Math.abs(pixel.getElement(1) - fpp1.getElement(1)) < DELTA);
	}
	
	@Test
	public void testSetRtFromCircularTrajectory2() {
		// initialize parameters randomly
		final SimpleVector rotationCenter = new SimpleVector(rand(-100.0, 100.0), rand(-100.0, 100.0), rand(-100.0, 100.0));
		final SimpleVector rotationAxis = new SimpleVector(randNonZero(), randNonZero(), randNonZero());
		final double sourceToAxisDistance = 100.0*randPositive();
		
		// create centerToCameraAtZeroAngle vector and orthogonalize it w.r.t. the rotation axis
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(rand(-100.0, 100.0), rand(-100.0, 100.0), rand(-100.0, 100.0));
		final SimpleVector rotationAxis_norm = rotationAxis.normalizedL2(); 
		centerToCameraAtZeroAngle = SimpleOperators.subtract(centerToCameraAtZeroAngle, rotationAxis_norm.multipliedBy(SimpleOperators.multiplyInnerProd(centerToCameraAtZeroAngle, rotationAxis_norm)));

		// initialize some more parameters randomly
		// this axis definition yields a -z viewing direction
		final double rotationAngle = randAng();
		final CameraAxisDirection uDirection = CameraAxisDirection.ROTATIONAXIS_PLUS;
		final CameraAxisDirection vDirection = CameraAxisDirection.DETECTORMOTION_PLUS;

		// set up the projection
		final Projection proj = new Projection();
		final double dir = proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
		proj.setKFromDistancesSpacingsSizeOffset(rand(1.5, 2.5)*sourceToAxisDistance, fspacing2, fsize2, foffset2, dir, fskew2);
		
		// test projecting the center point
		final SimpleVector pixel = new SimpleVector(2);
		double depth = proj.project(rotationCenter, pixel);
		assertEqualElementWise(pixel, fpp2, 10.0*DELTA);
		assertTrue(depth > 0.0);
		assertTrue(Math.abs(depth - sourceToAxisDistance) < DELTA);
		
		// test projecting a point on the detector (at its current angulation)
		SimpleVector v = centerToCameraAtZeroAngle.normalizedL2().negated().multipliedBy(sourceToAxisDistance);
		SimpleMatrix rot = Rotations.createRotationMatrixAboutAxis(rotationAxis, rotationAngle);
		v = SimpleOperators.add(SimpleOperators.multiply(rot, v), rotationCenter);
		depth = proj.project(v, pixel);
		assertEqualElementWise(pixel, fpp2, 10.0*DELTA);
		assertTrue(depth > 0.0);
		assertTrue(Math.abs(depth - 2.0*sourceToAxisDistance) < DELTA);
		
		// test projecting a point behind the camera
		v = centerToCameraAtZeroAngle.normalizedL2().multipliedBy(2.0*sourceToAxisDistance);
		rot = Rotations.createRotationMatrixAboutAxis(rotationAxis, rotationAngle);
		v = SimpleOperators.add(SimpleOperators.multiply(rot, v), rotationCenter);
		depth = proj.project(v, pixel);
		assertEqualElementWise(pixel, fpp2, 10.0*DELTA);
		assertTrue(depth < 0.0);
		assertTrue(Math.abs(depth + sourceToAxisDistance) < DELTA);
		
		// test projecting a point next to the camera center, displaced in +u direction
		final SimpleVector rotationAxisDirection = rotationAxis.normalizedL2();
		v = SimpleOperators.multiply(rot, rotationAxisDirection);
		v.add(rotationCenter);
		depth = proj.project(v, pixel);
		assertTrue(depth > 0.0);
		assertTrue(pixel.getElement(0) > fpp2.getElement(0));
		assertTrue(Math.abs(pixel.getElement(1) - fpp2.getElement(1)) < DELTA);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtFromCircularTrajectoryRotationAxisNull() {
		Projection proj = new Projection();
		SimpleVector rotationCenter = new SimpleVector(0.0, 0.0, 0.0);
		SimpleVector rotationAxis = new SimpleVector(0.0, 0.0, 0.0);
		double sourceToAxisDistance = rand(300.0, 700.0);
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(0.0, 0.0, 500.0);
		CameraAxisDirection uDirection = CameraAxisDirection.DETECTORMOTION_MINUS;
		CameraAxisDirection vDirection = CameraAxisDirection.ROTATIONAXIS_MINUS;
		double rotationAngle = 0.0;
		proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtFromCircularTrajectoryNonPosRadius() {
		Projection proj = new Projection();
		SimpleVector rotationCenter = new SimpleVector(0.0, 0.0, 0.0);
		SimpleVector rotationAxis = new SimpleVector(1.0, 0.0, 0.0);
		double sourceToAxisDistance = 500.0*randNonPositive();
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(0.0, 0.0, 500.0);
		CameraAxisDirection uDirection = CameraAxisDirection.DETECTORMOTION_MINUS;
		CameraAxisDirection vDirection = CameraAxisDirection.ROTATIONAXIS_MINUS;
		double rotationAngle = 0.0;
		proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtFromCircularTrajectoryCenterToCameraAtZeroAngleNull() {
		Projection proj = new Projection();
		SimpleVector rotationCenter = new SimpleVector(0.0, 0.0, 0.0);
		SimpleVector rotationAxis = new SimpleVector(1.0, 0.0, 0.0);
		double sourceToAxisDistance = rand(300.0, 700.0);
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(0.0, 0.0, 0.0);
		CameraAxisDirection uDirection = CameraAxisDirection.DETECTORMOTION_MINUS;
		CameraAxisDirection vDirection = CameraAxisDirection.ROTATIONAXIS_MINUS;
		double rotationAngle = 0.0;
		proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtFromCircularTrajectoryUVDirection1() {
		Projection proj = new Projection();
		SimpleVector rotationCenter = new SimpleVector(0.0, 0.0, 0.0);
		SimpleVector rotationAxis = new SimpleVector(1.0, 0.0, 0.0);
		double sourceToAxisDistance = rand(300.0, 700.0);
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(0.0, 0.0, 500.0);
		CameraAxisDirection uDirection = CameraAxisDirection.DETECTORMOTION_MINUS;
		CameraAxisDirection vDirection = CameraAxisDirection.DETECTORMOTION_PLUS;
		double rotationAngle = 0.0;
		proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetRtFromCircularTrajectoryUVDirection2() {
		Projection proj = new Projection();
		SimpleVector rotationCenter = new SimpleVector(0.0, 0.0, 0.0);
		SimpleVector rotationAxis = new SimpleVector(1.0, 0.0, 0.0);
		double sourceToAxisDistance = rand(300.0, 700.0);
		SimpleVector centerToCameraAtZeroAngle = new SimpleVector(0.0, 0.0, 500.0);
		CameraAxisDirection uDirection = CameraAxisDirection.ROTATIONAXIS_PLUS;
		CameraAxisDirection vDirection = CameraAxisDirection.ROTATIONAXIS_MINUS;
		double rotationAngle = 0.0;
		proj.setRtFromCircularTrajectory(rotationCenter, rotationAxis, sourceToAxisDistance, centerToCameraAtZeroAngle, uDirection, vDirection, rotationAngle);
	}

	@Test
	public void testComputeSourceToDetectorDistance() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, fsize1, foffset1, fdir1, fskew1);
		double[] dist = proj.computeSourceToDetectorDistance(fspacing1);
		assertEquals(dist[0], fcameraToImage1, DELTA);
		assertEquals(dist[1], fcameraToImage1, DELTA);
	}

	@Test
	public void testComputeOffset() {
		Projection proj = new Projection();
		proj.setKFromDistancesSpacingsSizeOffset(fcameraToImage1, fspacing1, fsize1, foffset1, fdir1, fskew1);
		SimpleVector offset = proj.computeOffset(fsize1);
		assertEqualElementWise(offset, foffset1, DELTA);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/