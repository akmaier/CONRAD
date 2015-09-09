package edu.stanford.rsl.conrad.geometry;


import java.io.Serializable;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;


/**
 * This class represents a finite perspective projection (or a camera associated
 * with it) and provides all sorts of routines for converting between different
 * representations and obtaining geometric information.
 *
 * A finite projective camera can be minimally represented either by a 3x4 projection
 * matrix P or its decomposed version of intrinsic (K) and extrinsic (R and t) parameters.
 * OpenGL uses a 4x4 matrix which contains an additional third row with depth
 * information. All these three representations are supported by this class.
 * 
 * <p>The following text defines all variable names used in the member documentation and
 * explains the projection representations.
 * 
 * <p>It is recommended to refer to chapter 6 ("Camera Models") of
 * <a href="http://www.amazon.com/exec/obidos/ASIN/0521540518">[Hartley and Zisserman: Multiple View %Geometry. Cambridge Univ. Press]</a>
 * for further information.
 *
 *
 * <h1>Projection Matrix</h1>
 *
 * A 3x4 projection matrix P transforms a world coordinate point v (given in homogeneous
 * coordinates) into a image pixel coordinate p (also in homogeneous coordinates):
 * {@latex.ilb \\[
 *   \\mathbf{P} \\cdot \\mathbf{v} = \\mathbf{p}
 * \\]}
 * Since the resulting pixel vector is normalized before usage, a scaling of P does not
 * affect the resulting pixel. Therefore, we have 11 degrees of freedom (3x4 entries up
 * to scale). The world coordinates v as well as the image coordinates p are assumed to
 * represent infinitesimally small points in 3D or 2D, respectively. Therefore, if you
 * work with voxels or pixels of finite size, always use their center coordinates.
 *
 * {@latex.inline $\\mathbf{P}$} is mainly composed of two parts, the left 3x3 sub matrix {@latex.inline $\\mathbf{M}$}
 * and the right column {@latex.inline $\\mathbf{p_4}$}. It can also be further decomposed into the
 * intrinsic and extrinsic parameters as shown in the next section.
 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
 *   \\mathbf{P} & = \\left(\\begin{array}{c|c} \\mathbf{M} & \\mathbf{p_4} \\end{array}\\right) \\\\
 *          & = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)
 * \\end{align*}}
 * {@latex.ilb \\[
 *   \\Rightarrow \\quad \\mathbf{M} = s \\cdot \\mathbf{K} \\cdot \\mathbf{R} \\ \\wedge \\ \\mathbf{p_4} = s \\cdot \\mathbf{K} \\cdot \\mathbf{t}
 * \\]}
 *
 * <h2>Viewing Direction</h2>
 * However, such a projection matrix only defines the relation of voxels to pixels and
 * therefore the rays of the projection. We have no definition of "viewing direction"
 * yet. A real camera has a defined direction which it "looks at" and thus only images
 * points in front of the camera, clipping those behind it.
 * 
 * <p>This is usually defined by explicitly modeling a camera coordinate system and then
 * defining that the camera looks into positive or negative z direction (implying that
 * the other side is invisible). Then, the 3rd coordinate w of the projected point
 * (w*u, w*v, w) can be used to clip either points with a negative or positive depth
 * value w.
 * 
 * <p>In order to define the visible half space, we use the sign of the homogeneous
 * coordinate of p and choose the convention that visible points result in a positive
 * value {@latex.inline $p_3$} for a volume point given with a positive homogeneous component.
 * This choice is arbitrary. But since a choice has to be made anyways
 * and to avoid complicating the internal formulas, we have chosen this convention.
 * 
 * <p>If your P matrix returns a negative third coordinate for a voxel which which should
 * be visible, then just multiply the whole matrix by -1 before using it with this
 * class. It should then fulfill the condition {@latex.inline $\\mathbf{P}_3 \\cdot \\mathbf{v} > 0$}
 * for all voxels in front of the camera. Note that this negative scaling does not affect the
 * resulting pixel coordinate (due to the homogeneous division) but only flips the visibility
 * coordinate.
 * 
 * <p>Note that the 3x4 matrix does does not rely on a definition of a camera coordinate
 * system but merely is a direct mapping from world coordinates to pixels. Therefore,
 * there is also no notion of "viewing in +z or -z direction". For understanding the
 * structure and effect of 3x4 projection matrix (using an intermediate camera
 * coordinate system), you might want to look at the decomposition into intrinsic and
 * extrinsic parameters described in the following section.
 *
 *
 * <h1>Intrinsic and Extrinsic Parameters K/R/t</h1>
 *
 * A finite perspective projection P can also be represented using intrinsic (related
 * to the camera's internal geometry) and extrinsic (position of the camera in the world)
 * parameters:
 * {@latex.ilb \\[
 *   \\mathbf{P} = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)
 * \\]}
 * We will now give detailed explanations for the parameters mentioned above:
 * 
 * <ul>
 * <li> s is a positive scaling factor used to normalize K. A projective mapping to
 *      homogeneous pixel coordinates is defined up to scale and therefore this constant
 *      can be discarded without affecting the projection geometry. Note, however, that s
 *      may not be negative since a negation of the projection matrix would revert the
 *      viewing direction of the associated camera. (And s=0 is of course also off limits.)
 * <li> K is a 3x3 intrinsic camera matrix, which is upper-triangular with the positive
 *      focal lengths {@latex.inline $f_x$} and {@latex.inline $f_y$} (in pixels) on the main diagonal, a skew
 *      parameter {@latex.inline $\\alpha$} in the middle of the first row, and the principal point
 *      {@latex.inline %preamble{\\usepackage{amsmath}} $\\begin{pmatrix} p_u \\\\ p_v \\end{pmatrix}$} (given as homogeneous vector in
 *      pixels with {@latex.inline $\\pm 1$} scaling) in the last column:
 *      {@latex.ilb \\[
 *        \\mathbf{K} = \\left(\\begin{array}{ccc} f_x & \\alpha & \\pm p_u \\\\ 0 & f_y & \\pm p_v \\\\ 0 & 0 & \\pm 1 \\end{array}\\right)
 *      \\]}
 *      The rightmost column of K is only scaled with -1 in case you need to model
 *      an image where the u/v/depth axes build a left-handed coordinate system. Such a
 *      negated right column results from the fact that you may have to model the
 *      camera coordinate system with the z axis pointing towards the viewer (and therefore a
 *      -z viewing direction) in order to get the correct desired arrangement of the image axes
 *      u and v and a positive scaling between camera axes x/y and image axes u/v.
 *      In order to obtain positive depth values for the perspective division, the
 *      camera system's z axis has to be inverted after the
 *      {@latex.inline $\\left(\\begin{array}{c|c}\\mathbf{R} & \\mathbf{t}\\end{array}\\right)$}
 *      transformation. Written explicitly, a negative rightmost column of {@latex.inline $\\mathbf{K}$}
 *      results from mirroring the z axis (for obtaining positive depth values for visible points
 *      and thereby a left-handed u/v/z system) and then applying the usual projective camera:
 *      {@latex.ilb \\[
 *        \\underbrace{
 *        \\left(\\begin{array}{ccc} f_x & \\alpha & p_u \\\\ 0 & f_y & p_v \\\\ 0 & 0 & 1 \\end{array}\\right)
 *        \\cdot
 *        \\left(\\begin{array}{ccc} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & -1 \\end{array}\\right)
 *        }_{\\mathbf{K}}
 *        \\cdot
 *        \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)
 *      \\]}
 *      <p>The focal lengths {@latex.inline $f_x$} and {@latex.inline $f_y$} are measured in pixels if the
 *      K matrix is given in the form above. This implies that they should be equal for perfectly square
 *      pixels. The focal length is the distance from the camera center to the principal point on the image
 *      plane. If one further composes K into a camera matrix working in world coordinate dimensions only
 *      (mapping to world coordinates instead of pixels) and a separate scaling transform for translating
 *      image coordinates to pixel values, one obtains the additional information about the real focal length
 *      and the pixel size. These parameters, however, are not necessary for a pure establishment of the
 *      relation between volume and pixel coordinates. They are merely augmenting the projection description
 *      with additional geometric information. 
 *      <p>Finally, the skew parameter is usually zero, unless your camera's pixel axes are not
 *      perpendicular. For CCDs this means that they have not been produced with accurately aligned
 *      axes. A skew {@latex.inline $\\alpha \\neq 0$} can never arise by inclined arrangements of camera and
 *      detector.
 * <li> R is a 3x3 rotation matrix, representing the extrinsic rotation of the camera.
 * <li> t is a 3x1 translation vector, representing the translation of the camera.
 * </ul>
 * 
 * <p>The conversion to/from a 3x4 P matrix as described above is "lossless" since
 * any scaling factor is stored in s and the same assumptions are made concerning
 * the viewing direction (positive third coordinate for points in front of the
 * camera). Again, the world coordinates as well as the image coordinates are assumed
 * to represent infinitesimally small points in 3D or 2D, respectively. Therefore, if
 * you work with voxels or pixels of finite size, always use their center coordinates.
 * 
 * <p>The representation using intrinsic and extrinsic parameters is the best-suited for
 * further computations. Therefore, this representation is used internally for storing a
 * projection in this class.
 * 
 * <p>Note that the K/R/t representation always represents scaling effects with the K
 * matrix of intrinsic parameters. Only rotations and translations are stored in R and t.
 *
 *
 * <h1>OpenGL Projections</h1>
 *
 * This class interfaces with OpenGL's matrices via the methods fromGL() and
 * computeGLMatrices(), respectively.
 * 
 * <p>OpenGL has two so-called "matrix modes": GL_MODELVIEW and GL_PROJECTION. The modelview
 * matrix basically covers all transformations from object coordinates to eye coordinates
 * (the OpenGL term for camera coordinates). This includes translation, rotation, and also
 * scaling (e.g. of standard unit cubes to another size). The projection matrix then covers
 * the projective part. (However, OpenGL in the end just multiplies both matrices so that
 * the user may also "abuse" the two matrices.)
 * 
 * <p>The first difference to the intrinsic / extrinsic representation using K, R, and t is
 * therefore that scaling effects are modeled twice in OpenGL (in the modelview as well
 * as in the projection matrix) and only once using the K/R/t representation (namely in
 * K's focal lengths). This is due to OpenGL's concept of not using a world coordinate
 * system but standard objects of normalized size.
 * 
 * <p>Moreover, OpenGL uses homogeneous 4x4 matrices throughout the full transformation
 * pipeline. It separates depth from the z coordinate by adding an additional row to the
 * coordinates:
 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
 *   \\begin{pmatrix}
 *     -z \\cdot u \\\\ -z \\cdot v \\\\ -z \\cdot d \\\\ -z
 *   \\end{pmatrix}
 * \\]}
 * In the above term, d is a depth coordinate in and is separated from the scaling coordinate
 * -z (which is the negated z coordinate and therefore positive for visible points in OpenGL).
 * u, v, and d are the normalized device coordinates, each in the range [-1,1]. Therefore,
 * conversion from OpenGL to a minimal representation (P or K/R/t) is easily achieved by deleting
 * the third row (containing the additional depth information) and only keeping the 4th row (containing
 * the flipped scaling/z coordinate). Conversion in the other direction (K/R/t -> OpenGL) is
 * achieved by adding the missing depth row. This row can be reconstructed from the existing z
 * coordinate.
 * 
 * <p>As a last adjustment, it has to be noted that OpenGL uses a so-called "normalized
 * device coordinate system" which represents all pixels to be drawn in a
 * {@latex.inline $[-1, 1]^3$} cube (where the 3rd coordinate represents the normalized depth). The
 * transformation from these normalized coordinates to the screen pixel coordinates is
 * done using the viewport transform in OpenGL. This final transformation has to be appended
 * to a given OpenGL projection matrix in order to obtain the full transformation pipeline
 * from world coordinates to screen pixel coordinates. In contrast to the other projection
 * representations, OpenGL also defines 6 clipping planes (top, bottom, left, right, near,
 * and far). That's the reason why computeGLMatrices() has to know these 6 parameters
 * for a proper conversion. (fromGL() only needs the viewport for the correct mapping from
 * normalized device coordinates to window coordinates.)
 * 
 * <p>Summarizing, the following adjustments have to be made between the K/R/t and the OpenGL
 * representation:
 * 
 * <ul>
 * <li>Adjust for the viewport. (OpenGL gives/wants normalized device coordinates
 *     in {@latex.inline $[-1, 1]^3$} which it scales to the viewport internally while a
 *     representation in P or K/R/t directly yields window / image coordinates.)
 * <li>Adjust the z coordinate (negative for visible points in OpenGL, positive in the
 *     Projection class).
 * <li>Add or remove the depth row, respectively.
 * <li>Choose the correct front/back face definition. (It has to be reverted for +z viewing
 *     direction.)
 * </ul>
 * 
 * <p><em>Warning:</em> Mind that OpenGL uses column-major order while the NuDeLib uses row-major
 *          matrices. This class' OpenGL connections, however, interface with OpenGL using
 *          C-style arrays in column-major order so that you can directly load/get them
 *          in OpenGL.
 *
 * <p>See also <a href="http://www.amazon.com/exec/obidos/ASIN/0672326019">[Wright et al.: OpenGL SuperBible. Sams]</a>
 * and <a href="http://www.amazon.com/exec/obidos/ASIN/0321481003">[Shreiner et al.: OpenGL Programming Guide]</a>
 * for more information on OpenGL.
 *
 *
 * <h1>Additional Variables</h1>
 *
 * In addition to the variables defined in the preceding sections, we will also use
 * <ul>
 * <li> {@latex.inline $\\mathbf{C}$} for the 3x1 vector of the camera center given in (world coordinates),
 * <li> {@latex.inline $\\mathbf{a}$} for the (normalized) 3x1 vector of the principal axis (in world coordinates), see {@link #computePrincipalAxis()}, and
 * <li> {@latex.inline $\\mathbf{d}$} for a (normalized) 3x1 vector describing an arbitray ray direction (which corresponds to some pixel location)
 * </ul>
 * throughout the documentation of this class.
 *
 * @author Andreas Keil
 */
public class Projection implements Serializable {

	private static final long serialVersionUID = -6890230919386599781L;

	/** Scaling factor for the decomposed projection. */
	private double s;

	/** Intrinsic parameters for the decomposed projection. */
	private SimpleMatrix K;

	/** Rotation matrix for the decomposed projection. */
	private SimpleMatrix R;

	/** Translation vector for the decomposed projection. */
	private SimpleVector t;

	/** Store precomputed matrix {@latex.inline $\\mathbf{R}^T \\cdot \\mathbf{K}^{-1}$} to speed up computeRayDirection() */
	private SimpleMatrix RTKinv;


	/**
	 * Default constructor.
	 *
	 * The default constructor assumes a camera placed at the world origin viewing in
	 * +z direction with focal lengths both equal 1, principal point (0, 0) and no skew.
	 * It is the best setting to start from when setting up the camera step by step by,
	 * e.g. by concatenating transformations for computing the external setup.
	 */
	public Projection() {
		this.initFromSKRT(
				1.0,
				SimpleMatrix.I_3.clone(),
				SimpleMatrix.I_3.clone(),
				new SimpleVector(3)
		);
	}
	
	/**
	 * Copy constructor.
	 */
	public Projection(Projection in) {
		this.s = in.s;
		this.K = in.K.clone();
		this.R = in.R.clone();
		this.RTKinv = in.RTKinv.clone();
		this.t = in.t.clone();
	}

	/**
	 * Construct this projection from a 3x4 Matrix.
	 *
	 * @see #initFromP(SimpleMatrix P)
	 */
	public Projection(final SimpleMatrix P) {
		this.initFromP(P);
	}

	/**
	 * Construct this projection from GL matrices.
	 *
	 * @see #initFromGL(double[] glProjectionGlVec, double[] glModelviewGlVec, int[] glViewport)
	 */
	public Projection(final double[] glProjectionGlVec, final double[] glModelviewGlVec, final int[] glViewport) {
		this.initFromGL(glProjectionGlVec, glModelviewGlVec, glViewport);
	}

	/**
	 * Creates a exemplary perspective projection.
	 *
	 * The exemplary projection assumes a camera placed at (0mm, 0mm, -500mm) so that it views the world
	 * origin in +z direction, projecting the world's x and y axes to the images' u and v axes, resp. The
	 * focal lengths are 1000px (which corresponds to a 1m source-to-image distance for 1mm square pixels),
	 * the principal point is (500px, 500px) and no there is no skew. This setting should enable a visibility
	 * of cuboid volumes centered at the origin. E.g., assuming 1mm pixel resolution and projection images of
	 * 1000 by 1000 pixels (1m x 1m), cubes of up to 16 2/3 cm side length are fully visible in the
	 * projection.
	 * 
	 * <p><em>Warning:</em> This exemplary camera may be subject to change in future versions. Only use it
	 * when first experimenting with this class. Set up your own camera for any application.
	 */
	public void initToExampleCamera() {
		this.initFromSKRT(
				1.0,
				new SimpleMatrix(new double[][] {
						{1000.0,   0.0,  500.0},
						{   0.0, 1000.0, 500.0},
						{   0.0,    0.0,   1.0}
				}),
				SimpleMatrix.I_3.clone(),
				new SimpleVector(0.0, 0.0, 500.0)
		);
	}

	/**
	 * Define the projection using a 3x4 projection matrix.
	 *
	 * Internally, the given matrix is decomposed into intrinsic and extrinsic
	 * parameters as well as a positive scalar. See the class documentation for further
	 * details on the internal representation.
	 *
	 * @param P  A 3x4 projection matrix.
	 *        See the class documentation for further details.
	 *
	 * @see "This function uses decomposition.RQ internally in order to decompose the
	 *      matrix into intrinsic and extrinsic parameters and storing those internally."
	 */
	public void initFromP(final SimpleMatrix P) {
		// input check
		if (P.getRows() != 3 || P.getCols() != 4)
			throw new IllegalArgumentException("Error: P must be a 3x4 matrix but is a " + P.getRows() + "x" + P.getCols() + " matrix!");

		// decompose the 3x4 projection P into P = [M|p4]
		final SimpleMatrix M = P.getSubMatrix(0, 0, 3, 3);
		final SimpleVector p4 = P.getSubCol(0, 3, 3);

		// input check
		if (M.isSingular(Math.sqrt(CONRAD.DOUBLE_EPSILON)))
			throw new IllegalArgumentException("Given matrix is numerically singular!");

		// use a RQ decomposition for obtaining K and R; Attention: R may have determinant -1 which has to be corrected
		edu.stanford.rsl.conrad.numerics.DecompositionRQ rq = new edu.stanford.rsl.conrad.numerics.DecompositionRQ(M);
		SimpleMatrix Ktmp = rq.getR().clone();
		SimpleMatrix Rtmp = rq.getQ().clone();

		// ensure that R has determinant 1 and not -1 which would still be orthogonal but describing a rotation with reflection
		if (Rtmp.determinant() < 0.0) {
			Rtmp.negate();
			Ktmp.negate();
		}

		// extract scalar to normalize K
		double stmp = Math.abs(Ktmp.getElement(2, 2));
		if (stmp < CONRAD.DOUBLE_EPSILON)
			throw new IllegalArgumentException("Input projection matrix is not a projection matrix since the scaling factor s is (close to) zero! K seems to be singular.");
		Ktmp.divideBy(stmp);

		// ensure positive focal lengths by transferring minus signs to R (via double 180 degrees rotations between K and R)
		if (Ktmp.getElement(0, 0) < 0.0) {
			Ktmp.setColValue(0, Ktmp.getCol(0).negated());
			Rtmp.setRowValue(0, Rtmp.getRow(0).negated());
			Ktmp.setColValue(2, Ktmp.getCol(2).negated());
			Rtmp.setRowValue(2, Rtmp.getRow(2).negated());
		}
		if (Ktmp.getElement(1, 1) < 0) {
			Ktmp.setColValue(1, Ktmp.getCol(1).negated());
			Rtmp.setRowValue(1, Rtmp.getRow(1).negated());
			Ktmp.setColValue(2, Ktmp.getCol(2).negated());
			Rtmp.setRowValue(2, Rtmp.getRow(2).negated());
		}

		// compute the translation t = K^-1 * p4 / s (since P = s*K*[R|t] = [s*K*R|s*K*t] = [M|p4] => s*K*t = p4 => K*t = p4/s)
		SimpleVector ttmp = Solvers.solveUpperTriangular(Ktmp, p4.dividedBy(stmp));

		// set it all
		this.setSValue(stmp);
		this.setKValue(Ktmp);
		this.setRValue(Rtmp);
		this.setTVector(ttmp);
	}

	/**
	 * Gets all projection parameters from a given OpenGL projection matrix.
	 *
	 * This method defines the projection using a given OpenGL projection matrix. The given
	 * matrix may be either OpenGL's projection matrix only or the product of OpenGL's
	 * projection and modelview matrix:
	 *
	 * <p><em>Remark:</em> Usually, the following OpenGL commands have to be used for getting OpenGL's
	 *         projection:
	 * {@code
	 * double glProjectionGlVec[16];
	 * double glModelviewGlVec[16];
	 * int glViewport[4];
	 *
	 * glGetDoublev(GL_PROJECTION_MATRIX, glProjectionGlVec);
	 * glGetDoublev(GL_MODELVIEW_MATRIX, glModelviewGlVec);
	 * glGetIntegerv(GL_VIEWPORT, glViewport);
	 *
	 * Projection proj;
	 * proj.fromGL(glProjectionGlVec, glModelviewGlVec, glViewport);
	 * }
	 * 
	 * <p><em>Remark:</em> Internally, the following 3x4 matrix is computed using the conversion matrix
	 *         {@latex.inline %preamble{\\usepackage{amsmath}} ${}^\\text{P}T_\\text{GL}$} and the product {@latex.inline $\\mathbf{MVP}$} of the given
	 *         4x4 matrices:
	 *         {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 *           P = {}^\\text{P}T_\\text{GL} \\cdot \\mathbf{MVP}
	 *         \\]}
	 *         where
	 *         {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *           {}^\\text{P}T_\\text{GL} & =
	 *           \\begin{pmatrix} \\frac{s_u}{2} & 0 & \\frac{s_u-1}{2}+m_u \\\\ 0 & \\frac{s_v}{2} & \\frac{s_v-1}{2}+m_v \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *           \\cdot
	 *           \\begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\end{pmatrix} \\\\
	 *           & =
	 *           \\begin{pmatrix} 1 & 0 & m_u \\\\ 0 & 1 & m_v \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *           \\cdot
	 *           \\begin{pmatrix} s_u-1 & 0 & 0 \\\\ 0 & s_v-1 & 0 \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *           \\cdot
	 *           \\begin{pmatrix} \\frac{s_u}{2 \\cdot (s_u-1)} & 0 & 0 \\\\ 0 & \\frac{s_v}{2 \\cdot (s_v-1)} & 0 \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *           \\cdot
	 *           \\begin{pmatrix} 1 & 0 & 1-\\frac{1}{s_u} \\\\ 0 & 1 & 1-\\frac{1}{s_v} \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *           \\cdot
	 *           \\begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\end{pmatrix}
	 *         \\end{align*} }
	 *         This chain of transformations first deletes the third row of the given MVP
	 *         matrix, then converts the normalized device coordinates to screen coordinates,
	 *         and finally feeds the result into fromP().
	 *
	 * <p><em>Warning:</em> Mind that OpenGL uses C-style arrays in column-major order whereas the
	 *          NuDeLib uses row-major matrices. However, you shouldn't bother with that
	 *          since this methods takes glProjectionGlVec and glModelviewGlVec in a column-major
	 *          C-style array as you get it using OpenGL's <tt>glGetDoublev()</tt> function.
	 *
	 * @param glProjectionGlVec  A given OpenGL projection matrix. This is a C-style array of
	 *        16 values, representing a 4x4 matrix in column-major / OpenGL order.
	 * @param glModelviewGlVec   A given OpenGL modelview matrix. This is a C-style array of
	 *        16 values, representing a 4x4 matrix in column-major / OpenGL order.
	 * @param glViewport    A given OpenGL viewport specification in the form
	 *        [imgMinU, imgMinV, imgSizeU, imgSizeV], where
	 *        <ul>
	 *        <li> imgMinU is the minimum image index (in px) in the u direction (usually 0),
	 *        <li> imgMinV is the minimum image index (in px) in the v direction (usually 0),
	 *        <li> imgSizeU is the image width (in px) in u direction (imgSizeU = imgMaxU-imgMinU+1),
	 *        <li> imgSizeV is the image width (in px) in v direction (imgSizeV = imgMaxV-imgMinV+1).
	 *        </ul>
	 */
	public void initFromGL(final double[] glProjectionGlVec, final double[] glModelviewGlVec, final int[] glViewport) {
		// input checks
		if (glProjectionGlVec.length != 16)
			throw new IllegalArgumentException("Error: glProjectionGlVec must be a 16-vector but is a " + glProjectionGlVec.length + "-vector!");
		if (glModelviewGlVec.length != 16)
			throw new IllegalArgumentException("Error: glModelviewGlVec must be a 16-vector but is a " + glModelviewGlVec.length + "-vector!");
		if (glViewport.length != 4)
			throw new IllegalArgumentException("Error: glViewport must be a 4-vector but is a " + glViewport.length + "-vector!");
		if (glViewport[2] <= 0)
			throw new IllegalArgumentException("glViewport[2] must be positive but equals " + glViewport[2] + "!");
		if (glViewport[3] <= 0)
			throw new IllegalArgumentException("glViewport[3] must be positive but equals " + glViewport[3] + "!");

		// get column-major OpenGL matrices and convert them to row-major Nude matrices
		final SimpleMatrix glProjectionJava = new SimpleMatrix(4, 4);
		glProjectionJava.setElementValue(0, 0, glProjectionGlVec[0]);
		glProjectionJava.setElementValue(1, 0, glProjectionGlVec[1]);
		glProjectionJava.setElementValue(2, 0, glProjectionGlVec[2]);
		glProjectionJava.setElementValue(3, 0, glProjectionGlVec[3]);
		glProjectionJava.setElementValue(0, 1, glProjectionGlVec[4]);
		glProjectionJava.setElementValue(1, 1, glProjectionGlVec[5]);
		glProjectionJava.setElementValue(2, 1, glProjectionGlVec[6]);
		glProjectionJava.setElementValue(3, 1, glProjectionGlVec[7]);
		glProjectionJava.setElementValue(0, 2, glProjectionGlVec[8]);
		glProjectionJava.setElementValue(1, 2, glProjectionGlVec[9]);
		glProjectionJava.setElementValue(2, 2, glProjectionGlVec[10]);
		glProjectionJava.setElementValue(3, 2, glProjectionGlVec[11]);
		glProjectionJava.setElementValue(0, 3, glProjectionGlVec[12]);
		glProjectionJava.setElementValue(1, 3, glProjectionGlVec[13]);
		glProjectionJava.setElementValue(2, 3, glProjectionGlVec[14]);
		glProjectionJava.setElementValue(3, 3, glProjectionGlVec[15]);
		final SimpleMatrix glModelviewJava = new SimpleMatrix(4, 4);
		glModelviewJava.setElementValue(0, 0, glModelviewGlVec[0]);
		glModelviewJava.setElementValue(1, 0, glModelviewGlVec[1]);
		glModelviewJava.setElementValue(2, 0, glModelviewGlVec[2]);
		glModelviewJava.setElementValue(3, 0, glModelviewGlVec[3]);
		glModelviewJava.setElementValue(0, 1, glModelviewGlVec[4]);
		glModelviewJava.setElementValue(1, 1, glModelviewGlVec[5]);
		glModelviewJava.setElementValue(2, 1, glModelviewGlVec[6]);
		glModelviewJava.setElementValue(3, 1, glModelviewGlVec[7]);
		glModelviewJava.setElementValue(0, 2, glModelviewGlVec[8]);
		glModelviewJava.setElementValue(1, 2, glModelviewGlVec[9]);
		glModelviewJava.setElementValue(2, 2, glModelviewGlVec[10]);
		glModelviewJava.setElementValue(3, 2, glModelviewGlVec[11]);
		glModelviewJava.setElementValue(0, 3, glModelviewGlVec[12]);
		glModelviewJava.setElementValue(1, 3, glModelviewGlVec[13]);
		glModelviewJava.setElementValue(2, 3, glModelviewGlVec[14]);
		glModelviewJava.setElementValue(3, 3, glModelviewGlVec[15]);
		final int imgMinU = glViewport[0];
		final int imgMinV = glViewport[1];
		final int imgSizeU = glViewport[2];
		final int imgSizeV = glViewport[3];

		// issue warning in case the modelview matrix is not just a simple rigid motion matrix
		if (!glModelviewJava.isRigidMotion3D(Math.sqrt(CONRAD.DOUBLE_EPSILON))) {
			System.out.println("Warning in " + this.getClass().getName() + "::fromGL(): The given modelview matrix is not a rigid motion matrix!");
			System.out.println("The projection represented by the " + this.getClass().getName() + " will yield the same results.");
			System.out.println("However, matrices returned by " + this.getClass().getName() + "::computeGLMatrices() will differ from the originally given OpenGL matrices.");
		}

		// create the full OpenGL pipeline matrix
		final SimpleMatrix glModelviewProjectionJava = SimpleOperators.multiplyMatrixProd(glProjectionJava, glModelviewJava);

		// delete third row which is used by OpenGL for depth clipping
		// but keep the 4th row which contains the flipped z coordinate
		// (which is then positive for visible points)
		final SimpleMatrix glProjJava3x4 = new SimpleMatrix(3, 4);
		glProjJava3x4.setRowValue(0, glModelviewProjectionJava.getRow(0)); // u
		glProjJava3x4.setRowValue(1, glModelviewProjectionJava.getRow(1)); // v
		glProjJava3x4.setRowValue(2, glModelviewProjectionJava.getRow(3)); // flipped w = flipped z => positive for visible points

		// transformation from normalized device coordinates [-1+1/w, 1-1/w] x [-1+1/h, 1-1/h]
		// to window coordinates [imgMinU, imgMinU+imgSizeU] x [imgMinV, imgMinV+imgSizeV];
		// this is the the viewport transform (keeping in mind that OpenGL floors the floating
		// point pixel coordinates, resulting in the pixel centers from [-1+1/w, 1-1/w]*[-1+1/h, 1-1/h]
		final SimpleMatrix normalized2Image = new SimpleMatrix(new double[][]{
				{imgSizeU/2.0, 0.0, (imgSizeU-1)/2.0+imgMinU},
				{0.0, imgSizeV/2.0, (imgSizeV-1)/2.0+imgMinV},
				{0.0, 0.0, 1.0}
		});

		// set the final 3x4 projection matrix
		this.initFromP(SimpleOperators.multiplyMatrixProd(normalized2Image, glProjJava3x4));
	}

	/**
	 * Set the projection's intrinsic and extrinsic parameters all at once.
	 *
	 * @param s  A positive scaling factor.
	 *        See the class documentation for further details.
	 * @param K  A 3x3 intrinsic camera matrix.
	 *        See the class documentation for further details.
	 * @param R  A 3x3 rotation matrix, representing the extrinsic rotation of the camera.
	 *        See the class documentation for further details.
	 * @param t  A 3x1 translation vector, representing the translation of the camera.
	 *        See the class documentation for further details.
	 *
	 * @see #setSValue(double)
	 * @see #setKValue(SimpleMatrix)
	 * @see #setRValue(SimpleMatrix R)
	 * @see #setTVector(SimpleVector)
	 */
	public void initFromSKRT(final double s, final SimpleMatrix K, final SimpleMatrix R, final SimpleVector t) {
		this.setSValue(s);
		this.setKValue(K);
		this.setRValue(R);
		this.setTVector(t);
	}

	/**
	 * Set the (positive) scaling of the projection.
	 *
	 * @param s  A positive scaling factor.
	 *        See the class documentation for further details.
	 *
	 * @see #initFromSKRT(double s, SimpleMatrix K, SimpleMatrix R, SimpleVector t)
	 */
	public void setSValue(final double s) {
		// input checks
		if (s <= CONRAD.DOUBLE_EPSILON) throw new IllegalArgumentException("Error: s should be a positive scalar but has the value " + s + "!");

		// set given value
		this.s = s;
	}

	/**
	 * Set the intrinsic parameters K of the projection.
	 *
	 * @param K  A 3x3 intrinsic camera matrix.
	 *        See the class documentation for further details.
	 *
	 * @see #initFromSKRT(double s, SimpleMatrix K, SimpleMatrix R, SimpleVector t)
	 */
	public void setKValue(final SimpleMatrix K) {
		// input checks
		if (K.getRows() != 3 || K.getCols() != 3 || K.getElement(0, 0) <= 0.0 || K.getElement(1, 1) <= 0.0 || !K.isUpperTriangular() || Math.abs(Math.abs(K.getElement(2, 2)) - 1.0) > Math.sqrt(CONRAD.DOUBLE_EPSILON))
			throw new IllegalArgumentException("Error: The supplied K matrix has to be a 3x3 upper-triangular matrix with positive focal lengths and normalized to K(2, 2) = +/-1 but it equals " + K);

		// set given value
		this.K = K.clone();

		// update precomputed matrices
		this.updateRTKinv();
	}

	/**
	 * Set the rotation part of the extrinsic parameters of the projection.
	 *
	 * @param R  A 3x3 rotation matrix, representing the extrinsic rotation of the camera.
	 *        See the class documentation for further details.
	 *
	 * @see #initFromSKRT(double s, SimpleMatrix K, SimpleMatrix R, SimpleVector t)
	 */
	public void setRValue(final SimpleMatrix R) {
		// input checks
		if (!R.isRotation3D(Math.sqrt(CONRAD.DOUBLE_EPSILON))) throw new IllegalArgumentException("Error: R must be a 3x3 rotation matrix but equals " + R);

		// set given value
		this.R = R.clone();

		// update precomputed matrices
		this.updateRTKinv();
	}

	/**
	 * Set the translation part of the extrinsic parameters of the projection.
	 *
	 * @param t  A 3x1 translation vector, representing the translation of the camera.
	 *        See the class documentation for further details.
	 *
	 * @see #initFromSKRT(double s, SimpleMatrix K, SimpleMatrix R, SimpleVector t)
	 */
	public void setTVector(final SimpleVector t) {
		// input checks
		if (t.getLen() != 3) throw new IllegalArgumentException("Error: t must be a 3-vector but equals " + t);

		// set given value
		this.t = t.clone();
	}

	/**
	 * Set the extrinsic parameters of the projection.
	 *
	 * @param Rt  A homogeneous 4x4 rigid motion matrix, representing the extrinsic rotation
	 *        and translation of the camera.
	 *        See the class documentation for further details.
	 *
	 * @see #getRt()
	 */
	public void setRtValue(final SimpleMatrix Rt) {
		// input checks
		if (!Rt.isRigidMotion3D(Math.sqrt(CONRAD.DOUBLE_EPSILON))) throw new IllegalArgumentException("Error: Rt must be a homogeneous 4x4 rigid motion matrix but equals " + Rt);

		// set given value
		SimpleMatrix Rt_normalized = Rt.dividedBy(Rt.getElement(3, 3));
		this.setRValue(Rt_normalized.getSubMatrix(0, 0, 3, 3));
		this.setTVector(Rt_normalized.getSubCol(0, 3, 3));
	}

	/**
	 * Sets the principal point in pixels.
	 *
	 * The principal point is the pixel onto which every voxel on the principal axis
	 * gets mapped by the projection. It is stored in the last column of K. This last
	 * column is a homogeneous vector with a scaling of +1 for a +z viewing direction
	 * and a scaling of -1 for -z viewing direction.
	 *
	 * @param p  The principal point {@latex.inline %preamble{\\usepackage{amsmath}} $\\begin{pmatrix} p_u \\\\ p_v \\end{pmatrix}$} in
	 *        image/pixel coordinates.
	 */
	public void setPrincipalPointValue(final SimpleVector p) {
		// input check
		if (p.getLen() != 2) throw new IllegalArgumentException("The principal point has to be a 2-vector but " + p + " was given instead!");

		// modify K
		this.K.setSubColValue(0, 2, p.multipliedBy(this.getViewingDirection()));

		// update precomputed matrices
		this.updateRTKinv();
	}

	/**
	 * Set the viewing direction of the camera with respect to the z
	 * axis of the camera coordinate system.
	 *
	 * The viewing direction can be either in positive or negative
	 * {@latex.inline %preamble{\\usepackage{amsmath}} $z_\\text{C}$} direction.
	 *
	 * @param dir  +1.0 for a camera looking in {@latex.inline %preamble{\\usepackage{amsmath}} $+z_\\text{C}$}
	 *             direction or -1.0 for a camera looking in
	 *             {@latex.inline %preamble{\\usepackage{amsmath}} $-z_\\text{C}$} direction.
	 */
	public void setViewingDirectionValue(final double dir) {
		// input check
		if (Math.abs(Math.abs(dir) - 1.0) > Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Error: dir has to be +/-1 but is " + dir + "!");

		// modify K
		this.K.setColValue(2, this.getPrincipalPoint().multipliedBy(dir));
		this.K.setElementValue(2, 2, dir);

		// update precomputed matrices
		this.updateRTKinv();
	}

	/**
	 * Returns a const reference to the scaling.
	 *
	 * Note that a projection matrix is defined up to (positive) scale. Therefore, this
	 * scaling factor is only of interest in case you want to exactly restore a given
	 * 3x4 projection matrix.
	 *
	 * @return The positive scaling factor s.
	 *         See the class documentation for further details.
	 *
	 * @see #setSValue(double s)
	 */
	public double getS() {
		return this.s;
	}

	/**
	 * Returns a const reference to the K matrix of intrinsic parameters.
	 *
	 * @return The upper-triangular 3x3 matrix K of intrinsic parameters.
	 *         See the class documentation for further details.
	 *
	 * @see #setKValue(SimpleMatrix K)
	 */
	public SimpleMatrix getK() {
		return this.K.clone();
	}

	/**
	 * Returns a const reference to the rotation matrix R.
	 *
	 * <p><em>Remark:</em> If you want to obtain the rigid motion matrix
	 *         {@latex.inline $\\left(\\begin{array}{cc} \\mathbf{R} & \\mathbf{t} \\\\ \\mathbf{0} & 1 \\end{array}\\right) $}
	 *         use createHomRigidMotionMatrix(const Matrix<T, S1>& R, const Vector<T, S2>& t).
	 * {@code
	 * Matrix<T> Rt = createHomRigidMotionMatrix(R, t);
	 * }
	 *
	 * @return The 3x3 rotation matrix R, representing the extrinsic rotation of the camera.
	 *         See the class documentation for further details.
	 *
	 * @see #setRValue(SimpleMatrix R)
	 * @see #getRt()
	 */
	public SimpleMatrix getR() {
		return this.R.clone();
	}

	/**
	 * Returns a const reference to the translation vector t.
	 *
	 * @return The 3x1 translation vector t, representing the extrinsic translation of the camera.
	 *         See the class documentation for further details.
	 *
	 * @see #setTVector(SimpleVector t)
	 * @see #getRt()
	 */
	public SimpleVector getT() {
		return this.t.clone();
	}

	/**
	 * Returns all extrinsic parameters (R and t) in a homogeneous rigid motion matrix.
	 *
	 * @return The homogeneous 4x4 motion matrix
	 *         {@latex.inline %preamble{\\usepackage{amsmath}} $\\begin{pmatrix} \\mathbf{R} & \\mathbf{t} \\\\ \\mathbf{0} & 1 \\end{pmatrix} $}.
	 *         See the class documentation for further details.
	 *
	 * @see #getR()
	 * @see #getT()
	 * @see #setRValue(SimpleMatrix R)
	 * @see #setTVector(SimpleVector t)
	 */
	public SimpleMatrix getRt() {
		return General.createHomAffineMotionMatrix(this.R, this.t);
	}

	/**
	 * Returns the principal point in pixels.
	 *
	 * The principal point is the pixel onto which every voxel on the principal axis
	 * gets mapped by the projection. It can be read out from the last column of K
	 * (after normalization).
	 *
	 * @return The principal point {@latex.inline %preamble{\\usepackage{amsmath}} $\\begin{pmatrix} p_u \\\\ p_v \\end{pmatrix}$} in
	 *         image/pixel coordinates.
	 */
	public SimpleVector getPrincipalPoint() {
		return this.K.getSubCol(0, 2, 2).dividedBy(this.getViewingDirection());
	}

	/**
	 * Returns the viewing direction of the camera with respect to
	 * the z axis of the camera coordinate system
	 *
	 * The viewing direction can be either in positive or negative
	 * {@latex.inline %preamble{\\usepackage{amsmath}} $z_\\text{C}$} direction.
	 *
	 * @return +1.0 for a camera looking in {@latex.inline %preamble{\\usepackage{amsmath}} $+z_\\text{C}$}
	 *         direction or -1.0 for a camera looking in
	 *         {@latex.inline %preamble{\\usepackage{amsmath}} $-z_\\text{C}$} direction.
	 */
	public double getViewingDirection() {
		return this.K.getElement(2, 2);
	}

	/**
	 * Computes the 3x4 projection matrix
	 * {@latex.inline $\\mathbf{P} = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)$}
	 * and returns it.
	 *
	 * @return The 3x4 projection matrix P defined by this object.
	 *         See the class documentation for further details.
	 *
	 * @see #initFromP(SimpleMatrix P)
	 */
	public SimpleMatrix computeP() {
		final SimpleMatrix P = new SimpleMatrix(3, 4);
		final SimpleMatrix sK = this.K.multipliedBy(this.s);
		P.setSubMatrixValue(0, 0, SimpleOperators.multiplyMatrixProd(sK, this.R));
		P.setColValue(3, SimpleOperators.multiply(sK, this.t));
		return P;
	}

	/**
	 * Computes the 3x4 projection matrix
	 * {@latex.inline $\\mathbf{P} = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)$}
	 * and returns it.
	 * 
	 * Some calibration algorithms return a projection matrix with non unique scaling.
	 * Here a scaling is applied such that the homogeneous coordinate of the projected point contains the depth in [mm]
	 *
	 * 
	 * @return The 3x4 projection matrix P defined by this object.
	 *         See the class documentation for further details.
	 *
	 * @see #initFromP(SimpleMatrix P)
	 */
	public SimpleMatrix computePMetric(double sourceToAxisDistance) {
		final SimpleMatrix P = new SimpleMatrix(3, 4);
		final SimpleMatrix sK = this.K.multipliedBy(this.s);
		P.setSubMatrixValue(0, 0, SimpleOperators.multiplyMatrixProd(sK, this.R));
		P.setColValue(3, SimpleOperators.multiply(sK, this.t));

		SimpleVector origin = new SimpleVector(0,0,0,1);
		SimpleVector projected = SimpleOperators.multiply(P, origin);
		//boolean negative = projected.getElement(2) < 0;
		double scaling = Math.abs(sourceToAxisDistance / projected.getElement(2));
		//System.out.println("Scaling"  + (scaling) +  "orig: " + projected );
		return P.multipliedBy(scaling);

	}

	/**
	 * Computes whether the depth values that are computed from this projection are positive or negative.
	 */
	public boolean negativeDepth() {
		final SimpleMatrix P = new SimpleMatrix(3, 4);
		final SimpleMatrix sK = this.K.multipliedBy(this.s);
		P.setSubMatrixValue(0, 0, SimpleOperators.multiplyMatrixProd(sK, this.R));
		P.setColValue(3, SimpleOperators.multiply(sK, this.t));

		SimpleVector origin = new SimpleVector(0,0,0,1);
		SimpleVector projected = SimpleOperators.multiply(P, origin);
		return  projected.getElement(2) < 0;
	}

	/**
	 * Computes the 3x4 projection matrix
	 * {@latex.inline $\\mathbf{P} = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)$}
	 * and returns it.
	 * <BR>
	 * Some calibration algorithms return a projection matrix with non unique scaling.<br>
	 * Here a scaling is applied such that the homogeneous coordinate of the projected point contains the depth in [mm]<br>
	 * <br>
	 * Source to Axis distance is taken from global configuration.
	 *
	 * @return The 3x4 projection matrix P defined by this object.
	 *         See the class documentation for further details.
	 *
	 * @see #initFromP(SimpleMatrix P)
	 */
	public SimpleMatrix computePMetric() {
		return computePMetric(Configuration.getGlobalConfiguration().getGeometry().getSourceToAxisDistance());
	}
	
	/**
	 * Computes the 3x4 projection matrix
	 * {@latex.inline $\\mathbf{P} = s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right)$}
	 * and returns it.
	 * <BR>
	 * Some calibration algorithms return a projection matrix with non unique scaling.<br>
	 * Here a scaling is applied such that the homogeneous coordinate of the projected point contains the depth in [mm]<br>
	 * <br>
	 * @param Source to Axis distance
	 *
	 * @return The 3x4 projection matrix P defined by this object.
	 *         See the class documentation for further details.
	 *
	 * @see #initFromP(SimpleMatrix P)
	 */
	public SimpleMatrix computePMetricKnownSourceToAxis(double sourceToAxisDistance) {
		return computePMetric(sourceToAxisDistance);
	}

	/**
	 * Compute the camera center in world coordinates.
	 *
	 * Formulas for the camera center C are derived from the condition
	 * {@latex.inline %preamble{\\usepackage{amsmath}} $\\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} = 0$} (since the
	 * camera center is the only point mapped to a "pixel at infinity"):
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} & = 0 \\\\
	 *   s \\cdot \\mathbf{K} \\cdot \\left(\\begin{array}{c|c} \\mathbf{R} & \\mathbf{t} \\end{array}\\right) \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} & = 0 \\\\
	 *   \\mathbf{K} \\cdot (\\mathbf{R} \\cdot \\mathbf{C} + \\mathbf{t}) & = 0 \\\\
	 *   \\mathbf{R} \\cdot \\mathbf{C} & = -\\mathbf{t} \\\\
	 *   \\mathbf{C} & = -\\mathbf{R}^\\mathsf{T} \\cdot \\mathbf{t}
	 * \\end{align*}}
	 * An alternative formula is derived as follows:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} & = 0 \\\\
	 *   \\left(\\begin{array}{c|c} \\mathbf{M} & \\mathbf{p_4} \\end{array}\\right) \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} & = 0 \\\\
	 *   \\mathbf{M} \\cdot \\mathbf{C} + \\mathbf{p_4} & = 0 \\\\
	 *   \\mathbf{M} \\cdot \\mathbf{C} & = -\\mathbf{p_4} \\\\
	 *   \\mathbf{C} & = -\\mathbf{M}^{-1} \\cdot \\mathbf{p_4}
	 * \\end{align*}}
	 *
	 * @return The camera center in world coordinates.
	 */
	public SimpleVector computeCameraCenter() {
		return SimpleOperators.multiply(this.R.transposed(), this.t).negated();
	}

	/**
	 * Compute the principal axis direction in world coordinates.
	 *
	 * Formulas for the normalized direction vector of principal axis are derived from the condition
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 * 	\\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} + \\lambda_1 \\cdot \\mathbf{a} \\\\ 1 \\end{pmatrix} = \\lambda_2 \\cdot \\mathbf{k_3} / k_{33}
	 * \\]}
	 * that any point {@latex.inline $\\mathbf{C} + \\lambda_1 \\cdot \\mathbf{a}$} on the principal axis gets mapped to the principal point {@latex.inline $\\mathbf{k_3} / k_{33}$}
	 * (with {@latex.inline $\\lambda_1,\\lambda_2 > 0$} being scalar factors, {@latex.inline $\\mathbf{k_3}$} being the 3rd column of {@latex.inline $\\mathbf{K}$}, and
	 * {@latex.inline $k_{33} = \\pm 1$} the lower-right entry of {@latex.inline $\\mathbf{K}$}).
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} + \\lambda_1 \\cdot \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{a} \\\\ 0 \\end{pmatrix} & = \\lambda_2 \\cdot \\mathbf{k_3} / k_{33} \\\\
	 *   \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{C} \\\\ 1 \\end{pmatrix} = 0 \\quad \\Rightarrow \\quad \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{a} \\\\ 0 \\end{pmatrix} & = \\frac{\\lambda_2}{\\lambda_1 \\cdot k_{33}} \\cdot \\mathbf{k_3} \\\\
	 *   s \\cdot \\mathbf{K} \\cdot \\left( \\begin{array}{c|c} \\mathbf{R} & t \\end{array} \\right) \\cdot \\begin{pmatrix} \\mathbf{a} \\\\ 0 \\end{pmatrix} & = \\frac{\\lambda_2}{\\lambda_1 \\cdot k_{33}} \\cdot \\mathbf{k_3} \\\\
	 *   \\mathbf{K} \\cdot \\mathbf{R} \\cdot \\mathbf{a} & = \\frac{\\lambda_2}{\\lambda_1  \\cdot k_{33} \\cdot s} \\cdot \\mathbf{k_3} \\\\
	 *   \\text{$\\mathbf{K}$ invertible and $\\mathbf{K} \\cdot \\begin{pmatrix} 0 \\\\ 0 \\\\ 1 \\end{pmatrix} = \\mathbf{k_3}$} \\quad \\Rightarrow \\quad \\mathbf{R} \\cdot \\mathbf{a} & = \\frac{d}{k_{33}} \\cdot \\begin{pmatrix} 0 \\\\ 0 \\\\ 1 \\end{pmatrix} \\quad \\text{with $d := \\frac{\\lambda_2}{\\lambda_1 \\cdot s} > 0$} \\\\
	 *   \\mathbf{a} & = \\frac{d}{k_{33}} \\cdot \\mathbf{R}^\\mathsf{T} \\cdot \\begin{pmatrix} 0 \\\\ 0 \\\\ 1 \\end{pmatrix} \\\\
	 *   \\mathbf{a} & = \\frac{d}{k_{33}} \\cdot \\mathbf{r_3} \\quad \\text{with $\\mathbf{r_3}$ being the third row of $\\mathbf{R}$} \\\\
	 * \\end{align*}}
	 * {@latex.ilb \\[
	 * 	\\|\\mathbf{r_3}\\| = 1,\\ \\|\\mathbf{a}\\| = 1 \\quad \\Rightarrow \\quad \\frac{d}{k_{33}} = \\pm 1
	 * \\]}
	 * {@latex.ilb \\[
	 * 	k_{33} = \\pm 1,\\ d > 0 \\quad \\Rightarrow \\quad d = 1
	 * \\]}
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 * 	\\Rightarrow \\quad \\mathbf{a} = \\mathbf{r_3} / k_{33}
	 * \\]}
	 * And since the third row of {@latex.inline $\\mathbf{M}$} is {@latex.inline $\\mathbf{m_3} = s \\cdot k_{33} \\cdot \\mathbf{r_3}$}, an equivalent formula is
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 * 	\\quad \\mathbf{a} = \\frac{\\mathbf{m_3}}{s \\cdot k_{33}^2} \\stackrel{k_{33} = \\pm 1}{=} \\frac{\\mathbf{m_3}}{s} \\overset{\\|\\mathbf{a}\\| = 1}{\\underset{s > 0}{=}} \\frac{\\mathbf{m_3}}{\\|\\mathbf{m_3}\\|}
	 * \\]}
	 *
	 * @return The (normalized) principal axis vector in world coordinates. The direction of
	 *         the principal axis is aligned with the viewing direction of the camera.
	 */
	public SimpleVector computePrincipalAxis() {
		return this.R.getRow(2).dividedBy(this.getViewingDirection()); // alternative formulas: a = P.getRow(2, 0, 3); AND a /= norm(a);. OR a = P.getRow(2, 0, 3) / s;
	}

	/**
	 * Computes the direction of the ray corresponding to a given pixel.
	 *
	 * The formula for computing a ray corresponding to a given pixel is derived as follows:
	 * Assume that {@latex.inline %preamble{\\usepackage{amsmath}} $\\mathbf{X} = \\begin{pmatrix} x \\\\ y \\\\ z \\end{pmatrix}$}
	 * is a point on the ray belonging to pixel {@latex.inline %preamble{\\usepackage{amsmath}} $\\mathbf{p} = \\begin{pmatrix} u \\\\ v \\end{pmatrix}$}.
	 * Further assume that this point is in front of the camera and different from the
	 * camera center C. Then
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{P} \\cdot \\begin{pmatrix} \\mathbf{X} \\\\ 1 \\end{pmatrix} & = w \\cdot \\begin{pmatrix} \\mathbf{p} \\\\ 1 \\end{pmatrix} \\quad \\text{with $w > 0$} \\\\
	 *   \\mathbf{M} \\cdot \\mathbf{X} + \\mathbf{p_4} & = w \\cdot \\begin{pmatrix} u \\\\ v \\\\ 1 \\end{pmatrix} \\\\
	 *   \\mathbf{M} \\cdot \\mathbf{X} - \\mathbf{M} \\cdot \\mathbf{C} & = w \\cdot \\begin{pmatrix} u \\\\ v \\\\ 1 \\end{pmatrix} \\\\
	 *   \\mathbf{M} \\cdot (\\mathbf{X}-\\mathbf{C}) & = w \\cdot \\begin{pmatrix} u \\\\ v \\\\ 1 \\end{pmatrix} \\\\
	 *   \\mathbf{K} \\mathbf{R} \\cdot (\\mathbf{X}-\\mathbf{C}) & = w \\cdot \\begin{pmatrix} u \\\\ v \\\\ 1 \\end{pmatrix}
	 * \\end{align*}}
	 * Therefore, solve {@latex.inline %preamble{\\usepackage{amsmath}} $\\mathbf{K} \\mathbf{R} \\cdot (\\mathbf{X}-\\mathbf{C}) = \\begin{pmatrix} u \\\\ v \\\\ 1 \\end{pmatrix}$}
	 * for the ray direction {@latex.inline $\\mathbf{X} - \\mathbf{C}$}. The normalized vector is
	 * returned by this method.
	 *
	 * @param p  A 2x1 pixel vector {@latex.inline %preamble{\\usepackage{amsmath}} $\\begin{pmatrix} u \\\\ v \\end{pmatrix}$}.
	 * @return The (normalized) ray direction {@latex.inline $\\mathbf{d} = \\frac{\\mathbf{X} - \\mathbf{C}}{\\|\\mathbf{X} - \\mathbf{C}\\|}$}
	 *         in world coordinates. Its orientation is chosen so that it points from the camera
	 *         center to the viewing volume.
	 */
	public SimpleVector computeRayDirection(final SimpleVector p) {
		// input checks
		assert (p.getLen() == 2) : new IllegalArgumentException("The pixel p has to be a 2-vector but is " + p + ".");

		// create homogeneous pixel coordinates
		final SimpleVector p_hom = new SimpleVector(3);
		p_hom.setSubVecValue(0, p);
		p_hom.setElementValue(2, 1.0);

		// solve K*R*(X-C) = p_hom
		final SimpleVector rd = SimpleOperators.multiply(this.RTKinv, p_hom);

		// normalize
		rd.normalizeL2();

		return rd;
	}

	/**
	 * Compute the 4x4 OpenGL projection and modelview matrices from this Projection.
	 *
	 * This method computes a 4x4 OpenGL projection matrix which can be set in OpenGL's
	 * GL_PROJECTION mode by the user as well as a 4x4 modelview matrix which can be set in
	 * OpenGL's GL_MODELVIEW mode. Once the returned matrices are set in OpenGL, their
	 * application to world coordinate vectors returns coordinates in OpenGl's clip
	 * coordinate system (CCS) and gluProject() should yield the same result as
	 * Projection::project() (up to scaling).
	 *
	 * <p><em>Remark:</em> The parameters imgMinU, imgMinV, imgSizeU, and imgSizeV are needed in order
	 *         to define a region of pixels which constitute an image. The image coordinates
	 *         resulting from this Projection are then mapped to normalized device
	 *         coordinates ([-1, 1] x [-1, 1]). These coordinates are later mapped
	 *         back to window coordinates by the viewport transformation in OpenGL.
	 *         Therefore, remember to set the OpenGL viewport using these four parameters!
	 *
	 * <p><em>Remark:</em> Usually, the following OpenGL commands have to be used for setting up
	 * OpenGL's rendering pipeline:
	 * {@code
	 * // instantiate projection
	 * Nude::Geometry::Projection proj;
	 *
	 * // ...
	 * // set up some projection in proj by using one or more of the from...() or set...() methods
	 * // define image viewport imgMinU, imgMinV, imgSizeU, imgSizeV and depth clipping planes n and f
	 * // ...
	 *
	 * // instantiate OpenGL matrices
	 * double glProjectionGlVec[16];
	 * double glModelviewGlVec[16];
	 *
	 * // convert the projection into OpenGL representation
	 * proj.computeGLMatrices(imgMinU, imgMinV, imgSizeU, imgSizeV, n, f, glProjectionGlVec, glModelviewGlVec);
	 *
	 * // set OpenGL matrices
	 * glMatrixMode(GL_PROJECTION);
	 * glLoadMatrixd(glProjectionGlVec);
	 * glMatrixMode(GL_MODELVIEW);
	 * glLoadMatrixd(glModelviewGlVec);
	 *
	 * // map normalized device coordinates to window coordinates
	 * glViewPort(imgMinU, imgMinV, imgSizeU, imgSizeV);
	 *
	 * // additionally, the following command can be executed for reversing front/back face
	 * // definition for cameras with +z viewing direction and, therefore, enable the correct culling
	 * glFrontFace((proj.getViewingDirection() > 0) ? GL_CW : GL_CCW);
	 * }
	 *
	 * <p><em>Remark:</em> Internally, this method returns the matrix
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 *   \\left(\\begin{array}{cccc} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & -\\frac{f+n}{f-n} & -2\\frac{fn}{f-n} \\\\ 0 & 0 & -1 & 0 \\end{array}\\right) 
	 *   \\cdot
	 *   \\left(\\begin{array}{cccc} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & -1 & 0 \\\\ 0 & 0 & 0 & 1 \\end{array}\\right) 
	 *   \\cdot
	 *   \\left(\\begin{array}{c|c} {}^\\text{N}T_\\text{I} \\cdot K & 0 \\\\ \\hline 0 & 1 \\end{array}\\right) \\ ,
	 * \\]}
	 * where
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\[
	 *   {}^\\text{N}T_\\text{I} = \\begin{pmatrix} \\frac{2}{s_u} & 0 & \\frac{1-2m_u}{s_u}-1 \\\\ 0 & \\frac{2}{s_v} & \\frac{1-2m_v}{s_v}-1 \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *   =
	 *   \\begin{pmatrix} 1 & 0 & -1+\\frac{1}{s_u} \\\\ 0 & 1 & -1+\\frac{1}{s_v} \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *   \\cdot
	 *   \\begin{pmatrix} 2-\\frac{2}{s_u} & 0 & 0 \\\\ 0 & 2-\\frac{2}{s_v} & 0 \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *   \\cdot
	 *   \\begin{pmatrix} \\frac{1}{s_u-1} & 0 & 0 \\\\ 0 & \\frac{1}{s_v-1} & 0 \\\\ 0 & 0 & 1 \\end{pmatrix}
	 *   \\cdot
	 *   \\begin{pmatrix} 1 & 0 & -m_u \\\\ 0 & 1 & -m_v \\\\ 0 & 0 & 1 \\end{pmatrix}
	 * \\]}
	 * is the transformation from image pixel coordinates
	 * [imgMinU, imgMinU+imgSizeU] x [imgMinV, imgMinV+imgSizeV] to normalized device
	 * coordinates [-1, 1] x [-1, 1], the middle matrix flips the z axis (as OpenGL also
	 * would do internally, resulting in a left-handed coordinate system at this point),
	 * and the left matrix inserts the third row with depth information created from the
	 * z value. The depth row created here yields a depth coordinate which, after
	 * perspective division, maps the distance range from the camera [-n, -f] (in world
	 * coordinates) to [-1, 1], so that the clipping volume includes only objects at a
	 * distance of n to f from the camera.
	 *
	 * <p><em>Warning:</em> Mind that OpenGL uses C-style arrays in column-major order whereas the
	 *          NuDeLib uses row-major matrices. However, you shouldn't bother with that
	 *          since this methods returns <tt>glProjectionGlVec</tt> and <tt>glModelviewGlVec</tt> in
	 *          a column-major C-style array as you set it using OpenGL's
	 *          <tt>glLoadMatrixd()</tt> function.
	 *
	 * @param imgMinU  The minimum image index (in px) in the u direction (usually 0).
	 * @param imgMinV  The minimum image index (in px) in the v direction (usually 0).
	 * @param imgSizeU  The image width (in px) in u direction (imgSizeU = imgMaxU-imgMinU+1).
	 * @param imgSizeV  The image width (in px) in v direction (imgSizeV = imgMaxV-imgMinV+1).
	 * @param distanceNear  The near plane clipping coordinate (in mm). The clipping distances are measured from
	 *        the camera center to the clipping plane in mm.
	 * @param distanceFar  The far plane clipping coordinate (in mm). The clipping distances are measured from
	 *        the camera center to the clipping plane in mm.
	 * @param glProjectionGlVec  Return value. This has to be an already allocated(!) C-style array of
	 *        16 values, representing a 4x4 OpengGL projection matrix in column-major / OpenGL order.
	 *        It can be directly set in OpenGL in the GL_PROJECTION mode.
	 * @param glModelviewGlVec  Return value. This has to be an already allocated(!) C-style array of
	 *        16 values, representing a 4x4 OpengGL modelview matrix in column-major / OpenGL order.
	 *        It can be directly set in OpenGL in the GL_MODELVIEW mode.
	 *
	 * @see #computeGLMatrices(int imgMinU, int imgMinV, int imgSizeU, int imgSizeV, SimpleVector cubmin, SimpleVector cubmax, double[] glProjectionGlVec, double[] glModelviewGlVec)
	 */
	public void computeGLMatrices(
			final int imgMinU,
			final int imgMinV,
			final int imgSizeU,
			final int imgSizeV,
			final double distanceNear,
			final double distanceFar,
			final double[] glProjectionGlVec,
			final double[] glModelviewGlVec
	) {
		// input checks
		if (imgSizeU < 1 || imgSizeV < 1)
			throw new IllegalArgumentException("Error: The given image size would yield an empty image!");
		if (distanceNear <= 0 || distanceFar <= 0 || distanceFar <= distanceNear)
			throw new IllegalArgumentException("Error: The near and far clipping coordinates must be positive with n < f!");

		// output checks
		if (glProjectionGlVec.length != 16 || glModelviewGlVec.length != 16) throw new IllegalArgumentException("Output vectors have to be allocated 16-vectors!");

		// output transformation from window coordinates [imgMinU, imgMinU+imgSizeU] x [imgMinV, imgMinV+imgSizeV]
		// to normalized device coordinates [-1+1/w, 1-1/w] x [-1+1/h, 1-1/h];
		// in OpenGL, the mapping from normalized device coordinates to window coordinates
		// is performed using the specified viewport (which should correspond to the
		// imgMinU, imgSizeU, imgMinV, imgSizeV coordinates given here);
		final SimpleMatrix image2Normalized = new SimpleMatrix(new double[][]{
				{2.0/imgSizeU, 0.0, (1.0-2.0*imgMinU)/imgSizeU-1.0},
				{0.0, 2.0/imgSizeV, (1.0-2.0*imgMinV)/imgSizeV-1.0},
				{0.0, 0.0, 1.0}
		});

		// the z coordinate is negated for the clip coordinate system since OpenGL assumes
		// a camera coordinate system viewing in negative z direction but we use the
		// convention that we get positive z values for points in front of the camera
		final SimpleMatrix flipZ = new SimpleMatrix(new double[][]{
				{1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0},
				{0.0, 0.0, -1.0, 0.0},
				{0.0, 0.0, 0.0, 1.0}
		});

		// output transform for computing the depth coordinate (third row) for OpenGL
		// this matrix is generated from the one resulting from an application of
		// glFrustum() or gluPerspective)(
		final SimpleMatrix addDepthInfo = new SimpleMatrix(new double[][]{
				{1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0},
				{0.0, 0.0, -(distanceFar + distanceNear)/(distanceFar - distanceNear), -2.0*distanceFar*distanceNear/(distanceFar - distanceNear)},
				{0.0, 0.0, -1.0, 0.0}
		});

		// compute projection from world coordinates to normalized device coordinates and
		// add depth information
		final SimpleMatrix nK = SimpleOperators.multiplyMatrixProd(image2Normalized, this.K);
		final SimpleMatrix glProjectionJava = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(addDepthInfo, flipZ), General.createHomAffineMotionMatrix(nK));
		final SimpleMatrix glModelviewJava = this.getRt();

		// return the column-major matrices
		glProjectionGlVec[0]  = glProjectionJava.getElement(0, 0);
		glProjectionGlVec[1]  = glProjectionJava.getElement(1, 0);
		glProjectionGlVec[2]  = glProjectionJava.getElement(2, 0);
		glProjectionGlVec[3]  = glProjectionJava.getElement(3, 0);
		glProjectionGlVec[4]  = glProjectionJava.getElement(0, 1);
		glProjectionGlVec[5]  = glProjectionJava.getElement(1, 1);
		glProjectionGlVec[6]  = glProjectionJava.getElement(2, 1);
		glProjectionGlVec[7]  = glProjectionJava.getElement(3, 1);
		glProjectionGlVec[8]  = glProjectionJava.getElement(0, 2);
		glProjectionGlVec[9]  = glProjectionJava.getElement(1, 2);
		glProjectionGlVec[10] = glProjectionJava.getElement(2, 2);
		glProjectionGlVec[11] = glProjectionJava.getElement(3, 2);
		glProjectionGlVec[12] = glProjectionJava.getElement(0, 3);
		glProjectionGlVec[13] = glProjectionJava.getElement(1, 3);
		glProjectionGlVec[14] = glProjectionJava.getElement(2, 3);
		glProjectionGlVec[15] = glProjectionJava.getElement(3, 3);
		glModelviewGlVec[0]  = glModelviewJava.getElement(0, 0);
		glModelviewGlVec[1]  = glModelviewJava.getElement(1, 0);
		glModelviewGlVec[2]  = glModelviewJava.getElement(2, 0);
		glModelviewGlVec[3]  = glModelviewJava.getElement(3, 0);
		glModelviewGlVec[4]  = glModelviewJava.getElement(0, 1);
		glModelviewGlVec[5]  = glModelviewJava.getElement(1, 1);
		glModelviewGlVec[6]  = glModelviewJava.getElement(2, 1);
		glModelviewGlVec[7]  = glModelviewJava.getElement(3, 1);
		glModelviewGlVec[8]  = glModelviewJava.getElement(0, 2);
		glModelviewGlVec[9]  = glModelviewJava.getElement(1, 2);
		glModelviewGlVec[10] = glModelviewJava.getElement(2, 2);
		glModelviewGlVec[11] = glModelviewJava.getElement(3, 2);
		glModelviewGlVec[12] = glModelviewJava.getElement(0, 3);
		glModelviewGlVec[13] = glModelviewJava.getElement(1, 3);
		glModelviewGlVec[14] = glModelviewJava.getElement(2, 3);
		glModelviewGlVec[15] = glModelviewJava.getElement(3, 3);
	}

	/**
	 * Compute the 4x4 OpenGL projection and modelview matrices from this Projection.
	 *
	 * This method is very similar to computeGLMatrices(const int imgMinU, const int imgMinV, const unsigned int imgSizeU, const unsigned int imgSizeV, const double n, const double f, double glProjectionGlVec[16], double glModelviewGlVec[16]).
	 * The only difference is that it takes bounding box dimensions and computes appropriate
	 * near and far clipping planes internally. Additionally, the return value tells you
	 * whether or not the bounding box is visible using the specified projection. In case it
	 * is not visible, the given GL matrix storage variables are left unchanged.
	 *
	 * <p><em>Remark:</em> The parameters imgMinU, imgMinV, imgSizeU, and imgSizeV are needed in order
	 *         to define a region of pixels which constitute an image. The image coordinates
	 *         resulting from this Projection are then mapped to normalized device
	 *         coordinates ([-1, 1] x [-1, 1]). These coordinates are later mapped
	 *         back to window coordinates by the viewport transformation in OpenGL.
	 *         Therefore, remember to set the OpenGL viewport using these four parameters!
	 *
	 * <p><em>Remark:</em> Usually, the following OpenGL commands have to be used for setting up
	 * OpenGL's rendering pipeline:
	 * {@code
	 * // instantiate projection
	 * Nude::Geometry::Projection proj;
	 *
	 * // ...
	 * // set up some projection in proj by using one or more of the from...() or set...() methods
	 * // define image viewport imgMinU, imgMinV, imgSizeU, imgSizeV and bounding box extents cubmin and cubmax.
	 * // ...
	 *
	 * // instantiate OpenGL matrices
	 * double glProjectionGlVec[16];
	 * double glModelviewGlVec[16];
	 *
	 * // convert the projection into OpenGL representation
	 * proj.computeGLMatrices(imgMinU, imgMinV, imgSizeU, imgSizeV, cubmin, cubmax, glProjectionGlVec, glModelviewGlVec);
	 *
	 * // set OpenGL matrices
	 * glMatrixMode(GL_PROJECTION);
	 * glLoadMatrixd(glProjectionGlVec);
	 * glMatrixMode(GL_MODELVIEW);
	 * glLoadMatrixd(glModelviewGlVec);
	 *
	 * // map normalized device coordinates to window coordinates
	 * glViewPort(imgMinU, imgMinV, imgSizeU, imgSizeV);
	 *
	 * // additionally, the following command can be executed for reversing front/back face
	 * // definition for cameras with +z viewing direction and, therefore, enable the correct culling
	 * glFrontFace((proj.getViewingDirection() > 0) ? GL_CW : GL_CCW);
	 * }
	 *
	 * <p><em>Warning:</em> Mind that OpenGL uses C-style arrays in column-major order whereas the
	 *          NuDeLib uses row-major matrices. However, you shouldn't bother with that
	 *          since this methods returns <tt>glProjectionGlVec</tt> and <tt>glModelviewGlVec</tt> in
	 *          a column-major C-style array as you set it using OpenGL's
	 *          <tt>glLoadMatrixd()</tt> function.
	 *
	 * <p><em>Warning:</em> Using this convenience method which automatically computes suitable near and far clipping planes
	 *          implies that you do not further change the returned matrices, esp. the modelview matrix. Applying
	 *          additional transformations in OpenGL would result in a displaced cube and therefore wrong depth
	 *          clipping planes. Therefore, either apply all motions/transformations to this projection and call
	 *          this method after each change or manually specify your own depth clipping planes using
	 *          computeGLMatrices(final int imgMinU, final int imgMinV, final int imgSizeU, final int imgSizeV, final double n, final double f, final double[] glProjectionGlVec, final double[] glModelviewGlVec).
	 *
	 * @param imgMinU       The minimum image index (in px) in the u direction (usually 0).
	 * @param imgMinV       The minimum image index (in px) in the v direction (usually 0).
	 * @param imgSizeU      The image width (in px) in u direction (imgSizeU = imgMaxU-imgMinU+1).
	 * @param imgSizeV      The image width (in px) in v direction (imgSizeV = imgMaxV-imgMinV+1).
	 * @param cubmin        The bounding box' minimum extents (in mm).
	 * @param cubmax        The bounding box' maximum extents (in mm).
	 * @param glProjectionGlVec  Return value. This is an already allocated(!) C-style array of
	 *        16 values, representing a 4x4 OpengGL projection matrix in column-major /
	 *        OpenGL order. It can be directly set in OpenGL in the GL_PROJECTION mode.
	 * @param glModelviewGlVec   Return value. This is an already allocated(!) C-style array of
	 *        16 values, representing a 4x4 OpengGL modelview matrix in column-major /
	 *        OpenGL order. It can be directly set in OpenGL in the GL_MODELVIEW mode.
	 * @return Returns true if the bounding box is (at least partly) visible and false otherwise.
	 *
	 * @see #computeGLMatrices(int imgMinU, int imgMinV, int imgSizeU, int imgSizeV, double n, double f, double[] glProjectionGlVec, double[] glModelviewGlVec)
	 * @see #computeOptimalClippingDepthsForCuboid
	 */
	public boolean computeGLMatrices(
			final int imgMinU,
			final int imgMinV,
			final int imgSizeU,
			final int imgSizeV,
			final SimpleVector cubmin,
			final SimpleVector cubmax,
			final double[] glProjectionGlVec,
			final double[] glModelviewGlVec
	) {
		// output checks
		if (glProjectionGlVec.length != 16 || glModelviewGlVec.length != 16) throw new IllegalArgumentException("Output vectors have to be allocated 16-vectors!");

		// compute near and far intersection depths
		double[] nf = new double[2];
		if (!this.computeOptimalClippingDepthsForCuboid(cubmin, cubmax, nf))
			return false;
		else {
			if (nf[0] <= 0.0) {
				// call sister method with adapted clipping range (near == 0 is not possible in OpenGL)
				nf[1] *= 1.01;
				nf[0] = nf[1]/1000.0;
				System.out.println("Warning: The cuboid contains the camera center! Near clipping plane is set to 1/1000 of the far clipping plane: n=" + nf[0] + ", f=" + nf[1]);
				this.computeGLMatrices(imgMinU, imgMinV, imgSizeU, imgSizeV, nf[0], nf[1], glProjectionGlVec, glModelviewGlVec);
			} else {
				// call sister method with slightly enlarged clipping range
				nf[0] *= 0.99;
				nf[1] *= 1.01;
				this.computeGLMatrices(imgMinU, imgMinV, imgSizeU, imgSizeV, 0.99*nf[0], 1.01*nf[1], glProjectionGlVec, glModelviewGlVec);
			}
			return true;
		}
	}

	/**
	 * Computes a given point's clipping depth with respect to the camera.
	 *
	 * A signed z-distance (not Eucledian distance!) between camera center and 3D point is
	 * returned. If this distance is positive, then the point is in front of the camera and
	 * therefore visible.
	 *
	 * @param v  A 3x1 vector of the given 3D point.
	 * @return The point's clipping depth w.r.t. the camera which is computed as
	 *         {@latex.inline $k_{33} \\cdot \\bigl( \\langle r_3, v \\rangle + t_3 \\bigr) $}
	 *         where {@latex.inline $r_3$} is the 3rd row of {@latex.inline $\\mathbf{R}$}.
	 */
	private double computeClippingDepth(final SimpleVector v) {
		// input check
		assert (v.getLen() == 3) : new IllegalArgumentException("Input volume point has to be a 3-vector!");

		// compute clipping depth, which is the z component of the volume point in camera coordinates (not the eucledian distance to the camera!)
		return this.getViewingDirection() * (SimpleOperators.multiplyInnerProd(this.R.getRow(2), v) + this.t.getElement(2));
	}

	/**
	 * Computes the nearest and farthest clipping coordinates of a cuboid for OpenGL methods.
	 *
	 * The smallest and biggest clipping depth values are needed when determining optimal near
	 * and far clipping planes in OpenGL. If the cuboid contains the camera center, the near
	 * depth value is negative whereas the far depth value is positive.
	 *
	 * @param cubmin  The minimum extents of the cuboid, given as 3x1 vector [x_min, y_min, z_min].
	 * @param cubmax  The maximum extents of the cuboid, given as 3x1 vector [x_max, y_max, z_max].
	 * @param distanceNearFar  The nearest [0] and farthest [1] points' clipping depths, where a negative
	 *        value corresponds to a point behind the camera.
	 * @return <tt>true</tt> if any point of the cuboid is visible, i.e. at least the farthest
	 *         point's depth is positive. If the cuboid is fully behind the camera, then
	 *         <tt>false</tt> is returned.
	 *
	 * @see #computeClippingDepth
	 */
	private boolean computeOptimalClippingDepthsForCuboid(
			final SimpleVector cubmin,
			final SimpleVector cubmax,
			final double[] distanceNearFar
	) {
		// check input
		if (distanceNearFar.length != 2) throw new IllegalArgumentException("tntf has to be a 2-vector!");

		// compute the depths for all corners of the cuboid
		distanceNearFar[0] = Double.POSITIVE_INFINITY;
		distanceNearFar[1] = Double.NEGATIVE_INFINITY;
		double depth;
		depth = this.computeClippingDepth(new SimpleVector(cubmin.getElement(0), cubmin.getElement(1), cubmin.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmin.getElement(0), cubmin.getElement(1), cubmax.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmin.getElement(0), cubmax.getElement(1), cubmin.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmin.getElement(0), cubmax.getElement(1), cubmax.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmax.getElement(0), cubmin.getElement(1), cubmin.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmax.getElement(0), cubmin.getElement(1), cubmax.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmax.getElement(0), cubmax.getElement(1), cubmin.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		depth = this.computeClippingDepth(new SimpleVector(cubmax.getElement(0), cubmax.getElement(1), cubmax.getElement(2)));
		distanceNearFar[0] = Math.min(distanceNearFar[0], depth);
		distanceNearFar[1] = Math.max(distanceNearFar[1], depth);
		return (distanceNearFar[1] > 0.0);
	}

	/**
	 * Computes a given point's Eucledian distance to the camera.
	 *
	 * A signed distance between camera center and 3D point is returned. If this distance
	 * is positive, then the point is in front of the camera and therefore visible.
	 *
	 * @param v  A 3x1 vector of the given 3D point.
	 * @return The point's depth w.r.t. the camera which is computed by transforming it
	 *         into camera coordinates, computing the norm
	 *         {@latex.inline $\\pm \\left\\| \\mathbf{R} \\cdot \\mathbf{v} + \\mathbf{t} \\right\\|_2 $},
	 *         and chosing the sign according to the transformed point's z coordinate
	 *         (taking into account the camera's viewing direction).
	 * 
	 * @see #project
	 */
	public double computeDepth(final SimpleVector v) {
		// input check
		assert (v.getLen() == 3) : new IllegalArgumentException("Input volume point has to be a 3-vector!");

		// compute v in camera coordinates (R*v + t)
		final SimpleVector v_cam = SimpleOperators.add(SimpleOperators.multiply(this.R, v), this.t);

		// the 2-norm of the point in camera coordinates is its distance, but we'll return a signed distance, based on whether the point is in front of or behind the camera
		return v_cam.normL2() * Math.signum(v_cam.getElement(2)) * this.getViewingDirection();
	}

	/**
	 * Projects a given voxel to a pixel and determines its visibility.
	 *
	 * @param  volumePoint  The 3x1 volume point in world coordinates.
	 * @param  pixel  The returned 2x1 image coordiante in pixels.
	 * @return The method also returns the depth of the projected point. This depth is
	 *         positive if the volume point is visible and false otherwise.
	 *
	 * @see #computeDepth
	 */
	public double project(final SimpleVector volumePoint, final SimpleVector pixel) {
		// input check
		assert (volumePoint.getLen() == 3) : new IllegalArgumentException("The given volume point must be a 3-vector!");
		assert (pixel.getLen() == 2) : new IllegalArgumentException("The given pixel container must have length 2!");

		// compute volumePoint in camera coordinates (R*v+t) and save intermediate result for later depth computation
		final SimpleVector volumePoint_cam = SimpleOperators.add(SimpleOperators.multiply(this.R, volumePoint), this.t);

		// compute projected point by 
		final SimpleVector pixel_hom = SimpleOperators.multiply(K, volumePoint_cam); // K*(R*v+t)
		pixel.init(General.normalizeFromHomogeneous(pixel_hom));

		// return depth
		return volumePoint_cam.normL2() * Math.signum(volumePoint_cam.getElement(2)) * this.getViewingDirection();
	}

	/**
	 * Computes the two intersections of a ray with a cuboid, called entry and
	 * exit point where the ray is defined by this projection and the given pixel.
	 * 
	 * Internally, this method first computes the camera center and ray direction and
	 * then calls intersectRayWithCuboid(final SimpleVector origin, final SimpleVector dir, final SimpleVector cubmin, final SimpleVector cubmax, final double[] tntf).
	 * The entry and exit points are returned as distances from the camera center along
	 * the projection ray. The world coordinates of the entry and exit voxels may be
	 * computed as follows:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align*}
	 *   \\mathbf{v}_n & = \\mathbf{C} + t_n \\cdot \\mathbf{d} \\\\
	 * 	 \\mathbf{v}_f & = \\mathbf{C} + t_f \\cdot \\mathbf{d}
	 * \\end{align*}}
	 *
	 * @param p  A 2x1 pixel vector.
	 * @param cubmin  The cuboid's minimal planes given as [min_x, min_y, min_z] in world
	 *        coordinates.
	 * @param cubmax  The cuboid's box' maximal planes given as [max_x, max_y, max_z] in
	 *        world coordinates.
	 * @param distanceNearFar  Return values. In case of a hit: Positive distance (in world coordinate
	 *        units) of nearest [0] and farthest [1] plane intersections.
	 * @param C  Return value. The ray origin / camera center (corresponding to the internal
	 *        projection matrix) in world coordinates [ro_x, ro_y, ro_z].
	 * @param d  Return value. The normalized(!) ray direction (corresponding to the given
	 *        pixel) in world coordinates [rd_x, rd_y, rd_z].
	 * @return  Boolean value which is true if the ray intersects with the bounding box and
	 *          false otherwise.
	 */
	public boolean intersectRayWithCuboid(
			final SimpleVector p,
			final SimpleVector cubmin,
			final SimpleVector cubmax,
			final double[] distanceNearFar,
			final SimpleVector C,
			final SimpleVector d
	) {
		// input checks
		assert (distanceNearFar.length == 2) : new IllegalArgumentException("tntf has to be a 2-vector!");

		// output checks
		assert (C.getLen() == 3) : new IllegalArgumentException("C has to be a 3-vector!");
		assert (d.getLen() == 3) : new IllegalArgumentException("d has to be a 3-vector!");

		// compute ray origin and direction
		C.init(this.computeCameraCenter());
		d.init(this.computeRayDirection(p));

		// compute the intersections
		return General.intersectRayWithCuboid(C, d, cubmin, cubmax, distanceNearFar);
	}

	/**
	 * Convenience method to compute the world coordinate of a detector coordinate. The world source position is computed with computeCameraCenter.
	 * @param sourcePosition the source position in world coordinates
	 * @param detectorCoordinate the detector cooridnate
	 * @param sourceDetectorDistance the distance from source to detector
	 * @param pixelDimensionX the detector pixel width
	 * @param pixelDimensionY the detector pixel heigth
	 * @return the world point of the detector pixel coordinate.
	 */
	public PointND computeDetectorPoint(SimpleVector sourcePosition, SimpleVector detectorCoordinate, double sourceDetectorDistance, double pixelDimensionX, double pixelDimensionY, int maxU, int maxV) {
		SimpleVector corner = sourcePosition.clone();
		SimpleVector offsets = computeOffset(new SimpleVector(pixelDimensionX, pixelDimensionY));
		double length = Math.sqrt (Math.pow((detectorCoordinate.getElement(0)+offsets.getElement(0)-(maxU))*pixelDimensionX, 2) +
				Math.pow((detectorCoordinate.getElement(1)+offsets.getElement(1)-(maxV))* pixelDimensionY, 2) +
				Math.pow(sourceDetectorDistance, 2));
		corner.add(computeRayDirection(detectorCoordinate).multipliedBy(length));
		return new PointND(corner);
	}

	/**
	 * Constructs the K matrix from distance and offset parameters.
	 * 
	 * This method sets this Projection's K matrix according to the given distance,
	 * pixel size, offset, and other parameters. Note that the distance and pixel size
	 * parameters include redundant information not needed for establishing a world-pixel
	 * projection relationship. They cannot be recovered from this Projection again. The
	 * only (non-redundant) information that can be recovered is the source-to-detector
	 * distances in pixels. Also note that the camera coordinate system is supposed to be set up
	 * using the R and t transform parameters in a way that the x and y axes are colinear with the
	 * projection's u and v axes, resp. Furthermore, the z axis must point into the direction of the
	 * principal axis (for dir == +1.0) or into the oppose direction (for dir == -1.0).
	 * 
	 * <p><em>Warning:</em> The parameters given to this method are mostly discarded since they
	 * are not essential or minimal for representing a single projection. This is why they also cannot
	 * be recovered from this object. This method is rather a convenience method for setting up this
	 * projection.
	 * 
	 * @param sourceToDetector  The distance from projection / camera / X-ray source
	 *        center to the image plane / detector in world coordinates.
	 * @param spacingUV  The pixel size in u- and v-direction.
	 * @param sizeUV  The projection image size in u- and v-direction.
	 * @param offset  The offset (in pixels) from the image center to the principal point (where the principal
	 *        ray hits the image plane orthogonally), i.e., the principal point is computed as (image center + offset).
	 * @param dir  +1.0 for a camera looking in {@latex.inline %preamble{\\usepackage{amsmath}} $+z_\\text{C}$} direction
	 *        direction or -1.0 for a camera looking in {@latex.inline %preamble{\\usepackage{amsmath}} $-z_\\text{C}$} direction.
	 * @param skew  The skew parameter (usually zero).
	 * 
	 * @see #setRtFromCircularTrajectory
	 */
	public void setKFromDistancesSpacingsSizeOffset(
			final double sourceToDetector,
			final SimpleVector spacingUV,
			final SimpleVector sizeUV,
			final SimpleVector offset,
			final double dir,
			final double skew
	) {
		// input checks
		if (sourceToDetector <= 0.0) throw new IllegalArgumentException("Source-to-detector has to be positive!");
		if (spacingUV.getLen() != 2) throw new IllegalArgumentException("Pixel spacing has to be a 2-vector!");
		if (spacingUV.getElement(0) <= 0.0) throw new IllegalArgumentException("Pixel spacing in u-direction has to be positive!");
		if (spacingUV.getElement(1) <= 0.0) throw new IllegalArgumentException("Pixel spacing in v-direction has to be positive!");
		if (sizeUV.getLen() != 2) throw new IllegalArgumentException("Image size has to be a 2-vector!");
		if (sizeUV.getElement(0) <= 0) throw new IllegalArgumentException("Image size in u-direction has to be positive!");
		if (sizeUV.getElement(1) <= 0) throw new IllegalArgumentException("Image size in v-direction has to be positive!");
		if (offset.getLen() != 2) throw new IllegalArgumentException("Offset must be a 2-vector!");
		if (Math.abs(Math.abs(dir) - 1.0) > Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Error: Viewing direction has to be +/-1 but is " + dir + "!");

		// construct K matrix
		this.K.setElementValue(0, 0, sourceToDetector / spacingUV.getElement(0)); // focal length (px) = distance (mm) / spacing (mm/px)
		this.K.setElementValue(1, 1, sourceToDetector / spacingUV.getElement(1)); // focal length (px) = distance (mm) / spacing (mm/px)
		this.K.setElementValue(0, 1, skew);
		this.setPrincipalPointValue(SimpleOperators.add(sizeUV.multipliedBy(0.5), offset)); // principal point = image center + offset
		this.setViewingDirectionValue(dir);

		// update precomputed RTKinv matrix
		this.updateRTKinv();
	}

	public static enum CameraAxisDirection {
		/** Indicates that an image axis is oriented along the detector's direction of motion */
		DETECTORMOTION_PLUS,
		/** Indicates that an image axis is oriented opposited to the detector's direction of motion */
		DETECTORMOTION_MINUS,
		/** Indicates that an image axis is oriented along the rotation direction of the camera */
		ROTATIONAXIS_PLUS,
		/** Indicates that an image axis is oriented opposite to the rotation direction of the camera */
		ROTATIONAXIS_MINUS,
		DETECTORMOTION_ROTATED,
		ROTATIONAXIS_ROTATED
	}

	/**
	 * Constructs the extrinsic parameters (R and t) of this projection from the extrensic parameters
	 * source-to-isocenter distance, rotation axis, rotation angle, and viewing axis.
	 * 
	 * It is assumed that this projection is part of a series of projections acquired along a circular
	 * trajectory with a known rotation center, axis, radius (i.e. the source-to-isocenter distance),
	 * and rotation angle. These parameters together with the direction to the zero angle position
	 * fully describe the camera's current position. The principal axis of the camera is assumed to 
	 * always cross the rotation center so that the image plane is not tilted, swiveled, or rotated
	 * with respect to the trajectory plane. For more general cases where the camera's rotational
	 * position is not defined that simple, the user has to modify the resulting [R|t] transform
	 * himself by multiplying it from the left with additional rotations in the camera coordinate
	 * system.
	 * 
	 * The principal axis (in world coordinates) of the projection
	 * with a zero angular argument. If the principal axis hits the center of rotation (i.e.,
	 * thr detector is neither tilted nor swiveled), the principal axis is just the negated
	 * center-to-camera vector.
	 * 
	 * <p><em>Warning:</em> The parameters given to this method are mostly discarded since they
	 * are not essential or minimal for representing a single projection. This is why they also cannot
	 * be recovered from this object. This method is rather a convenience method for setting up this
	 * projection.
	 * 
	 * @param rotationCenter  The center of rotation, given as 3D point in world coordinates. (This is
	 *        usually chosen to be (0,0,0).)
	 * @param rotationAxis  The rotation axis (in world coordinates) for a circular acquisition
	 *        trajectory (where the last projection's angular value is bigger than the first one's.)
	 *        (This is usually chosen to be (0,0,1).)
	 * @param sourceToAxisDistance  The radius of the camera rotation, i.e. the distance from the camera /
	 *        X-ray source to the iso center.
	 * @param centerToCameraAtZeroAngle  The direction from the center of rotation to the camera position
	 *        at zero angle. This direction vector can have any length but zero. Its normalized version
	 *        is used together with the sourceToAxisDistance for translating the camera to its zero angle
	 *        position. Since the zero angle position has to be on the circular trajectory defined by
	 *        the rotationAxis (and the sourceToAxisDistance), centerToCameraAtZeroAngle has to be
	 *        perpendicular to rotationAxis. (This is usually chosen to be (1,0,0).)
	 * @param uDirection  Sets the camera's u axis in terms of the direction of motion of the detector
	 *        (for increasing angles) and the rotation axis.
	 * @param vDirection  Sets the camera's v axis in terms of the direction of motion of the detector
	 *        (for increasing angles) and the rotation axis.
	 * @param rotationAngle  The angle of rotation (away from the zero position) for this projection.
	 * @return  After setting up the extrinsic parameters R and t using this method, the x and y axes
	 *          of the camera coordinate system are aligned with the u and v axis of the projections,
	 *          resp. The camera coordinate system's z axis may then either point into the camera's
	 *          viewing direction (return value 1.0) or into the opposite direction (return value
	 *          -1.0). This method already sets the viewing direction in this Projection accordingly,
	 *          but it has to be taken into account when re-setting up the internal parameters using
	 *          another method.
	 * 
	 * @see #setKFromDistancesSpacingsSizeOffset
	 * 
	 * TODO: Move this method to ProjectionSeries, taking an array of angles and generating the series
	 * of Projections?
	 */
	public double setRtFromCircularTrajectory(
			final SimpleVector rotationCenter,
			final SimpleVector rotationAxis,
			final double sourceToAxisDistance,
			final SimpleVector centerToCameraAtZeroAngle,
			final CameraAxisDirection uDirection,
			final CameraAxisDirection vDirection,
			final double rotationAngle
	) {
		// input checks
		if (rotationCenter.getLen() != 3) throw new IllegalArgumentException("Rotation center has to be a 3-vector!");
		if (rotationAxis.getLen() != 3) throw new IllegalArgumentException("Rotation axis has to be a 3-vector!");
		if (rotationAxis.normL2() < CONRAD.DOUBLE_EPSILON) throw new IllegalArgumentException("Rotation axis has to be a non-zero directional vector!");
		if (sourceToAxisDistance <= 0.0) throw new IllegalArgumentException("Camera motion's radius has to be positive!");
		if (centerToCameraAtZeroAngle.getLen() != 3) throw new IllegalArgumentException("Center to camera has to be a 3-vector!");
		if (centerToCameraAtZeroAngle.normL2() < CONRAD.DOUBLE_EPSILON) throw new IllegalArgumentException("The center-to-camera vector has to be a non-zero directional vector!");
		if (SimpleOperators.multiplyInnerProd(rotationAxis, centerToCameraAtZeroAngle) > Math.sqrt(CONRAD.DOUBLE_EPSILON)) throw new IllegalArgumentException("Rotation axis and center-to-camera vector have to be perpendicular!");
		if (
				(
						(uDirection == CameraAxisDirection.DETECTORMOTION_PLUS || uDirection == CameraAxisDirection.DETECTORMOTION_MINUS)
						&&
						(vDirection == CameraAxisDirection.DETECTORMOTION_PLUS || vDirection == CameraAxisDirection.DETECTORMOTION_MINUS)
				)
				||
				(
						(uDirection == CameraAxisDirection.ROTATIONAXIS_PLUS || uDirection == CameraAxisDirection.ROTATIONAXIS_MINUS)
						&&
						(vDirection == CameraAxisDirection.ROTATIONAXIS_PLUS || vDirection == CameraAxisDirection.ROTATIONAXIS_MINUS)
				)
		) throw new IllegalArgumentException("u and v direction are colinear but must be perpendicular!");

		// normalize axes
		rotationAxis.normalizeL2();
		centerToCameraAtZeroAngle.normalizeL2();

		// Remark on the notation used in the following lines for transformation matrices:
		// B_T_A is a transformation matrix, converting coordinates given in system A to coordinates in system B:
		// p_B = B_T_A * p_A
		// Therefore, the concatenation C_T_B * B_T_A converts coordinates from system A to system C.

		// transformation (translation only) from world coordinates (WORLD) to center of rotation system (CENT) with unmodified axes orientations
		final SimpleMatrix CENT_T_WORLD = General.createHomAffineMotionMatrix(rotationCenter.negated());

		// transformation (rotation only) of axes from (CENT) system to plane of rotation system (PLANE) with the rotationAxis as first and the -centerToCameraAtZeroAngle vector as third axis (ra, ra x cc, -cc)
		final SimpleMatrix PLANE_T_CENT = SimpleMatrix.I_4.clone();
		PLANE_T_CENT.setSubRowValue(0, 0, rotationAxis);
		PLANE_T_CENT.setSubRowValue(1, 0, General.crossProduct(rotationAxis, centerToCameraAtZeroAngle));
		PLANE_T_CENT.setSubRowValue(2, 0, centerToCameraAtZeroAngle.negated());

		// transformation (rotation only) from (PLANE) system to angulation-aligned (AA)
		final SimpleMatrix AA_T_PLANE = General.createHomAffineMotionMatrix(Rotations.createBasicXRotationMatrix(-rotationAngle));  

		// transformation (rotation only) from (AA) to imageaxes-aligned system (IA)
		final SimpleMatrix rot = new SimpleMatrix(3, 3);
		
		double beta = 0;
		if(Configuration.getGlobalConfiguration() != null){
			beta = (double)Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth()/
							(double)Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight();
			beta = Math.atan(beta);
		} 
		
		switch (uDirection) {
		case DETECTORMOTION_PLUS:
			rot.setColValue(0, General.E_Y.negated());
			break;
		case DETECTORMOTION_MINUS:
			rot.setColValue(0, General.E_Y);
			break;
		case ROTATIONAXIS_PLUS:
			rot.setColValue(0, General.E_X);
			break;
		case ROTATIONAXIS_MINUS:
			rot.setColValue(0, General.E_X.negated());
			break;
		case DETECTORMOTION_ROTATED:
			if(Configuration.getGlobalConfiguration()==null){
				rot.setColValue(0, General.E_Y.negated());
				break;
			}
			rot.setColValue(0,new SimpleVector(-Math.cos(beta), -Math.sin(beta),0));
			break;
		default:
			throw new RuntimeException("Unexpected axis definition for u direction!");
		}
		switch (vDirection) {
		case DETECTORMOTION_PLUS:
			rot.setColValue(1, General.E_Y.negated());
			break;
		case DETECTORMOTION_MINUS:
			rot.setColValue(1, General.E_Y);
			break;
		case ROTATIONAXIS_PLUS:
			rot.setColValue(1, General.E_X);
			break;
		case ROTATIONAXIS_MINUS:
			rot.setColValue(1, General.E_X.negated());
			break;
		case ROTATIONAXIS_ROTATED:
			if(Configuration.getGlobalConfiguration()==null){
				rot.setColValue(1, General.E_X);
				break;
			}
			rot.setColValue(1, new SimpleVector(Math.sin(beta), -Math.cos(beta), 0));
			break;
		default:
			throw new RuntimeException("Unexpected axis definition for v direction!");
		}
		rot.setColValue(2, General.crossProduct(rot.getCol(0), rot.getCol(1)));
		rot.transpose();
		final SimpleMatrix IA_T_AA = General.createHomAffineMotionMatrix(rot);

		// compute viewing direction by transforming CA's z axis (which is the zero camera's viewing direction) into the C0 system 
		final double viewingDirection = SimpleOperators.multiplyInnerProd(rot.getRow(2), General.E_Z);
		if (Math.abs(viewingDirection) != 1.0) throw new RuntimeException("Unexpected internal error!");

		// transformation (translation only) from (IA) to camera system (CAM)
		final SimpleVector transl = new SimpleVector(0.0, 0.0, viewingDirection*sourceToAxisDistance);
		final SimpleMatrix CAM_T_IA = General.createHomAffineMotionMatrix(transl);

		// assemble full transformation from world coordinates (WORLD) to camera system (CAM)
		SimpleMatrix CAM_T_WORLD = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(CAM_T_IA, IA_T_AA), AA_T_PLANE), SimpleOperators.multiplyMatrixProd(PLANE_T_CENT, CENT_T_WORLD));

		// set extrinsic parameters, correcting for possible -z == -(u x v) viewing direction
		this.setRtValue(CAM_T_WORLD);
		this.setViewingDirectionValue(viewingDirection);

		// return viewing direction (+/- 1) to let the user know in which direction he's looking
		return viewingDirection;
	}

	/**
	 * This convenience method computes the source-to-detector distance in world corrdinate dimensions.
	 * 
	 * The {@link Projection} class only stores the minimal information needed for establishing
	 * world to pixel correspondences. It therefore only know the focal length in pixels
	 * (stored as the first two diagonal entries of the K matrix). Computing the source-to-detector
	 * distance requires knowledge about the pixel size. Since there are focal length entries in K
	 * that could be used for computing the source-to-detector distance, this method returns two
	 * result values. They should, theoretically, be equal. This check is left to the caller of this method. 
	 * 
	 * @param spacingUV  Given pixel sizes in u- and v-direction.
	 * @return  An array of two computed source-to-detector distances that should be equal.
	 */
	public double[] computeSourceToDetectorDistance(final SimpleVector spacingUV) {
		// input check
		if (spacingUV.getLen() != 2) throw new IllegalArgumentException("Spacing has to be a 2-vector!");
		if (spacingUV.getElement(0) <= 0.0) throw new IllegalArgumentException("Spacing in u-direction has to be positive!");
		if (spacingUV.getElement(1) <= 0.0) throw new IllegalArgumentException("Spacing in v-direction has to be positive!");

		// distance (mm) = focal length (px) * spacing (mm/px)
		return new double[] {this.K.getElement(0, 0) * spacingUV.getElement(0), this.K.getElement(1, 1) * spacingUV.getElement(1)};
	}

	/**
	 * This convenience method computes the offset from the image center to the principal point.
	 * 
	 * @param sizeUV  Projection image size in u- and v-direction in pixels.
	 * @return  A 2-vector with the computed offset from the image center to the principal point
	 *          in pixels (so that principal point = center + offset).
	 */
	public SimpleVector computeOffset(final SimpleVector sizeUV) {
		// input check
		if (sizeUV.getLen() != 2) throw new IllegalArgumentException("Image size has to be a 2-vector!");
		if (sizeUV.getElement(0) <= 0) throw new IllegalArgumentException("Image size in u-direction has to be positive!");
		if (sizeUV.getElement(1) <= 0) throw new IllegalArgumentException("Image size in v-direction has to be positive!");

		return SimpleOperators.subtract(this.getPrincipalPoint(), sizeUV.multipliedBy(0.5)); // offset = principal point - image center
	}

	/**
	 * update precomputed matrices
	 */
	private void updateRTKinv() {
		if (this.K != null && this.R != null)
			this.RTKinv = SimpleOperators.multiplyMatrixProd(this.R.transposed(), this.K.inverse(SimpleMatrix.InversionType.INVERT_UPPER_TRIANGULAR));
	}

	/**
	 * Returns a String representation of this projection's 3x4 matrix.
	 * @return The 3x4 matrix in the numeric package's standard String format.
	 */
	@Override
	public String toString() {
		return this.computeP().toString();
	}

	/**
	 * This method is only used for XML serialization.
	 * @param projectionMatrix  String representation of the projection matrix to de-serialize.
	 */
	public void setPMatrixSerialization(String projectionMatrix) {
		this.initFromP(new SimpleMatrix(projectionMatrix));
	}

	/**
	 * This method is only used for XML serialization.
	 * @return A String representation of the projection matrix to serialize.
	 */
	public String getPMatrixSerialization() {
		return this.computeP().toString();
	}

	/**
	 * @return the rTKinv
	 */
	public SimpleMatrix getRTKinv() {
		return RTKinv;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
