package edu.stanford.rsl.conrad.filtering.multiprojection;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import Jama.Matrix;
import Jama.SingularValueDecomposition;
import ij.ImagePlus;
import ij.plugin.filter.GaussianBlur;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;

/**
 * This tool can be used to track a diaphragm in the projection data.
 * We assume that the diaphragm is a parabola that is opened to the bottom.
 * The initial position of the diaphragm is has to be defined during the configuration of the tool.
 * <BR><BR>
 * Internally the parameter estimation is refined with the RANSAC algorithm.
 * The method is described in the corresponding paper by Marco Boegel.
 * <BR><BR>
 * The image data is not altered. The filter is merely used to extract information.
 * 
 * @author Marco Boegel
 *
 */

public class DiaphragmTrackingTool extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 334423738188161138L;
	/**
	 * segments the detected diaphragm. Sets everything above the detected parabola to zero
	 */
	private static boolean segment = true;
	private static boolean centerDia = true;
	private static int ptX = 360;
	private static int ptY = 390;
	//this is the pt from 2 frames before, it is needed to specify the ROI of the previous frame
	private static int oldptX = 0;
	private static int oldptY = 0;
	private static int roiWidthLeft = 125;
	private static int roiWidthRight = 125;
	private static int roiWidthHalfMax = 125;
	private static int roiHeightBottomMax = 50;
	private static int roiHeightBottom = 50;
	private static int roiHeightTop = 5;
	private static int roiHeightTopMax = 5;
	private static int maxIterations = 20000;

	private ArrayList<int[]> a = new ArrayList<int[]>();
	private static int maxY, maxX;
	private static float lowHyst = 0.2f;
	private static float highHyst = 0.7f;
	private static double[] lastMdl = { 0, 0, 0 };

	private static double projTurningAngle = 0;
	//the x-coordinate of the frame before
	private static int lastX = 0;
	private static int lastY = 0;
	//number of frames for initialization of direction and avgXDisplacement
	private static final int initFrames = 5;
	//the average direction the model moved in the first initFrames frames
	private static int direction = 0;
	private static double[] primaryAngles = null;
	private static double avgXDisplacement = 0;
	private static int projRange = 0;

	private ImageProcessor ip;
	private ImageProcessor imp;
	private ImageProcessor impSeg;
	//index of current projection
	private static int projNr = 0;
	private static int threads = 4;
	private static int workload = maxIterations / threads;
	private static ExecutorService es = Executors.newFixedThreadPool(threads);

	//countdownlatch for increased performance, Future.get() blocks the main thread too often
	private static CountDownLatch latch = new CountDownLatch(threads);

	class ModelCallable implements Callable<double[]> {

		int workload = 0;

		ModelCallable(int workload) {
			this.workload = workload;
		}

		/**
		 * This method fits a parabola through an array of selected sample-points
		 * using least square estimation.
		 * 
		 * TODO: This should be refactored to a "Function".
		 * 
		 * @param pts ArrayList of sample-points
		 * @return parametric form of parabola ( a*x*x + b*x + c = y )
		 */
		double[] fitPoly(ArrayList<int[]> pts) {

			int size = pts.size();
			double[] model = new double[3];
			double[][] matrixVals = new double[size][3];
			double[] sol = new double[size];
			for (int i = 0; i < size; i++) {
				int x = pts.get(i)[0];
				int y = pts.get(i)[1];

				matrixVals[i][0] = x * x;
				matrixVals[i][1] = x;
				matrixVals[i][2] = 1;
				sol[i] = y;
			}

			Matrix M = new Matrix(matrixVals);
			Matrix y = new Matrix(sol, size);

			//SVD start
			SingularValueDecomposition svd = M.svd();

			Matrix S = svd.getS();

			for (int i = 0; i < Math.min(S.getColumnDimension(), S.getRowDimension()); i++) {
				double entry = 0;
				if ((entry = S.get(i, i)) != 0)
					S.set(i, i, 1 / entry);
			}
			Matrix U = svd.getU().transpose();
			Matrix V = svd.getV();
			Matrix mod = V.times(S).times(U).times(y);
			//SVD end

			for (int i = 0; i < 3; i++) {
				model[i] = mod.get(i, 0);
			}

			return model;
		}

		/**
		 * This method evaluates the fitness of a computed model.
		 * 
		 * TODO: Unite with optimization framework!
		 * 
		 * @param mdl model to compute fitness on
		 * @return fitness value
		 */
		double evaluateModel(double[] mdl) {

			double eval = 0;

			if (mdl[0] <= 0)
				return 0;

			if (centerDia == true) {
				if (projNr == 0) {
					lastMdl[0] = mdl[0];
				}
				double diffProcentual = Math.abs(lastMdl[0] - mdl[0]) / lastMdl[0];

				if (diffProcentual > 0.05)
					return 0;
				for (int x = 0; x < (roiWidthRight + roiWidthLeft); x++) {

					int y = (int) (mdl[0] * x * x + mdl[1] * x + mdl[2]);

					if ((y >= 0) && (y < (roiHeightBottom + roiHeightTop))
							&& ((ip.getPixelValue(x, y) != 0) || (ip.getPixelValue(x, y - 1) != 0)
									|| (ip.getPixelValue(x, y + 1) != 0) || (ip.getPixelValue(x - 1, y - 1) != 0)
									|| (ip.getPixelValue(x + 1, y - 1) != 0) || (ip.getPixelValue(x - 1, y + 1) != 0)
									|| (ip.getPixelValue(x + 1, y + 1) != 0)))
						eval++;

				}

				return eval;
			}

			int curX = (int) Math.round((-mdl[1] / (2 * mdl[0])) + ptX - roiWidthLeft);
			int curY = (int) Math.round(((4 * mdl[0] * mdl[2] - mdl[1] * mdl[1]) / (4 * mdl[0])) + ptY - roiHeightTop);

			if (projNr == 0) {
				lastX = curX;
				lastY = curY;
				lastMdl[0] = mdl[0];
				direction = 0;
				avgXDisplacement = 0;
			}
			if (Math.abs(lastY - curY) > 8)
				return 0;
			double diffProcentual = Math.abs(lastMdl[0] - mdl[0]) / lastMdl[0];

			if (diffProcentual > 0.05)
				return 0;

			int dir = curX - lastX;
			if (projNr > initFrames && Math.abs(dir) > 3 * avgXDisplacement)
				return 0;

			double projAngle = primaryAngles[projNr];

			if (projNr > initFrames && ((dir > 0 && direction < 0) || (dir < 0 && direction > 0))) {
				if (projTurningAngle - projRange > projAngle || projAngle > projTurningAngle + projRange)
					return 0;
			}

			for (int x = 0; x < (roiWidthRight + roiWidthLeft); x++) {

				int y = (int) (mdl[0] * x * x + mdl[1] * x + mdl[2]);

				if ((y >= 0) && (y < (roiHeightBottom + roiHeightTop))
						&& ((ip.getPixelValue(x, y) != 0) || (ip.getPixelValue(x, y - 1) != 0)
								|| (ip.getPixelValue(x, y + 1) != 0) || (ip.getPixelValue(x - 1, y - 1) != 0)
								|| (ip.getPixelValue(x + 1, y - 1) != 0) || (ip.getPixelValue(x - 1, y + 1) != 0)
								|| (ip.getPixelValue(x + 1, y + 1) != 0)))
					eval++;

				//3x3 region 
				//				if( (y >= 0) && (y < (roiHeightBottom+roiHeightTop))) {
				//					
				//					eval+=Math.max(ip.getPixelValue(x, y), Math.max(ip.getPixelValue(x, y-1),Math.max(ip.getPixelValue(x, y+1), Math.max(ip.getPixelValue(x-1, y-1), Math.max(ip.getPixelValue(x+1, y-1), Math.max(ip.getPixelValue(x-1, y+1),ip.getPixelValue(x+1, y+1)))))));
				//				}
				//only exact hit
				//				if( (y >= 0) && (y < (roiHeightBottom+roiHeightTop))) {
				//					eval+=ip.getPixelValue(x,y);
				//				}
			}
			if (dir == 0)
				eval *= 0.75;

			return eval;
		}

		@Override
		public double[] call() throws Exception {
			double[] chosenModel = null;
			double[] model = null;
			double eval = 0;
			double chosenEval = 0;

			for (int i = 0; i < workload; i++) {
				ArrayList<int[]> samples = new ArrayList<int[]>();

				for (int k = 0; k < 3; k++) {
					int ran = (int) (Math.random() * a.size());
					samples.add(a.get(ran));
				}

				model = fitPoly(samples);
				eval = evaluateModel(model);

				if (eval > chosenEval) {
					chosenModel = model;
					chosenEval = eval;
				}
			}

			//The evaluation result is needed to get the best model out of all threads	
			double[] result = new double[] { chosenModel[0], chosenModel[1], chosenModel[2], chosenEval };
			latch.countDown();
			return result;
		}

	}

	/**
	 * This method moves the seed for the next projection
	 * to the apex of the current parabola model.
	 * @param mdl the current parametric model of the parabola
	 */
	private static void moveSeedPt(double[] mdl) {
		oldptX = ptX;
		oldptY = ptY;

		ptX = (int) Math.round((-mdl[1] / (2 * mdl[0])) + ptX - roiWidthLeft);
		ptY = (int) Math.round(((4 * mdl[0] * mdl[2] - mdl[1] * mdl[1]) / (4 * mdl[0])) + ptY - roiHeightTop);

	}

	/**
	 * This method resets the ROI to the pre-configured Values. 
	 * This is needed because the ROI shrinks/grows
	 * as it approaches/leaves the boundary.
	 */
	private void resetRoi() {
		roiWidthLeft = roiWidthHalfMax;
		roiWidthRight = roiWidthHalfMax;
		roiHeightBottom = roiHeightBottomMax;
		roiHeightTop = roiHeightTopMax;
	}

	/**
	 * This method makes sure that the current seed point does not leave the image boundaries .
	 */
	private void checkSeedPt() {
		if (ptX < 0)
			ptX = 0;
		else if (ptX >= maxX)
			ptX = maxX - 1;
		if (ptY < 0)
			ptY = 0;
		else if (ptY >= maxY)
			ptY = maxY - 1;

	}

	/**
	 * This method sets the ROI-boundaries, so that they don't leave the image-boundaries.
	 * @return true if the ROI was changed, false otherwise
	 */
	private static boolean setRoi() {

		boolean change = false;
		if ((ptX - roiWidthLeft) < 0) {
			change = true;
			roiWidthLeft = ptX;
		}
		if (ptY - roiHeightTop < 0) {
			change = true;
			roiHeightTop = ptY;
		}
		if ((ptY + roiHeightBottom) >= maxY) {
			change = true;
			roiHeightBottom = maxY - ptY - 1;
		}
		if ((ptX + roiWidthRight) >= maxX) {
			change = true;
			roiWidthRight = maxX - ptX - 1;
		}

		return change;
	}

	/**
	 * This method computes a random sample consenus.
	 * It selects 1 random point out of every pre-computed region and fits a parabola.
	 * @return model that fits the data best
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public double[] computeRansac() throws InterruptedException, ExecutionException {

		double[] chosenModel = null;
		double oldEval = 0;

		//submitting threads
		ArrayList<Future<double[]>> results = new ArrayList<Future<double[]>>();
		for (int i = 0; i < threads; i++) {
			results.add(es.submit(new ModelCallable(workload)));
		}

		latch.await();

		//evaluate the results of all threads
		for (int i = 0; i < threads; i++) {
			double[] res = results.get(i).get();

			if (res[3] >= oldEval) {
				oldEval = res[3];
				chosenModel = new double[] { res[0], res[1], res[2] };
			}
		}
		if (centerDia == false) {
			int curX = (int) Math.round((-chosenModel[1] / (2 * chosenModel[0])) + ptX - roiWidthLeft);
			int curY = (int) Math.round(
					((4 * chosenModel[0] * chosenModel[2] - chosenModel[1] * chosenModel[1]) / (4 * chosenModel[0]))
							+ ptY - roiHeightTop);

			//computes the direction the model moves in the first initFrames frames, this direction is enforced until the turningpoint, and then inverted
			if (projNr > 0 && (projNr < initFrames || direction == 0)) {
				direction += (curX - lastX);
			}
			if (projNr > 0) {
				avgXDisplacement = (avgXDisplacement * projNr + Math.abs(curX - lastX)) / (projNr + 1);
			}
			//inverse the direction at the turningangle
			if (primaryAngles[projNr] == projTurningAngle) {
				direction *= -1.0;
			}
			lastY = curY;
			lastX = curX;
		}
		lastMdl = chosenModel;
		a.clear();

		return chosenModel;
	}

	/**
	 * This method plots the model into the original image.
	 * @param mdl model to be plotted
	 * @param c 1 for display original image, other for display in cropped image
	 */
	public void dispModell(double[] mdl, int c) {

		//display in the original image
		if (c == 1) {
			//translate ROI-coordinates to image-coordinates
			for (int x = 0; x < (roiWidthLeft + roiWidthRight); x++) {

				int y = (int) (mdl[0] * x * x + mdl[1] * x + mdl[2]);
				if (x + ptX - roiWidthLeft >= 0 && y + ptY - roiHeightTop >= 0 && x + ptX - roiWidthLeft < maxX
						&& y + ptY - roiHeightTop < maxY)
					imp.putPixelValue(x + ptX - roiWidthLeft, y + ptY - roiHeightTop, 0);

			}
		} else {

			//display in the cropped image
			for (int x = 0; x < (roiWidthRight + roiWidthLeft); x++) {

				int y = (int) (mdl[0] * x * x + mdl[1] * x + mdl[2]);
				if (y < 0 || y >= (roiHeightBottom + roiHeightTop))
					continue;
				ip.putPixelValue(x, y, 100);

			}
		}
	}

	@Override
	public ImageFilteringTool clone() {
		DiaphragmTrackingTool clone = new DiaphragmTrackingTool();
		clone.setConfigured(configured);
		return clone;
	}

	@Override
	public String getToolName() {
		return "RansacParallel";
	}

	@Override
	public boolean isDeviceDependent() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void configure() {

		Configuration config = Configuration.getGlobalConfiguration();

		int x = (int) Math.round(Integer.parseInt(JOptionPane.showInputDialog("Enter seedX: ", ptX)));
		int y = (int) Math.round(Integer.parseInt(JOptionPane.showInputDialog("Enter seedY: ", ptY)));
		int iter = (int) Math.round(
				Integer.parseInt(JOptionPane.showInputDialog("Enter maxIterations for RanSaC: ", maxIterations)));
		int rw = (int) Math
				.round(Integer.parseInt(JOptionPane.showInputDialog("Enter half ROI-width: ", roiWidthHalfMax)));
		int rh = (int) Math
				.round(Integer.parseInt(JOptionPane.showInputDialog("Enter ROI-height: ", roiHeightBottomMax)));
		float low = (float) Float
				.parseFloat(JOptionPane.showInputDialog("Enter Low Hysteresis Threshold for EdgeMap: ", lowHyst));
		float high = (float) Float
				.parseFloat(JOptionPane.showInputDialog("Enter High Hysteresis Threshold for EdgeMap: ", highHyst));
		double p = (double) Double.parseDouble(JOptionPane
				.showInputDialog("Enter angle of frontal view (90 for XCAT, 0 for medical:", projTurningAngle));

		config.setProjTurningAngle(p);
		config.setLowHyst(low);
		config.setHighHyst(high);
		config.setRoiWidthHalf(rw);
		config.setRoiHeightBottom(rh);
		config.setRoiHeightTop(rh / 5);
		config.setMaxIter(iter);
		config.setSeedX(x);
		config.setSeedY(y);

		setConfigured(true);

	}

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

	public static void setPtX(int x) {
		DiaphragmTrackingTool.ptX = x;
	}

	public static void setPtY(int y) {
		DiaphragmTrackingTool.ptY = y;
	}

	/**
	 * This method performs a Canny Edge Detection on the specified ImageProcessor
	 * @param ip ImageProcessor to be processed
	 * @return EdgeMap
	 */
	private ImageProcessor cannyEdge(ImageProcessor ip) {

		double pi = Math.PI;
		int width = ip.getWidth();
		int height = ip.getHeight();

		//generate the Sobel convolution masks
		float kernelx[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		float kernely[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

		ImageProcessor ipx = ip.duplicate();
		ImageProcessor ipy = ip.duplicate();

		ipx.convolve(kernelx, 3, 3);
		ipy.convolve(kernely, 3, 3);

		double[][] theta = new double[width][height];
		byte[][] thetaDirections = new byte[width][height];
		//compute gradient angle atan2(gradY,gradX)
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {

				theta[x][y] = Math.atan2(ipy.getPixelValue(x, y), ipx.getPixelValue(x, y));
			}
		}
		//thetaDirections contains the clamped angles of the gradients (nord/south...    1: N/S , 2: NE/SW , 3: W/E , 4: NW/SE ) 
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				double t = theta[i][j];
				if (t < 0)
					t += pi;

				if (t <= pi / 8)
					thetaDirections[i][j] = 3;
				else if (t <= 3 * pi / 8)
					thetaDirections[i][j] = 2;
				else if (t <= 5 * pi / 8)
					thetaDirections[i][j] = 1;
				else if (t <= 7 * pi / 8)
					thetaDirections[i][j] = 4;
				else
					thetaDirections[i][j] = 3;

			}
		}

		//magnitude start
		ipx.sqr();
		ipy.sqr();

		ImageUtil.addProcessors(ipx, ipy);
		ipx.sqrt();
		//magnitude end
		ipx = nonMaximumSuppression(ipx, thetaDirections);
		//computing percentiles for hysteresis
		ImageStatistics stats = ipx.getStatistics();
		int[] hist = stats.histogram;

		int i = hist.length - 1;
		double area = stats.area - hist[0];
		int count = (int) area;

		while (count > (area * highHyst))
			count -= hist[i--];
		float hi = (float) (i * stats.binSize);

		while (count > (area * lowHyst))
			count -= hist[i--];
		float lo = (float) (i * stats.binSize);

		ipx = hysteresis(ipx, lo, hi);

		return ipx;

	}

	/**
	 * This Method implements the non maximum suppression of the Canny Edge Detector
	 * @param grad contains the Gradient Magnitudes
	 * @param theta contains the clamped angles of the gradients (nord/south...    1: N/S , 2: NE/SW , 3: W/E , 4: NW/SE ) 
	 * @return ImageProcessor with only the maximum Edges
	 */
	private ImageProcessor nonMaximumSuppression(ImageProcessor grad, byte[][] theta) {
		int height = grad.getHeight();
		int width = grad.getWidth();

		ImageProcessor out = grad.duplicate();
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				int ay, ax, by, bx;
				int angle = theta[x][y];
				if (angle == 1) {
					ay = y + 1;
					ax = x;
					by = y - 1;
					bx = x;
				} else if (angle == 2) {
					ay = y + 1;
					ax = x + 1;
					by = y - 1;
					bx = x - 1;
				} else if (angle == 3) {
					ay = y;
					ax = x + 1;
					by = y;
					bx = x - 1;

					//            else if(angle == 0) {
					//	                out.putPixelValue(x, y, 0);
					//	            	continue;
				} else if (angle == 4) {
					ay = y - 1;
					ax = x + 1;
					by = y + 1;
					bx = x - 1;
				} else {
					System.err.println(
							"Non-Maximum-Suppression: Theta needs to be in [1,4], see javadoc for detailed description.");
					return out;
				}
				//rangecheck
				if (ay < 0 || ay >= height || ax < 0 || ax >= width)
					continue;
				// (xy) not maximal -> set zero
				else if (grad.getPixelValue(ax, ay) > grad.getPixelValue(x, y)) {
					out.putPixelValue(x, y, 0);
					continue;
				}

				//rangecheck
				if (by < 0 || by >= height || bx < 0 || bx >= width)
					continue;
				// (xy) not maximal -> set zero
				else if (grad.getPixelValue(bx, by) > grad.getPixelValue(x, y)) {
					out.putPixelValue(x, y, 0);
					continue;
				}
			}
		return out;
	}

	private ImageProcessor hysteresis(ImageProcessor input, float lowHyst2, float highHyst2) {

		int width = input.getWidth();
		int height = input.getHeight();
		ImageProcessor out = new FloatProcessor(width, height);

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				if (out.getPixelValue(x, y) == 0 && input.getPixelValue(x, y) >= highHyst2) {
					follow(input, x, y, out, lowHyst2);
				}
			}
		}
		return out;
	}

	private void follow(ImageProcessor input, int x, int y, ImageProcessor out, float lowHyst2) {

		int width = input.getWidth();
		int height = input.getHeight();

		int x2 = x == width - 1 ? x : x + 1;
		int y2 = y == height - 1 ? y : y + 1;
		int y0 = y == 0 ? y : y - 1;
		int x0 = x == 0 ? x : x - 1;

		out.putPixelValue(x, y, 10);

		for (int i = x0; i <= x2; i++) {
			for (int j = y0; j <= y2; j++) {
				if ((j != y || i != x) && out.getPixelValue(i, j) == 0 && input.getPixelValue(i, j) >= lowHyst2) {
					follow(input, i, j, out, lowHyst2);

					return;
				}
			}
		}
		return;
	}

	public static int getRoiWidthLeft() {
		return roiWidthLeft;
	}

	public static void setRoiWidthLeft(int roiWidthLeft) {
		DiaphragmTrackingTool.roiWidthLeft = roiWidthLeft;
	}

	public static int getRoiWidthRight() {
		return roiWidthRight;
	}

	public static void setRoiWidthRight(int roiWidthRight) {
		DiaphragmTrackingTool.roiWidthRight = roiWidthRight;
	}

	public static int getRoiHeightBottom() {
		return roiHeightBottom;
	}

	public static void setRoiHeightBottom(int roiHeightBottom) {
		DiaphragmTrackingTool.roiHeightBottom = roiHeightBottom;
	}

	public static int getRoiHeightTop() {
		return roiHeightTop;
	}

	public static void setRoiHeightTop(int roiHeightTop) {
		DiaphragmTrackingTool.roiHeightTop = roiHeightTop;
	}

	public static int getRoiWidthHalfMax() {
		return DiaphragmTrackingTool.roiWidthHalfMax;
	}

	public static int getRoiHeightBottomMax() {
		return DiaphragmTrackingTool.roiHeightBottomMax;
	}

	public static int getRoiHeightTopMax() {
		return DiaphragmTrackingTool.roiHeightTopMax;
	}

	public static void setRoiWidthHalfMax(int roiWidthHalfMax) {
		DiaphragmTrackingTool.roiWidthHalfMax = roiWidthHalfMax;
	}

	public static void setRoiHeightBottomMax(int roiHeightBottomMax) {
		DiaphragmTrackingTool.roiHeightBottomMax = roiHeightBottomMax;
	}

	public static void setRoiHeightTopMax(int roiHeightTopMax) {
		DiaphragmTrackingTool.roiHeightTopMax = roiHeightTopMax;
	}

	public static float getLowHyst() {
		return lowHyst;
	}

	public static void setLowHyst(float lowHyst) {
		DiaphragmTrackingTool.lowHyst = lowHyst;
	}

	public static float getHighHyst() {
		return highHyst;
	}

	public static void setHighHyst(float highHyst) {
		DiaphragmTrackingTool.highHyst = highHyst;
	}

	public static int getMaxIterations() {
		return maxIterations;
	}

	public static void setMaxIterations(int maxIterations) {
		DiaphragmTrackingTool.maxIterations = maxIterations;
	}

	public static int getWorkload() {
		return workload;
	}

	public static void setWorkload(int workload) {
		DiaphragmTrackingTool.workload = workload;
	}

	private void initialize() {

		Configuration config = Configuration.getGlobalConfiguration();

		roiHeightBottom = config.getRoiHeightBottom();
		roiHeightBottomMax = roiHeightBottom;
		roiWidthLeft = config.getRoiWidthHalf();
		roiWidthRight = roiWidthLeft;
		roiWidthHalfMax = roiWidthLeft;
		roiHeightTop = config.getRoiHeightTop();
		roiHeightTopMax = roiHeightTop;
		lowHyst = config.getLowHyst();
		highHyst = config.getHighHyst();
		ptX = config.getSeedX();
		ptY = config.getSeedY();
		projTurningAngle = config.getProjTurningAngle();

		maxIterations = config.getMaxIter();
		setWorkload(maxIterations / threads);

		//create MotionField, is set after each image is processed
		int primlength = config.getGeometry().getProjectionStackSize();

		config.setRespiratoryMotionField(new double[primlength]);
		config.setDiaphragmCoords(new double[primlength][2]);
		config.setDiaphragmModelField(new double[primlength][5]);

		primaryAngles = config.getGeometry().getPrimaryAngles();
		projRange = (int) (40.0 / config.getGeometry().getAverageAngularIncrement());

		int proj = 0;
		while (primaryAngles[proj] < projTurningAngle)
			proj++;

		projTurningAngle = primaryAngles[proj];

	}

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {

		if (projectionNumber == 0)
			initialize();

		projNr = projectionNumber;
		//convert to float to prevent overflow in edgedetection
		Grid2D grid = inputQueue.get(projectionNumber);
		ImageProcessor imageProcessor = new FloatProcessor(grid.getWidth(), grid.getHeight(), grid.getBuffer());

		if (new ImagePlus(null, imageProcessor).getType() != ImagePlus.GRAY32)
			imageProcessor = imageProcessor.convertToFloat();

		maxX = imageProcessor.getWidth();
		maxY = imageProcessor.getHeight();
		//orig image
		imp = imageProcessor;

		checkSeedPt();
		boolean roiChanged = setRoi();

		imageProcessor.setRoi(ptX - roiWidthLeft, ptY - roiHeightTop, roiWidthLeft + roiWidthRight,
				roiHeightBottom + roiHeightTop);
		ip = imageProcessor.crop();
		ip.resetMinAndMax();
		//sigma 1.0
		new GaussianBlur().blurGaussian(ip, 1.0, 1.0, 0.02);
		ip = cannyEdge(ip);
		ip.resetMinAndMax();

		int roiHeight = roiHeightBottom + roiHeightTop;
		int roiWidth = roiWidthLeft + roiWidthRight;
		int transX = -ptX + oldptX;
		int transY = -ptY + oldptY;
		;
		if (projNr == 0) {
			for (int x = 0; x < roiWidth; x++) {
				for (int y = 0; y < roiHeight; y++) {
					if (ip.getPixelValue(x, y) != 0) {
						int[] toAdd = { x, y };
						a.add(toAdd);
					}
				}
			}

		} else {

			ImageProcessor copy = new FloatProcessor(ip.getWidth(), ip.getHeight());

			for (int x = -transX; x < roiWidth - transX; x++) {

				int y = (int) (lastMdl[0] * x * x + lastMdl[1] * x + lastMdl[2]);
				//translation of the bounding box ( lastimg backtranslation + currentimg forwardtranslation)
				int xptXroi = x + transX;
				int yptYroi = y + transY;

				if (xptXroi >= 0 && xptXroi < roiWidth) {

					for (int k = -20; k <= 20; k++) {
						if (yptYroi + k >= 0 && yptYroi + k < roiHeight
								&& ip.getPixelValue(xptXroi, yptYroi + k) != 0) {
							int[] toAdd = { xptXroi, yptYroi + k };
							a.add(toAdd);
							copy.putPixelValue(xptXroi, yptYroi + k, 1.0);

						}
					}
				}

			}
			ip = copy;
		}

		double[] mdl = computeRansac();

		if (segment == false) {
			dispModell(mdl, 1);
		} else {
			impSeg = new FloatProcessor(grid.getWidth(), grid.getHeight());
			dispSegmented(mdl);

		}

		moveSeedPt(mdl);

		Configuration.getGlobalConfiguration().setDiaphragmModelEntry(projNr,
				new double[] { mdl[0], mdl[1], mdl[2], roiHeightTop, roiWidthLeft });

		if (roiChanged == true)
			resetRoi();

		Configuration.getGlobalConfiguration().setDiaphragmCoordsEntry(projNr, new double[] { ptX, ptY });

		System.out.println(projectionNumber);
		if (segment == false) {
			sink.process(new Grid2D((float[]) imp.getPixels(), imp.getWidth(), imp.getHeight()), projectionNumber);
		} else {
			sink.process(new Grid2D((float[]) impSeg.getPixels(), impSeg.getWidth(), impSeg.getHeight()),
					projectionNumber);
		}
	}

	public void dispSegmented(double[] mdl) {

		//display in the original image
		//translate ROI-coordinates to image-coordinates
		for (int x = -ptX + roiWidthLeft; x < maxX - ptX + roiWidthLeft; x++) {

			int y = (int) (mdl[0] * x * x + mdl[1] * x + mdl[2]);
			if (x + ptX - roiWidthLeft >= 0 && y + ptY - roiHeightTop >= 0 && x + ptX - roiWidthLeft < maxX
					&& y + ptY - roiHeightTop < maxY) {
				for (int i = y + ptY - roiHeightTop; i < maxY; i++)
					impSeg.putPixelValue(x + ptX - roiWidthLeft, i, imp.getPixelValue(x + ptX - roiWidthLeft, i));
			}

		}
	}

}
/*
 * Copyright (C) 2010-2014 - Marco Boegel 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/