package edu.stanford.rsl.conrad.cuda;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.IJ;
import ij.ImagePlus;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUaddress_mode;
import jcuda.driver.CUarray;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUdevprop;
import jcuda.driver.CUfilter_mode;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.dim3;

/**
This is the existing CUDABackProjector with added motion compensation. The motion compensated kernel is called and the respiration signal is added, other than that the class is the same.

@author Marco Bögel

 */

public class CUDACompensatedBackProjector extends VOIBasedReconstructionFilter implements Runnable {

	double mot[] = new double[200];
	/**
	 * 
	 */
	private static final long serialVersionUID = 7732291211252379464L;

	/**
	 * The CUDA module containing the kernel
	 */
	private CUmodule module = null;
	//private static Object lock = new Object();
	private static boolean debug = true;

	// Pre-determined kernel block size
	static int bpBlockSize[] = { 32, 16 };

	/**
	 * The handle for the CUDA function of the kernel that is to be called
	 */
	private CUfunction function = null;

	/**
	 * The volume data that is to be reconstructed
	 */
	protected float h_volume[];

	/**
	 * The 2D projection texture reference
	 */
	private CUtexref projectionTex = null;

	/**
	 * The grid size of the kernel execution
	 */
	private dim3 gridSize = null;

	/**
	 * the context
	 */
	private CUcontext cuCtx = null;

	/**
	 * The global variable of the module which stores the
	 * view matrix.
	 */
	private CUdeviceptr projectionMatrix = null;
	private CUdeviceptr volStride = null;
	private CUdeviceptr volumePointer = null;
	private CUarray projectionArray = null;

	protected ImageGridBuffer projections;
	protected ArrayList<Integer> projectionsAvailable;
	protected ArrayList<Integer> projectionsDone;
	private boolean largeVolumeMode = false;
	private int nSteps = 1;
	private int subVolumeZ = 0;

	private boolean initialized = false;

	public CUDACompensatedBackProjector() {
		super();
	}

	@Override
	public void prepareForSerialization() {
		super.prepareForSerialization();
		projectionMatrix = null;
		volStride = null;
		volumePointer = null;
		projectionArray = null;
		projections = null;
		projectionsAvailable = null;
		projectionsDone = null;
		cuCtx = null;
		gridSize = null;
		projectionTex = null;
		h_volume = null;
		initialized = false;
		function = null;
		module = null;
	}

	@Override
	public void configure() throws Exception {
		boolean success = true;
		configured = success;
	}

	public void reset() {
		projectionArray = null;
		JCuda.cudaThreadExit();
	}

	protected void init() {
		if (!initialized) {
			largeVolumeMode = false;

			int reconDimensionX = getGeometry().getReconDimensionX();
			int reconDimensionY = getGeometry().getReconDimensionY();
			int reconDimensionZ = getGeometry().getReconDimensionZ();
			projections = new ImageGridBuffer();
			projectionsAvailable = new ArrayList<Integer>();
			projectionsDone = new ArrayList<Integer>();
			// Initialize the JCudaDriver. Note that this has to be done from 
			// the same thread that will later use the JCudaDriver API.
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			CUdevice dev = CUDAUtil.getBestDevice();
			cuCtx = new CUcontext();
			JCudaDriver.cuCtxCreate(cuCtx, 0, dev);
			// check space on device:
			int[] memory = new int[1];
			int[] total = new int[1];
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			JCudaDriver.cuMemGetInfo(memory, total);
			int availableMemory = (int) (CUDAUtil.correctMemoryValue(memory[0]) / ((long) 1024 * 1024));
			int requiredMemory = (int) (((((double) reconDimensionX) * reconDimensionY * ((double) reconDimensionZ)
					* Sizeof.FLOAT)
					+ (((double) Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight())
							* Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth() * Sizeof.FLOAT))
					/ (1024.0 * 1024));
			if (debug) {
				System.out.println("Total available Memory on CUDA card:" + availableMemory);
				System.out.println("Required Memory on CUDA card:" + requiredMemory);
			}
			if (requiredMemory > availableMemory) {
				nSteps = CUDAUtil.iDivUp(requiredMemory, (int) (availableMemory));
				if (debug)
					System.out.println("Switching to large volume mode with nSteps = " + nSteps);
				largeVolumeMode = true;
			}
			if (debug) {
				CUdevprop prop = new CUdevprop();
				JCudaDriver.cuDeviceGetProperties(prop, dev);
				System.out.println(prop.toFormattedString());
			}

			// Load the CUBIN file containing the kernel
			module = new CUmodule();
			JCudaDriver.cuModuleLoad(module, "respirationCompensatedBackprojectWithCuda.ptx");

			// Obtain a function pointer to the kernel function. This function
			// will later be called. 
			// 
			function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module, "_Z17backprojectKernelPfiiiffffff");
			// create the reconstruction volume;
			int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * Sizeof.FLOAT;
			if (largeVolumeMode) {
				subVolumeZ = CUDAUtil.iDivUp(reconDimensionZ, nSteps);
				if (debug)
					System.out.println("SubVolumeZ: " + subVolumeZ);
				h_volume = new float[reconDimensionX * reconDimensionY * subVolumeZ];
				memorysize = reconDimensionX * reconDimensionY * subVolumeZ * Sizeof.FLOAT;
				if (debug)
					System.out.println("Memory: " + memorysize);
			} else {
				h_volume = new float[reconDimensionX * reconDimensionY * reconDimensionZ];
			}
			// copy volume to device
			volumePointer = new CUdeviceptr();
			JCudaDriver.cuMemAlloc(volumePointer, memorysize);
			JCudaDriver.cuMemcpyHtoD(volumePointer, Pointer.to(h_volume), memorysize);

			// compute adapted volume size 
			//    volume size in x = multiple of bpBlockSize[0]
			//    volume size in y = multiple of bpBlockSize[1]

			int adaptedVolSize[] = new int[3];
			if ((reconDimensionX % bpBlockSize[0]) == 0) {
				adaptedVolSize[0] = reconDimensionX;
			} else {
				adaptedVolSize[0] = ((reconDimensionX / bpBlockSize[0]) + 1) * bpBlockSize[0];
			}
			if ((reconDimensionY % bpBlockSize[1]) == 0) {
				adaptedVolSize[1] = reconDimensionY;
			} else {
				adaptedVolSize[1] = ((reconDimensionY / bpBlockSize[1]) + 1) * bpBlockSize[1];
			}
			adaptedVolSize[2] = reconDimensionZ;
			int volStrideHost[] = new int[2];
			// compute volstride and copy it to constant memory
			volStrideHost[0] = adaptedVolSize[0];
			volStrideHost[1] = adaptedVolSize[0] * adaptedVolSize[1];

			volStride = new CUdeviceptr();
			JCudaDriver.cuModuleGetGlobal(volStride, new int[1], module, "gVolStride");
			JCudaDriver.cuMemcpyHtoD(volStride, Pointer.to(volStrideHost), Sizeof.INT * 2);

			// Calculate new grid size
			gridSize = new dim3(CUDAUtil.iDivUp(adaptedVolSize[0], bpBlockSize[0]),
					CUDAUtil.iDivUp(adaptedVolSize[1], bpBlockSize[1]), adaptedVolSize[2]);

			// Obtain the global pointer to the view matrix from
			// the module
			projectionMatrix = new CUdeviceptr();
			JCudaDriver.cuModuleGetGlobal(projectionMatrix, new int[1], module, "gProjMatrix");

			initialized = true;
		}

	}

	private synchronized void unload() {
		if (initialized) {

			if (projectionArray != null) {
				JCudaDriver.cuArrayDestroy(projectionArray);
			}

			int reconDimensionX = getGeometry().getReconDimensionX();
			int reconDimensionY = getGeometry().getReconDimensionY();
			int reconDimensionZ = getGeometry().getReconDimensionZ();

			if ((projectionVolume != null) && (!largeVolumeMode)) {
				// fetch data
				int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * 4;
				JCudaDriver.cuMemcpyDtoH(Pointer.to(h_volume), volumePointer, memorysize);
				int width = projectionVolume.getSize()[0];
				int height = projectionVolume.getSize()[1];
				if (this.useVOImap) {
					for (int k = 0; k < projectionVolume.getSize()[2]; k++) {
						for (int j = 0; j < height; j++) {
							for (int i = 0; i < width; i++) {
								float value = h_volume[(((height * k) + j) * width) + i];
								if (voiMap[i][j][k]) {
									projectionVolume.setAtIndex(i, j, k, value);
								} else {
									projectionVolume.setAtIndex(i, j, k, 0);
								}
							}
						}
					}
				} else {
					for (int k = 0; k < projectionVolume.getSize()[2]; k++) {
						for (int j = 0; j < height; j++) {
							for (int i = 0; i < width; i++) {
								float value = h_volume[(((height * k) + j) * width) + i];
								projectionVolume.setAtIndex(i, j, k, value);
							}
						}
					}
				}
			} else {
				System.out.println("Check ProjectionVolume. It seems null.");
			}

			h_volume = null;
			// free memory on device

			JCudaDriver.cuMemFree(volumePointer);
			// destory context
			JCudaDriver.cuCtxDestroy(cuCtx);

			reset();

			initialized = false;
		}
	}

	private synchronized void initProjectionMatrix(int projectionNumber) {
		// load projection Matrix for current Projection.
		SimpleMatrix pMat = getGeometry().getProjectionMatrix(projectionNumber).computeP();
		float[] pMatFloat = new float[pMat.getCols() * pMat.getRows()];
		for (int j = 0; j < pMat.getRows(); j++) {
			for (int i = 0; i < pMat.getCols(); i++) {

				pMatFloat[(j * pMat.getCols()) + i] = (float) pMat.getElement(j, i);
			}
		}
		JCudaDriver.cuMemcpyHtoD(projectionMatrix, Pointer.to(pMatFloat), Sizeof.FLOAT * pMatFloat.length);
	}

	private synchronized void initProjectionData(Grid2D projection) {
		initialize(projection);
		if (projection != null) {
			float[] proj = new float[projection.getWidth() * projection.getHeight()];

			for (int i = 0; i < projection.getWidth(); i++) {
				for (int j = 0; j < projection.getHeight(); j++) {
					proj[(j * projection.getWidth()) + i] = projection.getPixelValue(i, j);
				}
			}

			if (projectionArray == null) {
				// Create the 2D array that will contain the
				// projection data. 
				projectionArray = new CUarray();
				CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
				ad.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
				ad.Width = projection.getWidth();
				ad.Height = projection.getHeight();
				ad.NumChannels = 1;//projection.getNChannels();
				JCudaDriver.cuArrayCreate(projectionArray, ad);
			}

			// Copy the projection data to the array  
			CUDA_MEMCPY2D copy2 = new CUDA_MEMCPY2D();
			copy2.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
			copy2.srcHost = Pointer.to(proj);
			copy2.srcPitch = projection.getWidth() * Sizeof.FLOAT;
			copy2.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
			copy2.dstArray = projectionArray;
			copy2.WidthInBytes = projection.getWidth() * Sizeof.FLOAT;
			copy2.Height = projection.getHeight();
			JCudaDriver.cuMemcpy2D(copy2);

			// Obtain the texture reference from the module, 
			// set its parameters and assign the projection  
			// array as its reference.
			projectionTex = new CUtexref();
			JCudaDriver.cuModuleGetTexRef(projectionTex, module, "gTex2D");
			JCudaDriver.cuTexRefSetFilterMode(projectionTex, CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
			JCudaDriver.cuTexRefSetAddressMode(projectionTex, 0, CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
			JCudaDriver.cuTexRefSetFlags(projectionTex, JCudaDriver.CU_TRSF_READ_AS_INTEGER);
			JCudaDriver.cuTexRefSetFormat(projectionTex, CUarray_format.CU_AD_FORMAT_FLOAT, 4);
			JCudaDriver.cuTexRefSetArray(projectionTex, projectionArray, JCudaDriver.CU_TRSA_OVERRIDE_FORMAT);

			// Set the texture references as parameters for the function call
			JCudaDriver.cuParamSetTexRef(function, JCudaDriver.CU_PARAM_TR_DEFAULT, projectionTex);
		} else {
			System.out.println("Projection was null!!");
		}
	}

	@Override
	public String getName() {
		return "CUDA Backprojector";
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

	public void waitForResult() {
		cudaRun();
	}

	@Override
	public void backproject(Grid2D projection, int projectionNumber) {
		appendProjection(projection, projectionNumber);
	}

	private void appendProjection(Grid2D projection, int projectionNumber) {
		projections.add(projection, projectionNumber);
		projectionsAvailable.add(new Integer(projectionNumber));
	}

	private synchronized void projectSingleProjection(int projectionNumber, int dimz) {
		// load projection matrix
		initProjectionMatrix(projectionNumber);
		// load projection
		Grid2D projection = projections.get(projectionNumber);
		initProjectionData(projection);
		if (!largeVolumeMode) {
			projections.remove(projectionNumber);
		}
		// backproject for each slice
		// CUDA Grids are only two dimensional!
		int[] zed = new int[1];
		int reconDimensionZ = dimz;
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();

		zed[0] = reconDimensionZ;
		double motionfieldShifted[] = new double[200];

		double val = Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(0);
		for (int i = 0; i < 200; i++) {
			if (i < 20) {
				motionfieldShifted[i] = 1.3 * 0.05 * i * val;
			} else
				motionfieldShifted[i] = 1.3
						* Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(i - 20);
		}

		mot[projectionNumber] = -1.0
				* Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(projectionNumber);
		//		Pointer diaMotion = Pointer.to(new int[]{(int) Math.round(-1.0*Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(projectionNumber)) });
		//		Pointer diaMotion = Pointer.to(new int[]{(int) Math.round(-1.0*motionfieldShifted[projectionNumber]) });
		System.out.println(Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(projectionNumber));
		double motionfield[] = new double[200];
		motionfield[0] = 0.0;
		motionfield[1] = -0.18582188544104383;
		motionfield[2] = -0.3716437708821445;
		motionfield[3] = -0.5574656563231883;
		motionfield[4] = -0.7432875417642322;
		motionfield[5] = -0.9291094272053044;
		motionfield[6] = -1.1149313126463767;
		motionfield[7] = -1.300753198087449;
		motionfield[8] = -1.4865750835284928;
		motionfield[9] = -1.672396968969565;
		motionfield[10] = -1.8582188544106089;
		motionfield[11] = -2.044040739851681;
		motionfield[12] = -2.2298626252927534;
		motionfield[13] = -2.415684510733797;
		motionfield[14] = -2.6015063961748695;
		motionfield[15] = -2.7873282816159417;
		motionfield[16] = -2.9731501670569855;
		motionfield[17] = -3.158972052498058;
		motionfield[18] = -3.34479393793913;
		motionfield[19] = -3.530615823380174;
		motionfield[20] = -3.7164377088212177;
		motionfield[21] = -3.9022595942623184;
		motionfield[22] = -4.088081479703362;
		motionfield[23] = -4.459705151586803;
		motionfield[24] = -4.88441504816808;
		motionfield[25] = -5.309124944749357;
		motionfield[26] = -5.7338348413306335;
		motionfield[27] = -6.15854473791191;
		motionfield[28] = -6.583254634493187;
		motionfield[29] = -7.007964531074435;
		motionfield[30] = -7.432674427655741;
		motionfield[31] = -7.857384324237017;
		motionfield[32] = -8.282094220818294;
		motionfield[33] = -8.706804117399543;
		motionfield[34] = -9.131514013980848;
		motionfield[35] = -9.556223910562096;
		motionfield[36] = -9.980933807143401;
		motionfield[37] = -10.40564370372465;
		motionfield[38] = -10.830353600305955;
		motionfield[39] = -11.255063496887203;
		motionfield[40] = -11.679773393468508;
		motionfield[41] = -12.104483290049757;
		motionfield[42] = -12.529193186631034;
		motionfield[43] = -12.953903083212339;
		motionfield[44] = -13.378612979793587;
		motionfield[45] = -13.767469952177692;
		motionfield[46] = -14.127644585203996;
		motionfield[47] = -14.487819218230328;
		motionfield[48] = -14.84799385125666;
		motionfield[49] = -15.208168484282993;
		motionfield[50] = -15.568343117309325;
		motionfield[51] = -15.928517750335658;
		motionfield[52] = -16.288692383361962;
		motionfield[53] = -16.648867016388294;
		motionfield[54] = -17.009041649414655;
		motionfield[55] = -17.36921628244096;
		motionfield[56] = -17.72939091546729;
		motionfield[57] = -18.089565548493624;
		motionfield[58] = -18.449740181519957;
		motionfield[59] = -18.80991481454629;
		motionfield[60] = -19.170089447572593;
		motionfield[61] = -19.530264080598926;
		motionfield[62] = -19.890438713625258;
		motionfield[63] = -20.25061334665159;
		motionfield[64] = -20.610787979677923;
		motionfield[65] = -20.970962612704227;
		motionfield[66] = -21.33113724573056;
		motionfield[67] = -21.58981084329045;
		motionfield[68] = -21.645482369917403;
		motionfield[69] = -21.701153896544355;
		motionfield[70] = -21.756825423171307;
		motionfield[71] = -21.81249694979826;
		motionfield[72] = -21.86816847642521;
		motionfield[73] = -21.92384000305219;
		motionfield[74] = -21.979511529679144;
		motionfield[75] = -22.035183056306096;
		motionfield[76] = -22.090854582933048;
		motionfield[77] = -22.14652610956003;
		motionfield[78] = -22.20219763618698;
		motionfield[79] = -22.25786916281396;
		motionfield[80] = -22.313540689440913;
		motionfield[81] = -22.369212216067865;
		motionfield[82] = -22.424883742694817;
		motionfield[83] = -22.48055526932177;
		motionfield[84] = -22.53622679594872;
		motionfield[85] = -22.591898322575673;
		motionfield[86] = -22.647569849202654;
		motionfield[87] = -22.703241375829606;
		motionfield[88] = -22.758912902456558;
		motionfield[89] = -22.791821477052054;
		motionfield[90] = -22.642626435395727;
		motionfield[91] = -22.493431393739428;
		motionfield[92] = -22.34423635208313;
		motionfield[93] = -22.19504131042683;
		motionfield[94] = -22.045846268770504;
		motionfield[95] = -21.896651227114205;
		motionfield[96] = -21.747456185457906;
		motionfield[97] = -21.598261143801608;
		motionfield[98] = -21.449066102145252;
		motionfield[99] = -21.299871060488954;
		motionfield[100] = -21.150676018832655;
		motionfield[101] = -21.001480977176357;
		motionfield[102] = -20.852285935520058;
		motionfield[103] = -20.70309089386376;
		motionfield[104] = -20.553895852207432;
		motionfield[105] = -20.404700810551105;
		motionfield[106] = -20.255505768894807;
		motionfield[107] = -20.106310727238508;
		motionfield[108] = -19.95711568558218;
		motionfield[109] = -19.807920643925883;
		motionfield[110] = -19.658725602269584;
		motionfield[111] = -19.509530560613257;
		motionfield[112] = -19.255930117765246;
		motionfield[113] = -18.98927899976826;
		motionfield[114] = -18.722627881771274;
		motionfield[115] = -18.455976763774288;
		motionfield[116] = -18.18932564577733;
		motionfield[117] = -17.922674527780345;
		motionfield[118] = -17.65602340978336;
		motionfield[119] = -17.3893722917864;
		motionfield[120] = -17.122721173789444;
		motionfield[121] = -16.85607005579243;
		motionfield[122] = -16.58941893779547;
		motionfield[123] = -16.322767819798514;
		motionfield[124] = -16.0561167018015;
		motionfield[125] = -15.789465583804542;
		motionfield[126] = -15.522814465807528;
		motionfield[127] = -15.25616334781057;
		motionfield[128] = -14.989512229813613;
		motionfield[129] = -14.722861111816627;
		motionfield[130] = -14.456209993819641;
		motionfield[131] = -14.189558875822684;
		motionfield[132] = -13.922907757825698;
		motionfield[133] = -13.656256639828712;
		motionfield[134] = -13.369791282952463;
		motionfield[135] = -13.073418806636539;
		motionfield[136] = -12.777046330320616;
		motionfield[137] = -12.480673854004749;
		motionfield[138] = -12.184301377688797;
		motionfield[139] = -11.887928901372902;
		motionfield[140] = -11.591556425056979;
		motionfield[141] = -11.295183948741084;
		motionfield[142] = -10.998811472425189;
		motionfield[143] = -10.702438996109265;
		motionfield[144] = -10.406066519793399;
		motionfield[145] = -10.109694043477475;
		motionfield[146] = -9.813321567161552;
		motionfield[147] = -9.516949090845657;
		motionfield[148] = -9.220576614529762;
		motionfield[149] = -8.924204138213838;
		motionfield[150] = -8.627831661897943;
		motionfield[151] = -8.33145918558202;
		motionfield[152] = -8.035086709266125;
		motionfield[153] = -7.73871423295023;
		motionfield[154] = -7.442341756634335;
		motionfield[155] = -7.145969280318411;
		motionfield[156] = -6.879726764496922;
		motionfield[157] = -6.651146699293491;
		motionfield[158] = -6.42256663409006;
		motionfield[159] = -6.193986568886601;
		motionfield[160] = -5.965406503683141;
		motionfield[161] = -5.7368264384797385;
		motionfield[162] = -5.508246373276279;
		motionfield[163] = -5.279666308072848;
		motionfield[164] = -5.051086242869417;
		motionfield[165] = -4.822506177665957;
		motionfield[166] = -4.593926112462498;
		motionfield[167] = -4.365346047259095;
		motionfield[168] = -4.1367659820556355;
		motionfield[169] = -3.908185916852233;
		motionfield[170] = -3.6796058516487733;
		motionfield[171] = -3.4510257864453138;
		motionfield[172] = -3.2224457212418542;
		motionfield[173] = -2.9938656560384516;
		motionfield[174] = -2.765285590834992;
		motionfield[175] = -2.536705525631561;
		motionfield[176] = -2.30812546042813;
		motionfield[177] = -2.0795453952246987;
		motionfield[178] = -1.8827432910657933;
		motionfield[179] = -1.7971640505627988;
		motionfield[180] = -1.7115848100598043;
		motionfield[181] = -1.6260055695568099;
		motionfield[182] = -1.5404263290538438;
		motionfield[183] = -1.454847088550821;
		motionfield[184] = -1.3692678480478548;
		motionfield[185] = -1.283688607544832;
		motionfield[186] = -1.1981093670418659;
		motionfield[187] = -1.1125301265388714;
		motionfield[188] = -1.026950886035877;
		motionfield[189] = -0.9413716455328824;
		motionfield[190] = -0.855792405029888;
		motionfield[191] = -0.7702131645268935;
		motionfield[192] = -0.6846339240239274;
		motionfield[193] = -0.5990546835209329;
		motionfield[194] = -0.5134754430179385;
		motionfield[195] = -0.427896202514944;
		motionfield[196] = -0.3423169620119779;
		motionfield[197] = -0.256737721508955;
		motionfield[198] = -0.17115848100598896;
		motionfield[199] = -0.08557924050296606;

		Pointer diaMotion = Pointer.to(new int[] { (int) (Math.round(motionfield[projectionNumber])) });

		if (projectionNumber == 199) {
			VisualizationUtil.createPlot(motionfield, "motionfield real", "frame", "z").show();
			VisualizationUtil.createPlot(mot, "motionfield scaled", "frame", "z").show();
			VisualizationUtil.createPlot(motionfieldShifted, "shifted", "frame", "z").show();
		}
		//Heart respiration compensation

		/*
		 * Lung Compensation		
		 */ /*  MotionFieldReader motion = null;
				try {
				motion = new MotionFieldReader();
				} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				}*/
		//double diaMot = Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(projectionNumber);

		//double diaPos = Configuration.getGlobalConfiguration().getDiaphragmPositionFieldEntry(projectionNumber);
		//float fMot = (float) (diaMot/Configuration.getGlobalConfiguration().getMaxMotion());
		//float amplitude = 0.5f;
		//Pointer m = Pointer.to(new float[]{(float) motion.getGlobalCompensationLinearScaling()});
		//Pointer m = Pointer.to(new float[]{0.f});
		//Pointer m = Pointer.to(new float[]{(float) motion.getGlobalCompensationLinearMinMax(fMot, diaPos, reconDimensionZ)});
		//System.out.println(motion.getGlobalCompensationLinearMinMax(fMot, diaPos, reconDimensionZ*voxelSpacingZ-offsetZ));
		//Pointer m = Pointer.to(new float[]{(float) motion.getInterpolatedGlobalCompensationLinearScaling(fMot)});
		//System.out.println(motion.getInterpolatedGlobalCompensationLinearScaling(fMot));
		//Pointer diaPosition = Pointer.to(new float[]{(float) diaPos});
		//Pointer diaMotion = Pointer.to(new float[]{(float)(-amplitude*fMot) });	
		//for old constant version set m to zero		

		//	Pointer interX = Pointer.to(motion.getBilinearInterpolationDataPosition(diaMot));
		//	Pointer interY = Pointer.to(motion.getBilinearInterpolationDataMotion(diaMot));

		//		System.out.println((int) Math.round(Configuration.getGlobalConfiguration().getRespiratoryMotionFieldEntry(projectionNumber)));
		Pointer dOut = Pointer.to(volumePointer);
		Pointer pWidth = Pointer.to(new int[] { (int) lineOffset });
		Pointer pZOffset = Pointer.to(zed);
		float[] vsx = new float[] { (float) voxelSpacingX };
		Pointer pvsx = Pointer.to(vsx);
		Pointer pvsy = Pointer.to(new float[] { (float) voxelSpacingY });
		Pointer pvsz = Pointer.to(new float[] { (float) voxelSpacingZ });
		Pointer pox = Pointer.to(new float[] { (float) offsetX });
		Pointer poy = Pointer.to(new float[] { (float) offsetY });
		Pointer poz = Pointer.to(new float[] { (float) offsetZ });

		int offset = 0;
		//System.out.println(dimz + " " + zed[0] + " " + offsetZ + " " + voxelSpacingZ);

		offset = CUDAUtil.align(offset, Sizeof.POINTER);
		JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
		offset += Sizeof.POINTER;
		/*// new for lung motion
		offset = CUDAUtil.align(offset,Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, m, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;
		//new for lung motion
		offset = CUDAUtil.align(offset,Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, diaPosition, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;
		 */
		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, diaMotion, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, pWidth, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, pZOffset, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsx, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsy, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsz, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pox, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, poy, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, poz, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		JCudaDriver.cuParamSetSize(function, offset);

		// Call the CUDA kernel, writing the results into the volume which is pointed at
		JCudaDriver.cuFuncSetBlockShape(function, bpBlockSize[0], bpBlockSize[1], 1);
		JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
		JCudaDriver.cuCtxSynchronize();

	}

	public void cudaRun() {
		try {
			while (projectionsAvailable.size() > 0) {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				if (showStatus) {
					float status = (float) (1.0 / projections.size());
					if (largeVolumeMode) {
						IJ.showStatus("Streaming Projections to CUDA Buffer");
					} else {
						IJ.showStatus("Backprojecting with CUDA");
					}
					IJ.showProgress(status);
				}
				if (!largeVolumeMode) {
					workOnProjectionData();
				} else {
					checkProjectionData();
				}
			}
			System.out.println("large Volume " + largeVolumeMode);
			if (largeVolumeMode) {
				// we have collected all projections.
				// now we can reconstruct subvolumes and stich them together.
				int reconDimensionX = getGeometry().getReconDimensionX();
				int reconDimensionY = getGeometry().getReconDimensionY();
				int reconDimensionZ = getGeometry().getReconDimensionZ();
				double voxelSpacingX = getGeometry().getVoxelSpacingX();
				double voxelSpacingY = getGeometry().getVoxelSpacingY();
				double voxelSpacingZ = getGeometry().getVoxelSpacingZ();
				useVOImap = false;
				initialize(projections.get(0));
				double originalOffsetZ = offsetZ;
				double originalReconDimZ = reconDimensionZ;
				reconDimensionZ = subVolumeZ;
				int memorysize = reconDimensionX * reconDimensionY * subVolumeZ * Sizeof.FLOAT;
				int maxProjectionNumber = projections.size();
				float all = nSteps * maxProjectionNumber * 2;
				for (int n = 0; n < nSteps; n++) { // For each subvolume
					// set all to 0;
					Arrays.fill(h_volume, 0);
					JCudaDriver.cuMemcpyHtoD(volumePointer, Pointer.to(h_volume), memorysize);
					offsetZ = originalOffsetZ - (reconDimensionZ * voxelSpacingZ * n);
					for (int p = 0; p < maxProjectionNumber; p++) { // For all projections
						float currentStep = (n * maxProjectionNumber * 2) + p;
						if (showStatus) {
							IJ.showStatus("Backprojecting with CUDA");
							IJ.showProgress(currentStep / all);
						}
						//System.out.println("Current: " + p);
						try {
							projectSingleProjection(p, reconDimensionZ);
						} catch (Exception e) {
							System.out.println("Backprojection of projection " + p + " was not successful.");
							e.printStackTrace();
						}
					}
					// Gather volume
					JCudaDriver.cuMemcpyDtoH(Pointer.to(h_volume), volumePointer, memorysize);

					// move data to ImagePlus;
					if (projectionVolume != null) {
						for (int k = 0; k < reconDimensionZ; k++) {
							int index = (n * subVolumeZ) + k;
							if (showStatus) {
								float currentStep = (n * maxProjectionNumber * 2) + maxProjectionNumber + k;
								IJ.showStatus("Fetching Volume from CUDA");
								IJ.showProgress(currentStep / all);
							}
							if (index < originalReconDimZ) {
								for (int j = 0; j < projectionVolume.getSize()[1]; j++) {
									for (int i = 0; i < projectionVolume.getSize()[0]; i++) {
										float value = h_volume[(((projectionVolume.getSize()[1] * k) + j)
												* projectionVolume.getSize()[0]) + i];
										double[][] voxel = new double[4][1];
										voxel[0][0] = (voxelSpacingX * i) - offsetX;
										voxel[1][0] = (voxelSpacingY * j) - offsetY;
										voxel[2][0] = (voxelSpacingZ * index) - originalOffsetZ;
										if (interestedInVolume.contains(voxel[0][0], voxel[1][0], voxel[2][0])) {
											projectionVolume.setAtIndex(i, j, index, value);
										} else {
											projectionVolume.setAtIndex(i, j, index, 0);
										}
									}
								}
							}
						}
					}
				}
			}

		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		if (showStatus)
			IJ.showProgress(1.0);
		unload();
		if (debug)
			System.out.println("Unloaded");
	}

	private synchronized void workOnProjectionData() {
		if (projectionsAvailable.size() > 0) {
			Integer current = projectionsAvailable.get(0);
			projectionsAvailable.remove(0);
			projectSingleProjection(current.intValue(), getGeometry().getReconDimensionZ());
			projectionsDone.add(current);
		}
	}

	private synchronized void checkProjectionData() {
		if (projectionsAvailable.size() > 0) {
			Integer current = projectionsAvailable.get(0);
			projectionsAvailable.remove(current);
			projectionsDone.add(current);
		}
	}

	public void reconstructOffline(ImagePlus imagePlus) throws Exception {
		ImagePlusDataSink sink = new ImagePlusDataSink();
		configure();
		init();
		for (int i = 0; i < imagePlus.getStackSize(); i++) {
			backproject(ImageUtil.wrapImageProcessor(imagePlus.getStack().getProcessor(i + 1)), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling())
			applyHounsfieldScaling();
		int[] size = projectionVolume.getSize();
		System.out.println(size[0] + " " + size[1] + " " + size[2]);
		for (int k = 0; k < projectionVolume.getSize()[2]; k++) {
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		sink.close();
		ImagePlus revan = ImageUtil.wrapGrid3D(sink.getResult(),
				"Compensated Reconstruction of" + imagePlus.getTitle());
		revan.setTitle(imagePlus.getTitle() + " reconstructed");
		revan.show();
		reset();
	}

	@Override
	protected void reconstruct() throws Exception {
		init();
		for (int i = 0; i < nImages; i++) {
			backproject(inputQueue.get(i), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling())
			applyHounsfieldScaling();
		int[] size = projectionVolume.getSize();

		for (int k = 0; k < size[2]; k++) {
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		sink.close();
	}

	@Override
	public String getToolName() {
		return "CUDA Compensated Backprojector";
	}

}

/*
 * Copyright (C) 2010-2014 - Marco Bögel 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
