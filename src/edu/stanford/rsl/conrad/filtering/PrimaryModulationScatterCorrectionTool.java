package edu.stanford.rsl.conrad.filtering;


//import java.awt.Rectangle;

import java.awt.geom.Point2D;
import java.awt.geom.Point2D.Float;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;


import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;



import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;

import ij.process.FloatProcessor;

import ij.process.ImageProcessor;


import static java.lang.Math.*;

/**
 * @author HEWEI GAO 
 * 
 * <p>THE principle and method of the primary modulation method were described 
 * in Lei Zhu and Hewei Gao's publications: 
 * 
 * <ul>
 * <li>[1] L. Zhu, N. R. Bennett, and R. Fahrig, "Scatter correction method for 
 * X-ray CT using primary modulation: Theory and preliminary results," IEEE Trans. 
 * Med. Imag., vol. 25, no. 12, pp. 1573-1587, 2006. 
 * (<a href="http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4016177">link</a>);
 * 
 * <li>[2] H. Gao, R. Fahrig, N. R. Bennett, M. Sun, J. Star-Lack, and L. Zhu, 
 * "Scatter correction method for x-ray CT using primary modulation: 
 * Phantom studies, Med. Phys., vol. 37, no. 2, pp. 934-946, 2010. 
 * (<a href="http://scitation.aip.org/content/aapm/journal/medphys/37/2/10.1118/1.3298014">link</a>)
 * </ul>
 * 
 * <p>THE original algorithm of scatter estimation [primary-modulation implementation (PMI) algorithm] 
 * is written as 
 * <p> {@latex.ilb \\[ 
 * \\mathbf{S}_{\\textit{PMI}}(\\omega) = \\mathbf{P'}(\\omega)\\mathbf{H}(\\omega) - 
 *  \\frac{1+\\alpha}{1-\\alpha}\\mathbf{P'}(\\omega-\\pi)\\mathbf{H}(\\omega)
 * \\]}
 * <p>where, {@latex.inline $\\mathbf{P'}(\\omega)$} is the Fourier transform of the <b>DOWNSAMPLED</b> primary-modulated X-ray image with 
 * the primary modulator in place, {@latex.inline $\\mathbf{H}(\\omega)$} is the low pass filter, and 
 * {@latex.inline $\\alpha$} is the transmission factor of the primary modulator. 
 * 
 * <p>IN stead of the above algorithm, however, this tool implements an improved algorithm
 * using the virtual scatter modulation [scatter-modulation implementation (SMI) algorithm] as
 * <p> {@latex.ilb \\[ 
 * \\mathbf{S}_{\\textit{SMI}}(\\omega) = -\\frac{2\\alpha}{1-\\alpha}\\mathbf{P''}(\\omega-\\pi)\\mathbf{H}(\\omega)
 * \\]} 
 * where, {@latex.inline $\\mathbf{P''}(\\omega)$} is the Fourier transform of the <b>DOWNSAMPLED</b> virtual scatter-modulated
 * X-ray image obtained by dividing the primary-modulated X-ray image by the modulation function. A comparative
 * study of the two algorithms were conducted in a manuscript (Hewei Gao, et al, "A New Algorithm of Scatter Estimation for X-ray 
 * Scatter Correction Using Primary Modulation"). 
 * 
 * <p>THIS tool is rewritten from the matlab codes used in most of data processing on Hewei's computer 
 * ("Step_1_IndexPrep.m", "Step_2_ScatterEst.m", "Step_3_smooth_multiProj.m", and "Step_4_Subtraction.m"), such as  
 * <ul>
 * <li>"D:\Works\SC_Modulator\03-19-2009\Process_AlMod_CTP404_PhantomStudies\" 
 * 
 * <li>"C:\PMSCData\2010-06-26\Process_ErMod_0p35_2D\"
 * </ul>
 * 
 * 
 * 
 */
public class PrimaryModulationScatterCorrectionTool extends
MultiProjectionFilter {

	private static final long serialVersionUID = 7870494646455524074L;

	private int meanfilterLen;  //the length of the mean filter used to smooth the input X-ray image, default is 3     
	private double lowPassFilterWidth; //the frequency cutoff of the low pass filter H(w) in x (horizontal) direction, default is 1/7
	private double lowPassFilterHeight; //the frequency cutoff of the low pass filter H(w) in z (vertical) direction, default is 1/5
	private double gaincutfactor; //a factor to refine the modulation function (i.e., \alpha), default is 0.05 
	private double scatterEstScale; //a factor to scale the estimated scatter distribution, default is 0.9
	private boolean bUniformAlpha; //true to use a uniform \alpha (the average of \alpha's over the whole image), false to use the spatially non-uniform \alpha, default is true
	private double softcutfactor;  //a factor to avoid negative result after scatter removal, default is 0.8, see Lei's TMI paper.
	
	private String modulationFunctionFilename; //the file name of the scan of the primary modulator with no object in the FOV 
	private ImageGridBuffer scatterEstimatesBuffer = new ImageGridBuffer(); //Buffer to store the estimated scatter distribution
	private static ImagePlus modulationFunction = null; //the modulation function from the modulationFunctionFilename file
	private FloatProcessor blkIdxX = null; //A 2D array To store the x coordinates of the downsampling points in the X-ray image
	private FloatProcessor blkIdxY = null; //A 2D array store the y coordinates of the downsampling points in the X-ray image
	private FloatProcessor blkIdxVal = null; //A 2D array To store the value (+1 for non-blocker center, -1 for blocker center) of the downsampling points in the X-ray image
	private FloatProcessor dsModFun = null; //the downsampled modulation function
	private double dsModFunMean = 0; //the mean value of the downsampled modulation function
	private FloatProcessor upsampIdxX = null; //A 2D array to store the x coordinates for the upsampling procedure
	private FloatProcessor upsampIdxY = null; //A 2D array to store the y coordinates for the upsampling procedure
	private FloatProcessor alphaDist = null; //A 2D array to store the non-uniform \alpha's 
	private boolean initModulation = false; //true to initialize the modulation parameters 


	/**
	 * @return the low Pass Filter Width
	 */
	public double getLowPassFilterWidth() {
		return lowPassFilterWidth;
	}


	/**
	 * @param lPassFilterWidth the lowPassFilterWidth to set
	 */
	public void setLowPassFilterWidth(double lPassFilterWidth) {
		lowPassFilterWidth = lPassFilterWidth;
	}


	/**
	 * @return the lowPassFilterHeight
	 */
	public double getLowPassFilterHeight() {
		return lowPassFilterHeight;
	}


	/**
	 * @param lowPassFilterHeight the lowPassFilterHeight to set
	 */
	public void setLowPassFilterHeight(double lowPassFilterHeight) {
		this.lowPassFilterHeight = lowPassFilterHeight;
	}
	
	/**
	 * Initialization for the primary modulation method
	 */
	public PrimaryModulationScatterCorrectionTool (){
		
		meanfilterLen = 3;
		lowPassFilterWidth = 1.0f/7;
		lowPassFilterHeight = 1.0f/5;
		gaincutfactor = 0.05;
		scatterEstScale = 1.0;
		bUniformAlpha = false;
		softcutfactor = 0.8;
	}
	
	
	/**
	 * Serialization for the primary modulation method
	 */
	@Override
	public void prepareForSerialization(){
		modulationFunction = null;
		blkIdxX = null;
		blkIdxY = null;
		blkIdxVal = null;
		dsModFun = null;
		upsampIdxX = null;
		upsampIdxY = null;
		alphaDist = null;
		dsModFunMean = 0;
		scatterEstimatesBuffer = null;
		initModulation = false;
		super.prepareForSerialization();

	}

	/**
	 * Loads the modulation function from disk if required. Will buffer the image for later use.
	 * 
	 * @param filename the filename
	 * @return the modulation function as ImagePlus
	 */
	public static synchronized ImagePlus getModulationFunction(String filename){
		modulationFunction = IJ.openImage(filename);
		return modulationFunction;
	}



	@Override
	public ImageFilteringTool clone() {
//		PrimaryModulationScatterCorrectionTool clone = new PrimaryModulationScatterCorrectionTool();
//		clone.modulationFunctionFilename = modulationFunctionFilename;
//		clone.modulationFunction = modulationFunction;
//		clone.blkIdxX = blkIdxX;
//		clone.blkIdxY = blkIdxY;
//		clone.blkIdxVal = blkIdxVal;
//		clone.dsModFun = dsModFun;
//		clone.upsampIdxX = upsampIdxX;
//		clone.upsampIdxY = upsampIdxY;
//		clone.alphaDist = alphaDist;
//		clone.dsModFunMean = dsModFunMean;
//		clone.setConfigured(configured);
		return this;
	}

	@Override
	public String getToolName() {
		return "Primary Modulation Scatter Correction";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{LeiZhu2006_TMI,"+
		"author={L Zhu and N.R. Bennett and R Fahrig},"+
		"title={Scatter Correction Method for X-Ray CT Using Primary Modulation: Theory and Preliminary Results},"+
		"journal={IEEE Transactions on Medical Imaging},"+
		"volume={25},"+
		"number={12},"+
		"pages={1573-1587},"+
		"url={http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4016177&tag=1},"+
		"year={2006}"+
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "L Zhu, N.R. Bennett, R Fahrig." +
		"Scatter Correction Method for X-Ray CT Using Primary Modulation: Theory and Preliminary Results." +
		"IEEE Trans. Med. Imag. 25(12):1573-1587. 2006";
	}
	
	
   /**
    * Calculate/load the downsampling (x,y) coordinates on the modulaiton function 
    * image, and the upsampling (x,y) coordinates on the downsampled image. Save 
    * them on the disk if necessary.
    */
	private void InitModFunParameters() 
	{

		//Four corner points of a selected area for downsampling   
		Point2D.Float ptTopL = new Point2D.Float(0,0); 
		Point2D.Float ptTopR = new Point2D.Float(0,0);
		Point2D.Float ptBtmL = new Point2D.Float(0,0);
		Point2D.Float ptBtmR = new Point2D.Float(0,0);
		int dsWidth = 0;  //downsampling points in width
		int dsHeight = 0;   //downsampling points in height

		          

		boolean bConf = false; 
		try{
			bConf = ReadModFunConfigFile(modulationFunctionFilename);
		}
		catch (IOException e) {
			e.printStackTrace();	
		}
		if (bConf){   //load from disk
			ReadDownSamplingPts(); //load downsample/upsample information

			dsWidth = blkIdxX.getWidth();
			dsHeight = blkIdxX.getHeight();

			ptTopL.x = blkIdxX.getPixelValue(0, 0);
			ptTopR.x = blkIdxX.getPixelValue(dsWidth-1, 0);
			ptBtmL.x = blkIdxX.getPixelValue(0, dsHeight-1);
			ptBtmR.x = blkIdxX.getPixelValue(dsWidth-1, dsHeight-1);

			ptTopL.y = blkIdxY.getPixelValue(0, 0);
			ptTopR.y = blkIdxY.getPixelValue(dsWidth-1, 0);
			ptBtmL.y = blkIdxY.getPixelValue(0, dsHeight-1);
			ptBtmR.y = blkIdxY.getPixelValue(dsWidth-1, dsHeight-1);

		}
		else{        //calculate from the modulation function file
			
			//specify the effective downsampling area and the total number of points
			
			// ErMod_0p35_2D_Wide_Ave128.viv
			ptTopL = new Point2D.Float(25,13);
			ptTopR = new Point2D.Float(1012,21);
			ptBtmL = new Point2D.Float(20,739);
			ptBtmR = new Point2D.Float(1005,747);
			dsWidth = 73;
			dsHeight = 54;
			
			// ErMod_0p35_2D_(120,13,35)_NoFilter_Ave128.viv
			/*		Point2D.Float ptTopL = new Point2D.Float(22,14);
			Point2D.Float ptTopR = new Point2D.Float(1006,20);
			Point2D.Float ptBtmL = new Point2D.Float(16,751);
			Point2D.Float ptBtmR = new Point2D.Float(1000,759);
			int dsWidth = 73;
			int dsHeight = 55;*/


			// CuMod_0p35_6oz_2D_Wide_Ave128.viv
/*			ptTopL = new Point2D.Float(9,22);
			ptTopR = new Point2D.Float(999,18);
			ptBtmL = new Point2D.Float(13,751);
			ptBtmR = new Point2D.Float(1002,745);
			dsWidth = 73;
			dsHeight = 54;*/


			// CuMod_0p35_3oz_2D_Wide_Ave128.viv
/*			ptTopL = new Point2D.Float(16,20);
			ptTopR = new Point2D.Float(1007,15);
			ptBtmL = new Point2D.Float(19,749);
			ptBtmR = new Point2D.Float(1009,744);
			dsWidth = 73;
			dsHeight = 54;   */     
			
/*			// ErMod_0p35_2D_Wide_Ave128_Chest.viv                           
			ptTopL = new Point2D.Float(12,13);
			ptTopR = new Point2D.Float(1013,20);
			ptBtmL = new Point2D.Float(8,739);
			ptBtmR = new Point2D.Float(1007,746);
			dsWidth = 74;
			dsHeight = 54;    */                               

			GenericDialog gd = new GenericDialog("Downsampling points Setting");
			gd.addNumericField("Top Left  x:", ptTopL.x, 1);
			gd.addNumericField("  y:", ptTopL.y, 1);
			gd.addNumericField("Top Right  x:", ptTopR.x, 1);
			gd.addNumericField("  y:", ptTopR.y, 1);
			gd.addNumericField("Btm Left  x:", ptBtmL.x, 1);
			gd.addNumericField("  y:", ptBtmL.y, 1);
			gd.addNumericField("Btm Right  x:", ptBtmR.x, 1);
			gd.addNumericField("  y:", ptBtmR.y, 1);
			gd.addNumericField("No. pts x:", dsWidth, 0);
			gd.addNumericField(" y:", dsHeight, 0);

			gd.showDialog();
			if (gd.wasCanceled()){
				throw new RuntimeException("User cancelled selection.");
			} 
			ptTopL.x = (float)gd.getNextNumber();
			ptTopL.y = (float)gd.getNextNumber();
			ptTopR.x = (float)gd.getNextNumber();
			ptTopR.y = (float)gd.getNextNumber();
			ptBtmL.x = (float)gd.getNextNumber();
			ptBtmL.y = (float)gd.getNextNumber();
			ptBtmR.x = (float)gd.getNextNumber();
			ptBtmR.y = (float)gd.getNextNumber();
			dsWidth = (int)gd.getNextNumber();
			dsHeight = (int)gd.getNextNumber();


			if (blkIdxX == null) blkIdxX = new FloatProcessor(dsWidth, dsHeight);
			if (blkIdxY == null) blkIdxY = new FloatProcessor(dsWidth, dsHeight);
			if (blkIdxVal == null) blkIdxVal = new FloatProcessor(dsWidth, dsHeight);
			if (upsampIdxX == null) upsampIdxX = new FloatProcessor(modulationFunction.getWidth(),modulationFunction.getHeight());
			if (upsampIdxY == null) upsampIdxY = new FloatProcessor(modulationFunction.getWidth(),modulationFunction.getHeight());

			float idxTopx = 0;
			float idxTopy = 0;
			float idxBtmx = 0;
			float idxBtmy = 0;

			float crntPtx = 0;
			float crntPty = 0;

			for (int i=0;i<dsWidth;i++) // uniformly sample the points column by column
			{

				idxTopx = ptTopL.x + (ptTopR.x - ptTopL.x)/(dsWidth-1)*i;
				idxTopy = ptTopL.y + (ptTopR.y - ptTopL.y)/(dsWidth-1)*i;
				idxBtmx = ptBtmL.x + (ptBtmR.x - ptBtmL.x)/(dsWidth-1)*i;
				idxBtmy = ptBtmL.y + (ptBtmR.y - ptBtmL.y)/(dsWidth-1)*i;

				for (int j=0;j<dsHeight;j++) {
					crntPtx = idxTopx + (idxBtmx - idxTopx)/(dsHeight-1)*j;
					crntPty = idxTopy + (idxBtmy - idxTopy)/(dsHeight-1)*j;			
					
					blkIdxX.putPixelValue(i, j, crntPtx);
					blkIdxY.putPixelValue(i, j, crntPty);
					if ((i+j)%2==0){
						blkIdxVal.putPixelValue(i, j, 1);
					}
					else {


						blkIdxVal.putPixelValue(i, j, -1);
					}
				}
			}

			Point2D.Float fIndex = new Point2D.Float();
			Point2D.Float[][] fMinBlk = null;

			//calculate the minimum upsampling points 
			fMinBlk = FindMinBlkPoints(fMinBlk);

			int nDestW = modulationFunction.getWidth();
			int nDestH = modulationFunction.getHeight();
			int nMinW = fMinBlk[0].length;		
			int nMinH = fMinBlk.length;
			int nOrigW = blkIdxX.getWidth();
			int nOrigH = blkIdxY.getHeight();

			int nDiffW = (nOrigW - nMinW)/2;
			int nDiffH = (nOrigH - nMinH)/2;

			for (int i=0;i<nDestW;i++){
				for (int j=0;j<nDestH;j++){				
					//calculate the coordinates for interpolation in upsampling.
					fIndex = FindPosition(fMinBlk,new Point2D.Float(i,j));
					float x = fIndex.x+nDiffW;
					if (x<=0){
						x=0;
					}
					else if (x>=nOrigW-1){ 
						x = nOrigW-1;
					}
					float y = fIndex.y+nDiffH;
					if(y<=0) {
						y=0;
					}
					else if (y>=nOrigH-1){
						y = nOrigH-1;
					}
					upsampIdxX.putPixelValue(i, j, x);
					upsampIdxY.putPixelValue(i, j, y);
				}
			}
			try {
				WriteModFunConfigFile(modulationFunctionFilename);
				SaveDownSamplingPts();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	/**
	 * 
	 * @param ptTopL Top left point
	 * @param ptTopR Top right point
	 * @param ptBtmL Bottom left point
	 * @param ptBtmR Bottom right point
	 * @param nWidth  upsample width
	 * @param nHeight upsample height
	 * @param fMinBlk the minimum area size needed for upsampling
	 * @return the next points needed for upsampling
	 */
	private Point2D.Float[][] GetNextPoints(Point2D.Float [] ptTopL,
			Point2D.Float [] ptTopR,
			Point2D.Float [] ptBtmL,
			Point2D.Float [] ptBtmR,
			int []  nWidth,
			int [] nHeight,
			Point2D.Float[][] fMinBlk)
	{
		int nOrigW = blkIdxX.getWidth();
		int nOrigH = blkIdxX.getHeight();


		Point2D.Float ptTopL0 = new Point2D.Float(blkIdxX.getPixelValue(0, 0),blkIdxY.getPixelValue(0, 0));
		Point2D.Float ptTopR0 = new Point2D.Float(blkIdxX.getPixelValue(nOrigW-1, 0),blkIdxY.getPixelValue(nOrigW-1, 0));
		Point2D.Float ptBtmL0 = new Point2D.Float(blkIdxX.getPixelValue(0, nOrigH-1),blkIdxY.getPixelValue(0, nOrigH-1));
		Point2D.Float ptBtmR0 = new Point2D.Float(blkIdxX.getPixelValue(nOrigW-1, nOrigH-1),blkIdxY.getPixelValue(nOrigW-1, nOrigH-1));

		float dTopx = (ptTopR0.x - ptTopL0.x)/(nOrigW-1);
		float dTopy = (ptTopR0.y - ptTopL0.y)/(nOrigW-1);

		float dBtmx = (ptBtmR0.x - ptBtmL0.x)/(nOrigW-1);
		float dBtmy = (ptBtmR0.y - ptBtmL0.y)/(nOrigW-1);

		ptTopL[0].x = ptTopL0.x - dTopx * ((nWidth[0]-nOrigW)/2+1);
		ptTopL[0].y = ptTopL0.y - dTopy * ((nWidth[0]-nOrigW)/2+1);		
		ptBtmL[0].x = ptBtmL0.x - dBtmx * ((nWidth[0]-nOrigW)/2+1);
		ptBtmL[0].y = ptBtmL0.y - dBtmy * ((nWidth[0]-nOrigW)/2+1);

		ptTopR[0].x = ptTopR0.x + dTopx * ((nWidth[0]-nOrigW)/2+1);
		ptTopR[0].y = ptTopR0.y + dTopy * ((nWidth[0]-nOrigW)/2+1);
		ptBtmR[0].x = ptBtmR0.x + dBtmx * ((nWidth[0]-nOrigW)/2+1);
		ptBtmR[0].y = ptBtmR0.y + dBtmy * ((nWidth[0]-nOrigW)/2+1);

		nWidth[0] = nWidth[0] + 2;

		float dLx = (ptBtmL[0].x-ptTopL[0].x)/(nOrigH-1);
		float dLy = (ptBtmL[0].y-ptTopL[0].y)/(nOrigH-1);

		float dRx = (ptBtmR[0].x - ptTopR[0].x)/(nOrigH-1);
		float dRy = (ptBtmR[0].y - ptTopR[0].y)/(nOrigH-1);


		ptTopL[0].x = ptTopL[0].x - dLx * ((nHeight[0]-nOrigH)/2+1);
		ptTopL[0].y = ptTopL[0].y - dLy * ((nHeight[0]-nOrigH)/2+1);
		ptBtmL[0].x = ptBtmL[0].x + dLx * ((nHeight[0]-nOrigH)/2+1);
		ptBtmL[0].y = ptBtmL[0].y + dLy * ((nHeight[0]-nOrigH)/2+1);

		ptTopR[0].x = ptTopR[0].x - dRx * ((nHeight[0]-nOrigH)/2+1);
		ptTopR[0].y = ptTopR[0].y - dRy * ((nHeight[0]-nOrigH)/2+1);
		ptBtmR[0].x = ptBtmR[0].x + dRx * ((nHeight[0]-nOrigH)/2+1);
		ptBtmR[0].y = ptBtmR[0].y + dRy * ((nHeight[0]-nOrigH)/2+1);

		nHeight[0] = nHeight[0] + 2;

		Point2D.Float [][] fTemp  = new Point2D.Float[nHeight[0]][nWidth[0]];

		for (int i=1;i<nHeight[0]-1;i++){
			for (int j=1;j<nWidth[0]-1;j++){
				fTemp[i][j] = new Point2D.Float(fMinBlk[i-1][j-1].x,fMinBlk[i-1][j-1].y);
			}
		}

		for (int i=0;i<nWidth[0];i++){
			fTemp[0][i] = new Point2D.Float(ptTopL[0].x + (ptTopR[0].x - ptTopL[0].x)/(nWidth[0]-1)*i,
					ptTopL[0].y + (ptTopR[0].y - ptTopL[0].y)/(nWidth[0]-1)*i);
			fTemp[nHeight[0]-1][i] = new Point2D.Float(ptBtmL[0].x + (ptBtmR[0].x - ptBtmL[0].x)/(nWidth[0]-1)*i,
					ptBtmL[0].y + (ptBtmR[0].y - ptBtmL[0].y)/(nWidth[0]-1)*i);			
		}
		for (int i=0;i<nHeight[0];i++){
			fTemp[i][0] = new Point2D.Float(ptTopL[0].x + (ptBtmL[0].x - ptTopL[0].x)/(nHeight[0]-1)*i,
					ptTopL[0].y + (ptBtmL[0].y - ptTopL[0].y)/(nHeight[0]-1)*i);
			fTemp[i][nWidth[0]-1] = new Point2D.Float(ptTopR[0].x + (ptBtmR[0].x - ptTopR[0].x)/(nHeight[0]-1)*i,
					ptTopR[0].y + (ptBtmR[0].y - ptTopR[0].y)/(nHeight[0]-1)*i);
		}
		return fTemp;
	}

	/**
	 * 
	 * @param fMinBlk the upsampling area size
	 * @return the upsampling area size
	 */
	private Point2D.Float[][] FindMinBlkPoints(Point2D.Float[][] fMinBlk)
	{
		int nOrigW = blkIdxX.getWidth();
		int nOrigH = blkIdxX.getHeight();

		int nDestW = modulationFunction.getWidth();
		int nDestH = modulationFunction.getHeight();

		Point2D.Float ptTopL = new Point2D.Float(blkIdxX.getPixelValue(0, 0),blkIdxY.getPixelValue(0, 0));
		Point2D.Float ptTopR = new Point2D.Float(blkIdxX.getPixelValue(nOrigW-1, 0),blkIdxY.getPixelValue(nOrigW-1, 0));
		Point2D.Float ptBtmL = new Point2D.Float(blkIdxX.getPixelValue(0, nOrigH-1),blkIdxY.getPixelValue(0, nOrigH-1));
		Point2D.Float ptBtmR = new Point2D.Float(blkIdxX.getPixelValue(nOrigW-1, nOrigH-1),blkIdxY.getPixelValue(nOrigW-1, nOrigH-1));

		Point2D.Float [] ptTopLNew = new Point2D.Float[1];
		Point2D.Float [] ptTopRNew = new Point2D.Float[1];
		Point2D.Float [] ptBtmLNew = new Point2D.Float[1];
		Point2D.Float [] ptBtmRNew = new Point2D.Float[1];
		int [] nWidthNew = new int[1];
		int [] nHeightNew = new int[1];


		ptTopLNew[0] = (Float) ptTopL.clone();
		ptTopRNew[0] = (Float) ptTopR.clone();
		ptBtmLNew[0] = (Float) ptBtmL.clone();
		ptBtmRNew[0] = (Float) ptBtmR.clone();

		nWidthNew[0] = nOrigW;
		nHeightNew[0] = nOrigH;


		fMinBlk = new Point2D.Float[nOrigH][nOrigW];

		for (int i=0;i<nOrigH;i++){
			for (int j=0;j<nOrigW;j++){
				fMinBlk[i][j] = new Point2D.Float(blkIdxX.getPixelValue(j, i),blkIdxY.getPixelValue(j, i));
			}
		}

		boolean bInside = false;
		while (bInside==false) //determine if a point is inside a polygon
		{
			fMinBlk = GetNextPoints(ptTopLNew,ptTopRNew,ptBtmLNew,ptBtmRNew,nWidthNew,nHeightNew,fMinBlk);
			if (InsidePolygon(ptTopLNew[0],ptTopRNew[0],ptBtmLNew[0],ptBtmRNew[0],new Point2D.Float(0,0))==false)
			{
				bInside = false;
			}
			else if (InsidePolygon(ptTopLNew[0],ptTopRNew[0],ptBtmLNew[0],ptBtmRNew[0],new Point2D.Float(nDestW-1,0))==false)
			{
				bInside = false;
			}
			else if (InsidePolygon(ptTopLNew[0],ptTopRNew[0],ptBtmLNew[0],ptBtmRNew[0],new Point2D.Float(0,nDestH-1))==false)
			{
				bInside = false;
			}
			else if (InsidePolygon(ptTopLNew[0],ptTopRNew[0],ptBtmLNew[0],ptBtmRNew[0],new Point2D.Float(nDestW-1,nDestH-1))==false)
			{
				bInside = false;
			}
			else {
				bInside = true;
			}
		}

		return fMinBlk;
	}
	
	/**
	 * 
	 * @param ptTopL Top Left corner points for a polygon
	 * @param ptTopR Top Right corner points for a polygon
	 * @param ptBtmL Bottom Left corner points for a polygon
	 * @param ptBtmR Bottom Right corner points for a polygon
	 * @param p the point to test if its inside the polygon
	 * @return true if it is inside the polygon 
	 */
	private boolean InsidePolygon(Point2D.Float ptTopL,
			Point2D.Float ptTopR,
			Point2D.Float ptBtmL,
			Point2D.Float ptBtmR,
			Point2D.Float p)
	{
		//if the p point is inside the polygon, then the areas of the four triangles should not be negative
		if (AreaTriangle(ptTopL,ptTopR,p)<0) 
			return false;
		if (AreaTriangle(ptTopR,ptBtmR,p)<0)
			return false;
		if (AreaTriangle(ptBtmR,ptBtmL,p)<0)
			return false;
		if (AreaTriangle(ptBtmL,ptTopL,p)<0)
			return false;

		return true;
	}

	/**
	 * 
	 * @param pt0 point 0 for a rectangle in clockwise order
	 * @param pt1 point 1 for a rectangle in clockwise order
	 * @param pt2 point 2 for a rectangle in clockwise order
	 * @return the area of the rectangle
	 */
	private double AreaTriangle(Point2D.Float pt0, Point2D.Float pt1, Point2D.Float pt2)
	{
		return (double) (0.5*(pt1.x * pt2.y - pt1.y * pt2.x - pt0.x * pt2.y + pt0.y * pt2.x + pt0.x * pt1.y - pt0.y * pt1.x));
	}

	/**
	 * 
	 * @param fMinBlock the minimum area size needed for upsampling
	 * @param p the index in the upsampled image
	 * @return the (x,y) coordinates in the downsampled image corresponds the index in the upsampled image
	 */
	private Point2D.Float FindPosition(Point2D.Float[][] fMinBlock,Point2D.Float p)
	{
		int nW = fMinBlock[0].length;
		int nH = fMinBlock.length;

		Point2D.Float p1 = null;
		Point2D.Float p2 = null;
		Point2D.Float p3 = null;
		Point2D.Float p4 = null;	
		double fA1 = 0;
		double fA2 = 0;
		double fA3 = 0;
		double fA4 = 0;
		Point2D.Float fW = new Point2D.Float();
		int index=0;
		int i=0;
		int j=0;
		boolean bFlag = true;
		while(bFlag)
		{
			i = (int)(index/(nW-1));
			j = index%(nW-1);

			p1 = fMinBlock[i][j];
			p2 = fMinBlock[i][j+1];
			p3 = fMinBlock[i+1][j];
			p4 = fMinBlock[i+1][j+1];
			if(InsidePolygon(p1,p2,p3,p4,p)==true){
				fA1 = AreaTriangle(p1,p2,p);
				fA2 = AreaTriangle(p2,p4,p);
				fA3 = AreaTriangle(p4,p3,p);
				fA4 = AreaTriangle(p3,p1,p);
				fW.x = (float) (j + fA4/(fA2+fA4)); // calculate the weights for interpolation
				fW.y = (float) (i + fA1/(fA1+fA3));
				break;

			}
			else if(i>=nH-1){
				fW = new Point2D.Float(-1000,-1000);
				break;
			}
			index++;
		}

		return fW;
	}

	/**
	 * Save the downsample and upsample information for the modulation function
	 */
	private void SaveDownSamplingPts()
	{
		if(blkIdxX!=null && blkIdxY!=null && blkIdxVal!=null && upsampIdxX!=null && upsampIdxY!=null){

			IJ.saveAs(new ImagePlus("blkIdxX",blkIdxX), "tiff", modulationFunctionFilename+".blkIdxX.tif");
			IJ.saveAs(new ImagePlus("blkIdxY",blkIdxY), "tiff", modulationFunctionFilename+".blkIdxY.tif");
			IJ.saveAs(new ImagePlus("blkIdxVal",blkIdxVal), "tiff", modulationFunctionFilename+".blkIdxVal.tif");
			IJ.saveAs(new ImagePlus("upsampIdxX",upsampIdxX), "tiff", modulationFunctionFilename+".upsampIdxX.tif");
			IJ.saveAs(new ImagePlus("upsampIdxY",upsampIdxY), "tiff", modulationFunctionFilename+".upsampIdxY.tif");
		}

	}

	/**
	 * load the downsample and upsample information corresponds to the modulation function file 
	 */
	private void ReadDownSamplingPts()
	{
		File f = new File(modulationFunctionFilename);
		if (f.exists()==true){
			blkIdxX = (FloatProcessor) IJ.openImage(modulationFunctionFilename+".blkIdxX.tif").getProcessor();
			blkIdxY = (FloatProcessor) IJ.openImage(modulationFunctionFilename+".blkIdxY.tif").getProcessor();
			blkIdxVal = (FloatProcessor) IJ.openImage(modulationFunctionFilename+".blkIdxVal.tif").getProcessor();
			upsampIdxX = (FloatProcessor) IJ.openImage(modulationFunctionFilename+".upsampIdxX.tif").getProcessor();
			upsampIdxY = (FloatProcessor) IJ.openImage(modulationFunctionFilename+".upsampIdxY.tif").getProcessor();
		}
	}

   /**
    * 
    * @param filename corresponding to the modulation function file
    * @throws IOException
    */
	private void WriteModFunConfigFile(String filename) throws IOException 
	{
		String modFunfilename = filename + ".conf";

		File f = new File(modFunfilename);
		RandomAccessFile output = new RandomAccessFile(f,"rw");
		int dsW = blkIdxX.getWidth();
		int dsH = blkIdxY.getHeight();
		output.writeInt(dsW);
		output.writeInt(dsH);
		output.writeInt((int)blkIdxX.getPixelValue(0, 0));
		output.writeInt((int)blkIdxX.getPixelValue(dsW-1, 0));
		output.writeInt((int)blkIdxX.getPixelValue(0, dsH-1));
		output.writeInt((int)blkIdxX.getPixelValue(dsW-1, dsH-1));
		output.writeInt((int)blkIdxY.getPixelValue(0, 0));
		output.writeInt((int)blkIdxY.getPixelValue(dsW-1, 0));
		output.writeInt((int)blkIdxY.getPixelValue(0, dsH-1));
		output.writeInt((int)blkIdxY.getPixelValue(dsW-1, dsH-1));		
		output.close();
	}

	/**
	 * 
	 * @param filename the name of the file storing modulation function information
	 * @return true if succeeds 
	 * @throws IOException
	 */
	private boolean ReadModFunConfigFile(String filename) throws IOException 
	{
		String modFunfilename = filename + ".conf";

		File f = new File(modFunfilename);

		if (f.exists()){
			RandomAccessFile input = new RandomAccessFile(f,"r");
			int dsW =  input.readInt();
			int dsH =  input.readInt();

			if (blkIdxX == null) blkIdxX = new FloatProcessor(dsW, dsH);
			if (blkIdxY == null) blkIdxY = new FloatProcessor(dsW, dsH);

			blkIdxX.putPixelValue(0, 0,input.readInt());
			blkIdxX.putPixelValue(dsW-1, 0,input.readInt());
			blkIdxX.putPixelValue(0, dsH-1,input.readInt());
			blkIdxX.putPixelValue(dsW-1, dsH-1,input.readInt());
			blkIdxY.putPixelValue(0, 0,input.readInt());
			blkIdxY.putPixelValue(dsW-1, 0,input.readInt());
			blkIdxY.putPixelValue(0, dsH-1,input.readInt());
			blkIdxY.putPixelValue(dsW-1, dsH-1,input.readInt());
			input.close();
			return true;
		}
		else {
			return false;
		}
	}
	
	
	/**
	 * 
	 * @param fp the downsampled modulation function 
	 * @return the mean value of the modulation function
	 */
	private double GetMeanofDownSampledModFun(FloatProcessor fp){
		if(dsModFunMean!=0)
			return dsModFunMean;
		if(fp==null) {
			dsModFunMean = 0;
			return 0;
		}

		int nDsW = fp.getWidth();
		int nDsH = fp.getHeight();
		double dMean = 0;
		for (int i=0;i<nDsH;i++){
			for (int j=0;j<nDsW;j++){
				dMean += fp.getPixelValue(j, i);
			}
		}
		dsModFunMean = dMean/(nDsW*nDsH);
		return dsModFunMean;
	}

	/**
	 * 
	 * @param bUniform true if a uniform {@latex.inline $\\alpha$} is required
	 * @return the {@latex.inline $\\alpha$} distribution 
	 */
	private FloatProcessor GetAlphaDistribution(boolean bUniform)
	{
		if(modulationFunction==null||blkIdxX==null||blkIdxY==null||blkIdxVal==null)
			return null;
		int nW = modulationFunction.getWidth();
		int nH = modulationFunction.getHeight();
		if (bUniform) {
			if (alphaDist!=null) {
				return alphaDist;
			}
			else{
				alphaDist = new FloatProcessor(nW,nH);
				int nDsW = blkIdxX.getWidth();
				int nDsH = blkIdxX.getHeight();
				int pp = 0;
				int nn = 0;
				double dp = 0;
				double dn = 0;
				for (int i=0;i<nDsH;i++){
					for (int j=0;j<nDsW;j++){
						if(blkIdxVal.getPixelValue(j, i)==1){
							dp += modulationFunction.getProcessor().getInterpolatedValue(blkIdxX.getPixelValue(j, i), blkIdxY.getPixelValue(j, i));
							pp++;
						}
						else {
							dn += modulationFunction.getProcessor().getInterpolatedValue(blkIdxX.getPixelValue(j, i), blkIdxY.getPixelValue(j, i));
							nn++;
						}	
					}
				}
				double a = (dn/nn)/(dp/pp);
				
				for(int i=0;i<nH;i++){
					for(int j=0;j<nW;j++){
						alphaDist.putPixelValue(j, i, a);
					}
				}
				
			}
		}
		else {
			if (alphaDist!=null) {
				return alphaDist;
			}
			else{
				alphaDist = new FloatProcessor(nW,nH);
				
				dsModFun = (FloatProcessor)downsampleProjection(modulationFunction.getProcessor());
				int nDsW = dsModFun.getWidth();
				int nDsH = dsModFun.getHeight();
				int nZW = max(128,(int) Math.pow(2, nextPower2(nDsW)));
				int nZH = max(128,(int) Math.pow(2, nextPower2(nDsH)));
                
				// the procedure to calculate alpha distribution is the same as to estimate scatter 
				float[][] dsModFunAP = AlternatePadding(getFloatArray(dsModFun),nZW,nZH);

				float [][] LowPassFilter = Hamming(nZW,nZH,lowPassFilterWidth,lowPassFilterHeight);
				float [][] HighPassFilter = FFTUtil.fftshift(LowPassFilter, false, true);
				float [][] LowPassImage = new float[nZH][nZW];
				float [][] HighPassImage = new float[nZH][nZW];

				FFTUtil.GetLowandHighPassImage(dsModFunAP,LowPassFilter,HighPassFilter,LowPassImage,HighPassImage);

				LowPassImage = AlternatePaddingInverse(LowPassImage,nDsW,nDsH);
				HighPassImage = AlternatePaddingInverse(HighPassImage,nDsW,nDsH);
				FloatProcessor fPLowPassImage = (FloatProcessor) upsampleProjection(Array2FloatProcessor(LowPassImage));
				FloatProcessor fPHighPassImage = (FloatProcessor) upsampleProjection(Array2FloatProcessor(HighPassImage));
				
				double a = 0;
				for (int i=0;i<nH;i++){
					for (int j=0;j<nW;j++){
						a = (fPHighPassImage.getPixelValue(j, i)-fPLowPassImage.getPixelValue(j, i))/(fPLowPassImage.getPixelValue(j, i)+fPHighPassImage.getPixelValue(j, i));
						alphaDist.putPixelValue(j, i, a);
					}
				}		
			}
		}
		return alphaDist;
	}
	
	/**
	 * 
	 * @param ip a 2D array for the downsample image
	 */
	private void DownSamplePointsMark(ImageProcessor ip)
	{
		if(blkIdxVal==null) return;
		int nW = blkIdxVal.getWidth();
		int nH = blkIdxVal.getHeight();
		int x = 0;
		int y = 0;
		double val = 0;
		for (int i=0;i<nH;i++){
			for (int j=0;j<nW;j++){
				x = (int)round(blkIdxX.getPixelValue(j, i));
				y = (int)round(blkIdxY.getPixelValue(j, i));
				val = blkIdxVal.getPixelValue(j, i)*65535;
				ip.putPixelValue(x, y, val);
			}
		}		
	}
	
	@Override
	public void configure() throws Exception {
		context = 3;
		scatterEstimatesBuffer = new ImageGridBuffer();
		modulationFunctionFilename = FileUtil.myFileChoose(".viv", false);
	
		configured = true;
	}

	/**
	 * Scatter correction is not device dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	/**
	 * 
	 * @param ip the original projection image
	 * @return the downsampled image
	 */
	private ImageProcessor downsampleProjection(ImageProcessor ip)
	{
		if(blkIdxX==null) return null;
		int nW = blkIdxX.getWidth();
		int nH = blkIdxX.getHeight();
		FloatProcessor dsIp = new FloatProcessor(nW, nH);
		double val = 0;
		for (int i=0;i<nW;i++)
		{
			for(int j=0;j<nH;j++)
			{
				val = ip.getInterpolatedValue(blkIdxX.getPixelValue(i, j), blkIdxY.getPixelValue(i, j));
				dsIp.putPixelValue(i, j, val);
			}
		}
		return dsIp;
	}

	/**
	 * 
	 * @param ip the downsampled projection image
	 * @return the upsampled image
	 */
	private ImageProcessor upsampleProjection(ImageProcessor ip)
	{
		//int nOrigW = ip.getWidth();
		//int nOrigH = ip.getHeight();

		int nDestW = modulationFunction.getWidth();
		int nDestH = modulationFunction.getHeight();
		float x=0;
		float y=0;
		FloatProcessor fP = new FloatProcessor(nDestW,nDestH);
		for(int i=0;i<nDestH;i++){
			for(int j=0;j<nDestW;j++){
				x = upsampIdxX.getPixelValue(j, i);
				y = upsampIdxY.getPixelValue(j, i);
				fP.putPixelValue(j, i, ip.getInterpolatedValue(x, y));
			}
		}

		return fP;

	}

	/**
	 * convert FloatProcessor to 2D Aray
	 * @param fp the FloatProcessor to convert
	 * @return the 2D array
	 */
	private float[][] getFloatArray(FloatProcessor fp)
	{
		int W = fp.getWidth();
		int H = fp.getHeight();

		float[][] f = new float[H][W];
		for (int j=0;j<H;j++){
			for(int i=0;i<W;i++){
				f[j][i] = fp.getPixelValue(i, j);
			}
		}
		return f;
	}

	/**
	 * Convert 2D array to FloatProcessor
	 * @param f 2D array to convert
	 * @return the FloatProcessor
	 */
	private FloatProcessor Array2FloatProcessor(float[][] f)
	{
		int W = f[0].length;
		int H = f.length;
		FloatProcessor fp = new FloatProcessor(W, H);
		for (int j=0;j<H;j++){
			for(int i=0;i<W;i++){
				fp.putPixelValue(i,j,f[j][i]) ;
			}
		}
		return fp;
	}

	/**
	 * compute the power of 2 equal or larger than a
	 * @param a the number a to compute
	 * @return the power of 2 equal or larger than a
	 */
	private int nextPower2(int a)
	{
		String bits= Long.toBinaryString((long)a);
		if (bits.indexOf("1") == bits.lastIndexOf("1")){

			return bits.length() - bits.indexOf("1") - 1;
		}
		else {
			return bits.length() - bits.indexOf("1");
		}

	}

	/*private ImageProcessor AlternatePadding(ImageProcessor ip,int nZW,int nZH)
	{
		int nW = ip.getWidth();
		int nH = ip.getHeight();


		int nWm = (int)(floor(nZW/2)-floor(nW/2));
		int nHm = (int)(floor(nZH/2)-floor(nH/2));

		int i=0;
		int j=0;

		FloatProcessor zpIp = new FloatProcessor(nZW, nZH);
		for (i= nWm;i<nWm+nW;i++){
			for (j=nHm;j<nHm+nH;j++){
				zpIp.putPixelValue(i, j, ip.getPixelValue(i-nWm, j-nHm));
			}
		}

		for (j=nHm;j<nHm+nH;j++){
			for (i=0;i<nWm;i++){
				if ((nWm-i)%2==0)
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(nWm, j));
				else {
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(nWm+1, j));
				}
			}
			for (i=nWm+nW;i<nZW;i++){
				if ((i-nWm-nW)%2==0)
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(nWm+nW-2, j));
				else 
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(nWm+nW-1, j));

			}
		}

		for (i=0;i<nZW;i++){
			for (j=0;j<nHm;j++){
				if ((nHm-j)%2==0)
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(i, nHm));
				else 
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(i, nHm+1));					
			}
			for (j=nHm+nH;j<nZH;j++){
				if ((j-nHm-nH)%2==0)
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(i, nHm+nH-2));
				else 
					zpIp.putPixelValue(i, j, zpIp.getPixelValue(i, nHm+nH-1));					
			}
		}
		return zpIp;
	}
*/
	
	/**
	 * pad image size to the nextpow2. In order to match the checkerboard pattern of the modulator, the image should be alternatively padded   
	 * @param data the original image to pad
	 * @param nW the width of the image
	 * @param nH the height of the image
	 * @return the padded image
	 */
	private float[][] AlternatePadding(float[][] data,int nZW,int nZH)
	{
		int nW = data[0].length;
		int nH = data.length;


		int nWm = (int)(floor(nZW/2)-floor(nW/2));
		int nHm = (int)(floor(nZH/2)-floor(nH/2));

		int i=0;
		int j=0;

		float[][] zpData = new float[nZH][nZW];
		for (i= nWm;i<nWm+nW;i++){
			for (j=nHm;j<nHm+nH;j++){
				zpData[j][i] =data[j-nHm][i-nWm];
			}
		}

		for (j=nHm;j<nHm+nH;j++){
			for (i=0;i<nWm;i++){
				if ((nWm-i)%2==0)
					zpData[j][i] = zpData[j][nWm];
				else 
					zpData[j][i] = zpData[j][nWm+1];
			}
			for (i=nWm+nW;i<nZW;i++){
				if ((i-nWm-nW)%2==0)
					zpData[j][i] = zpData[j][nWm+nW-2];
				else 
					zpData[j][i] = zpData[j][nWm+nW-1];					
			}
		}

		for (i=0;i<nZW;i++){
			for (j=0;j<nHm;j++){
				if ((nHm-j)%2==0)
					zpData[j][i] = zpData[nHm][i];
				else 
					zpData[j][i] = zpData[nHm+1][i];					
			}
			for (j=nHm+nH;j<nZH;j++){
				if ((j-nHm-nH)%2==0)
					zpData[j][i] = zpData[nHm+nH-2][i];
				else 
					zpData[j][i] = zpData[nHm+nH-1][i];					
			}
		}
		return zpData;
	}

	/**
	 * get the original-size data form the padded data
	 * @param data the padded data
	 * @param nW the width
	 * @param nH the height 
	 * @return the original-size data
	 */
	private float[][] AlternatePaddingInverse(float[][] data,int nW,int nH)
	{
		int nZW = data[0].length;
		int nZH = data.length;


		int nWm = (int)(floor(nZW/2)-floor(nW/2));
		int nHm = (int)(floor(nZH/2)-floor(nH/2));

		int i=0;
		int j=0;

		float[][] zpIData = new float[nH][nW];
		for (j=0;j<nH;j++){
			for (i= 0;i<nW;i++){
				zpIData[j][i] = data[j+nHm][i+nWm];
			}
		}
		return zpIData;
	}

	/**
	 * Hamming Window "(54 - 46*cos(2*pi*(0:m-1)'/(n-1)))/100;"
	 * @param nWidth number of columns
	 * @param nHeight number of rows
	 * @param lowpassfilterW2 cutoff frequency in Width
	 * @param lowpassfilterH2 cutoff frequency in Height
	 * @return Hamming window
	 */
	private float[][] Hamming(int nWidth, int nHeight,double lowpassfilterW2,double lowpassfilterH2)
	{
		int filterLenW = (int) round(nWidth*lowpassfilterW2);
		int filterLenH = (int) round(nHeight*lowpassfilterH2);

		filterLenW = filterLenW+filterLenW%2;
		filterLenH = filterLenH+filterLenH%2;

		int mW = (int)(filterLenW/2);
		int mH = (int)(filterLenH/2);

		double [] filterW = new double [filterLenW];
		double [] filterH = new double [filterLenH];

		for (int i=0;i<mW;i++){
			filterW[mW-1-i] = 0.54 - 0.46*cos(2*Math.PI*i/(filterLenW-2));
		}
		for (int i=0;i<mH;i++){
			filterH[mH-1-i] = 0.54 - 0.46*cos(2*Math.PI*i/(filterLenH-2));
		}

		filterW[mW] = 0;
		filterH[mH] = 0;

		for(int i=mW+1;i<filterLenW;i++){
			filterW[i] = filterW[filterLenW-i];
		}
		for(int i=mH+1;i<filterLenH;i++){
			filterH[i] = filterH[filterLenH-i];
		}	
		filterW = FFTUtil.fftshift(filterW, false, true);
		filterH = FFTUtil.fftshift(filterH, false, true);	

		float [][] fHammW = new float[nHeight][nWidth];
		int cH = (int) (floor(nHeight/2.0) - floor(filterLenH/2.0) );
		int cW = (int) (floor(nWidth/2.0) - floor(filterLenW/2.0) );
		for (int j=0;j<filterLenH;j++){
			for(int i=0;i<filterLenW;i++){
				fHammW[j+cH][i+cW] = (float) (filterW[i]*filterH[j]);
			}
		}	
		return FFTUtil.fftshift(fHammW, false, false);

	}
	
	
    /**
     * Estimate scatter from the primary-modulated image
     * @param ip the primary-modulated image
     * @return scatter distribution
     */
	private ImageProcessor GetScatterDistributionSMI(ImageProcessor ip)
	{
		if (modulationFunction==null || alphaDist==null)
			return null;
		int nW = modulationFunction.getWidth();
		int nH = modulationFunction.getHeight();
		
		//smooth the primary-modulated image
		MeanFilteringTool mean = new MeanFilteringTool();
		mean.configure(meanfilterLen, meanfilterLen);		
		
		//downsample projection
		dsModFun = (FloatProcessor)downsampleProjection(
				ImageUtil.wrapGrid2D(
				mean.applyToolToImage(ImageUtil.wrapImageProcessor(modulationFunction.getProcessor())))
				);
		double dMean = GetMeanofDownSampledModFun(dsModFun);
		
		//refine modulation function
		dsModFun.add(-dMean*gaincutfactor);
		
		FloatProcessor dsIp = (FloatProcessor)downsampleProjection(
				ImageUtil.wrapGrid2D(mean.applyToolToImage(ImageUtil.wrapImageProcessor(ip))));
		
		int nDsW = dsModFun.getWidth();
		int nDsH = dsModFun.getHeight();
		double val = 0;
		
		for(int i=0;i<nDsH;i++){
			for (int j=0;j<nDsW;j++){
				val = dsIp.getPixelValue(j, i)/dsModFun.getPixelValue(j, i);
				dsIp.putPixelValue(j, i, val);
			}
		}
		
		int nZW = max(128,(int) Math.pow(2, nextPower2(nDsW)));
		int nZH = max(128,(int) Math.pow(2, nextPower2(nDsH)));
		//padding image for FFT
		float[][] dsAP = AlternatePadding(getFloatArray(dsIp),nZW,nZH);
		//create low-pass filter and the corresponding high-pass filter
		float [][] LowPassFilter = Hamming(nZW,nZH,lowPassFilterWidth,lowPassFilterHeight);
		float [][] HighPassFilter = FFTUtil.fftshift(LowPassFilter, false, true);
		float [][] LowPassImage = new float[nZH][nZW];
		float [][] HighPassImage = new float[nZH][nZW];

		//Get low and high frequency image
		FFTUtil.GetLowandHighPassImage(dsAP,LowPassFilter,HighPassFilter,LowPassImage,HighPassImage);

		//LowPassImage = AlternatePaddingInverse(LowPassImage,nDsW,nDsH);
		HighPassImage = AlternatePaddingInverse(HighPassImage,nDsW,nDsH);
		//FloatProcessor fPLowPassImage = (FloatProcessor) upsampleProjection(Array2FloatProcessor(LowPassImage));
		FloatProcessor fPHighPassImage = (FloatProcessor) upsampleProjection(Array2FloatProcessor(HighPassImage));
		
		// get scatter distribution
		double dAlpha=0;
		double dVal = 0;
		FloatProcessor fSc = new FloatProcessor(nW,nH);
		for (int i=0;i<nH;i++){
			for (int j=0;j<nW;j++){
				dAlpha = alphaDist.getPixelValue(j, i);				
				dVal = 4 * scatterEstScale * dAlpha/(1-dAlpha*dAlpha) * dMean * fPHighPassImage.getPixelValue(j, i);
				fSc.putPixelValue(j, i, dVal);
			}
		}		
		return fSc;
	}

	/**
	 * soft cut to avoid negative after scatter removal
	 * @param ip the primary modulated image
	 * @param sc the scatter corrected image
	 */
	private void TuneupScatterDistribution(ImageProcessor ip, ImageProcessor sc)
	{
		int nW = ip.getWidth();
		int nH = ip.getHeight();
		double val = 0;
		double dd = 0;
		double pp = 0;
		for(int i=0;i<nH;i++){
			for(int j=0;j<nW;j++){
				val = sc.getPixelValue(j, i);
				if(val<0){
					sc.putPixelValue(j, i,0);
				}
				pp = ip.getPixelValue(j, i);
				if(val>=softcutfactor*pp){
					dd = pp + 
					 (softcutfactor-1)*pp*exp(-(val-softcutfactor*pp)/((1-softcutfactor)*pp));
					sc.putPixelValue(j, i,dd);
				}
			}
		}
	}
	
	public ImageProcessor applyToolToProcessorOld(ImageProcessor imageProc) {
	
		
		ImageProcessor sc = GetScatterDistributionSMI(imageProc);
		
		//sc = TuneupScatterDistribution(imageProc,sc);
			
		new ImagePlus("Sc Image",sc).show();


		return dsModFun; //ZeroPadding(imageProc);//downsampleProjection(mean.applyToolToProcessor(imageProc)));
	}

	/**
	 * smooth scatter estimation over 2*context+1 views
	 */
	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		if (!initModulation) initModulation();
		// check all estimates are present
		System.out.println(lowerEnd(projectionNumber) + " " + upperEnd(projectionNumber));
		
		for (int i=lowerEnd(projectionNumber); i< upperEnd(projectionNumber); i++){
			if (scatterEstimatesBuffer == null){
				scatterEstimatesBuffer = new ImageGridBuffer(); 
			}
			if (scatterEstimatesBuffer.get(i) == null){
				scatterEstimatesBuffer.add(ImageUtil.wrapImageProcessor(GetScatterDistributionSMI((ImageProcessor)ImageUtil.wrapGrid2D(inputQueue.get(i)))),i);
			}
		}
		// compute mean over scatter estimates
		int count =  upperEnd(projectionNumber) - lowerEnd(projectionNumber);
		if (count==2*context+1){
			ImageProcessor buffer = ImageUtil.wrapGrid2D(scatterEstimatesBuffer.get(lowerEnd(projectionNumber)));
			for (int i=lowerEnd(projectionNumber)+1; i< upperEnd(projectionNumber); i++){
				ImageUtil.addProcessors(buffer, ImageUtil.wrapGrid2D(scatterEstimatesBuffer.get(i)));
			}
			buffer.multiply(1.0/count);
			TuneupScatterDistribution(ImageUtil.wrapGrid2D(inputQueue.get(projectionNumber)), buffer);
			sink.process(ImageUtil.wrapImageProcessor(buffer), projectionNumber);
		}
		else{
			TuneupScatterDistribution(ImageUtil.wrapGrid2D(inputQueue.get(projectionNumber)), ImageUtil.wrapGrid2D(scatterEstimatesBuffer.get(projectionNumber)));
			sink.process(scatterEstimatesBuffer.get(projectionNumber), projectionNumber);
		}
		
		for (int i =0; i< projectionNumber - (2*context); i++){
			scatterEstimatesBuffer.remove(i);
			inputQueue.remove(i);
		}
		

	}

	/**
	 * do Initialization
	 */
	private void initModulation(){
		ImagePlus modFun = new ImagePlus("Modulation Function", 
                getModulationFunction(modulationFunctionFilename).getProcessor().duplicate());

		modFun.show();
		InitModFunParameters();		
		
		DownSamplePointsMark(modFun.getProcessor());
		
		modFun.setTitle("Downsampling points in Modulation Function");
		modFun.updateAndDraw();

		GetAlphaDistribution(bUniformAlpha);
		initModulation = true;
	}

	/**
	 * @return the modulationFunctionFilename
	 */
	public String getModulationFunctionFilename() {
		return modulationFunctionFilename;
	}

	/**
	 * @param modulationFunctionFilename the modulationFunctionFilename to set
	 */
	public void setModulationFunctionFilename(String modulationFunctionFilename) {
		this.modulationFunctionFilename = modulationFunctionFilename;
	}

	/**
	 * @return the meanfilterLen
	 */
	public int getMeanfilterLen() {
		return meanfilterLen;
	}

	/**
	 * @param meanfilterLen the meanfilterLen to set
	 */
	public void setMeanfilterLen(int meanfilterLen) {
		this.meanfilterLen = meanfilterLen;
	}


	/**
	 * @return the gaincutfactor
	 */
	public double getGaincutfactor() {
		return gaincutfactor;
	}

	/**
	 * @param gaincutfactor the gaincutfactor to set
	 */
	public void setGaincutfactor(double gaincutfactor) {
		this.gaincutfactor = gaincutfactor;
	}

	/**
	 * @return the scatterEstScale
	 */
	public double getScatterEstScale() {
		return scatterEstScale;
	}

	/**
	 * @param scatterEstScale the scatterEstScale to set
	 */
	public void setScatterEstScale(double scatterEstScale) {
		this.scatterEstScale = scatterEstScale;
	}

	/**
	 * @return the bUniformAlpha
	 */
	public boolean isbUniformAlpha() {
		return bUniformAlpha;
	}

	/**
	 * @param bUniformAlpha the bUniformAlpha to set
	 */
	public void setbUniformAlpha(boolean bUniformAlpha) {
		this.bUniformAlpha = bUniformAlpha;
	}

	/**
	 * @return the softcutfactor
	 */
	public double getSoftcutfactor() {
		return softcutfactor;
	}

	/**
	 * @param softcutfactor the softcutfactor to set
	 */
	public void setSoftcutfactor(double softcutfactor) {
		this.softcutfactor = softcutfactor;
	}
	
	

}
/*
 * Copyright (C) 2010-2014 - Hewei Gao 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/