package edu.stanford.rsl.tutorial.WeightedTV;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;

import loci.poi.util.SystemOutLogger;
import ij.IJ;
import ij.io.FileInfo;
import ij.io.FileOpener;
import edu.stanford.rsl.apps.gui.RawDataOpener;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;

public class Perform3DwTV_clinical {
	static int maxIter=30;
	static int TVmaxIter=10;
	static int iter=0;
	static double eps=1.0e-10;
	static int samplingFactor=2;
	static double error=1;
	private static float reconAngle=160.0f;
	private static float radius=288.0f;
	private static int showInt=20;
	private static  int saveInt=1;
	static boolean isInitial=true;
	static int IterIni=11;
	static boolean isFDKinitial=false;
	static boolean debug=false;
	static boolean debugMem=false;
	static boolean isSave=true;
	static boolean isSartSave=false;
	static boolean isResample=false;
	static boolean isResetWeps=false;
	static boolean needSinogram=true;
	static boolean isAdaptiveW=false;
	static boolean isWTV=true;
	
	//static private float amp=1.0f;
	static private String path="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\0RealData\\HighRes\\3SART\\";
    //static private String initialPath="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\0RealData\\AwTV\\5\\80thResult.tif";
	static private String initialPath="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\0RealData\\HighRes\\3SART\\10thResult.tif";
	static String sinoSavePath="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\Data\\MagdeburgBrain\\sinogram\\";
    static private BufferedWriter bw;
	static  CLCommandQueue testqueue;

	 //geometry
	static Grid3D ProjGrid3D;//The Grid3D storing the projection matrix
	static Projection[] projMats;
	static protected Trajectory geo = null;
	static protected int maxProjs;
	static protected int width;
	static protected int height;
	static int imgSizeX;
	static int imgSizeY;
	static int imgSizeZ;
	static double spacingX;
	static double spacingY;
	static double spacingZ;
	//static double originX;
	//static double originY;
	//static double originZ;
	//
	static protected CLContext context = null;
	
	//SART
	static float beta=0.8f;
	protected static ConeBeamProjector cbp;
	protected static ConeBeamBackprojector cbbp;
//	private static OpenCLGrid3D volCL=null;//Object
	private static OpenCLGrid3D reconCL;

	private static OpenCLGrid3D reconCLpre;//Current reconCL before SART and wTV iteration
	//private static OpenCLGrid3D sinoCL=null;//measured sinogram
	private static OpenCLGrid3D updBP; 
	private static OpenCLGrid2D sinoP;
	//private static OpenCLGrid2D normSino;
	private static float normSinomean;
	//private static OpenCLGrid3D onesVol;
	private static float normGridmean;
	private static Grid3D sinogram;
	//WTV
	private static TVgradient3D TVgrad;
	static TVOpenCLGridOperators TVOperators;
	private static double step=0;
	
	
	public static void main(String[] args) throws Exception {
		new ImageJ();
	
		
		testqueue = OpenCLUtil.getStaticContext().getMaxFlopsDevice().createCommandQueue();
	     File outName=new File(path+"log.txt");
		outName.createNewFile();
		bw=new BufferedWriter(new FileWriter(outName));
		TVOperators=new TVOpenCLGridOperators();
		
		initCLDatastructure();
		if(needSinogram)
		getMeasuredSinoCL();
	
		//****************************prepare for SART ******************************************************	
		cbp=new ConeBeamProjector();
		cbbp=new ConeBeamBackprojector();
		sinoP=new OpenCLGrid2D(new Grid2D(width,height));
		//normSino=new OpenCLGrid2D(new Grid2D(width,height));
		 updBP = new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		 
			
	
		
		initialReconCL();		
		createNormGrid();
		createNormProj(); 
	
		TVgrad=new TVgradient3D(reconCL);
		//TVgrad.imgGradientCL.setSpacing(spacingX,spacingY,spacingZ);		
		TVgrad.initialWmatrixCL();
		if(isResetWeps)
			TVgrad.weps=0.0001f;
		if(isInitial)
		{if(isWTV)
			TVgrad.WmatrixCLUpdate(reconCL);
		}
		//ImagePlus imp4=ImageUtil.wrapGrid3D(TVgrad.TVgradientCL, null);
		//IJ.saveAs(imp4, "Tiff", (path+"imgGradientCL.tif"));
		
		long t_start=System.currentTimeMillis();
		
		//bw.write("Perfect TV:"+pTV+"\r\n");
		//bw.flush();
		//**********************************************************
		//Start iteration
		
		
		while(iter<=maxIter&&error>eps){
		  // reconCLpre=new OpenCLGrid3D(reconCL);
		

					
			SartCLiterate();//SART iteration
			
			//if(iter==1)
			if(isSartSave)
			{
			ImagePlus imp1=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imp1, "Tiff", (path+iter+"_"+"thSARTafterResult.tif"));
			}
			TVOperators.FOVmask(reconCL, radius);
			reconCL.getGridOperator().removeNegative(reconCL);
			//reconCL.show(" reconCL");
			
			if(iter==0)
				reconCL.show(" 0th SART");
			//wTV part
			WTViterate();
			if(isWTV)
				TVgrad.WmatrixCLUpdate(reconCL);
				
			OutPutError();
				
			iter++;
		}
		
		long t_end=System.currentTimeMillis();
		System.out.println("time is "+(t_end-t_start)/1000.0);
		reconCL.show("final reconCL");
		
		ImagePlus imp3=ImageUtil.wrapGrid3D(reconCL, null);
		IJ.saveAs(imp3, "Tiff", (path+iter+"_"+"FinalReconCL.tif"));
		bw.close();
		
}
	private static void SartCLiterate() throws Exception{
	   OpenCLGrid2D tempSino;//=new OpenCLGrid2D(sinoP);
	   //normGrid=new OpenCLGrid3D(volCL);//***********************
	   //for(int projIndex=(int)(10*maxProjs/200.0);projIndex<maxProjs*(reconAngle+10)/200.0;projIndex++){
		//for(int projIndex=0;projIndex<maxProjs*reconAngle/200.0;projIndex++){
	   for(int projIndex=0;projIndex<maxProjs;projIndex++){
			if(debugMem)
				System.out.println(testqueue.getContext().getMemoryObjects().size()+"Here 10");
			cbp.fastProjectRayDrivenCL(sinoP, reconCL, projIndex);
		
			if(debugMem)
				System.out.println(testqueue.getContext().getMemoryObjects().size()+"Here 11");
			tempSino=new OpenCLGrid2D(sinogram.getSubGrid(projIndex));		      
			sinoP.getGridOperator().subtractBy(sinoP,tempSino);			
			tempSino.release();
			/*
			ImagePlus impsino=IJ.openImage(sinoSavePath+projIndex+"thProjection.tif");
			Grid2D tempSinoP=ImageUtil.wrapImagePlus(impsino).getSubGrid(0);
		
			tempSino=new OpenCLGrid2D(tempSinoP);		      
			sinoP.getGridOperator().subtractBy(sinoP,tempSino);
			
			tempSino.release();
			*/
			sinoP.getGridOperator().divideBySave(sinoP,normSinomean);
						
            updBP.getGridOperator().fill(updBP, 0);  
            if(debugMem)
				System.out.println(testqueue.getContext().getMemoryObjects().size()+"Here 12");
            cbbp.fastBackprojectPixelDrivenCL(sinoP,updBP, projIndex);
        
            if(debugMem)
				System.out.println(testqueue.getContext().getMemoryObjects().size()+"Here 13");
            updBP.getGridOperator().multiplyBy(updBP, -beta);
            // createNormGrid2(projIndex);  //****************************
            updBP.getGridOperator().divideBy(updBP, normGridmean);
       
            reconCL.getGridOperator().addBy(reconCL, updBP);
        System.out.print(" "+projIndex);
		}	
	   System.out.println(" ");
	}
	
	private static void WTViterate() throws IOException{
		OpenCLGrid3D tv_gradient;
		double wTV=TVgrad.getwTVvalueCL(reconCL);
		int i=0;
		double deltaTV=-1.0;
		while(i<TVmaxIter){
			double preTV=wTV;
				tv_gradient=TVgrad.compute_wTV_Gradient(reconCL);
			
			
			//if(debug)
				//if(i==0)
				//OutPutMediateResult( iter, i);
			tv_gradient.getGridOperator().divideBy(tv_gradient, tv_gradient.getGridOperator().max(tv_gradient));				
	
			wBTLsearch3(tv_gradient);
			
			wTV=TVgrad.getwTVvalueCL(reconCL);
			deltaTV=wTV-preTV;
			//System.out.println("iter="+iter+" i="+i+" L1: "+reconCL.getGridOperator().normL1(TVgrad.imgGradientCL)+" wTV="+wTV+" step="+step);
			System.out.println("iter="+iter+" i="+i+" wTV="+wTV+" step="+step);
			if(debug)
				//if(isSave)
			{
			ImagePlus imp3=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imp3, "Tiff", (path+iter+"_"+i+"thResult.tif"));
			}
			//bw.write("iter="+iter+" i="+i+" L1: "+reconCL.getGridOperator().normL1(TVgrad.imgGradientCL)+" wTV="+wTV+" step="+step+"\r\n");
			bw.write("iter="+iter+" i="+i+" wTV="+wTV+" step="+step+"\r\n");
			bw.flush();
			i++;
		}
		
	}

		
	private static void wBTLsearch3( OpenCLGrid3D tv_grad){//weighted TV
		double t=1.0,tmin=0.0000001;
		double alpha=0.3, beta=0.6;
		double delta=1.0f,temp1,temp2;
		
		double TV=TVgrad.getwTVvalueCL(reconCL);
		double Gradnorm=alpha*Grid3DCLnorm(tv_grad);
	
		
		tv_grad.getGridOperator().multiplyBy(tv_grad,(float) t);
		
		reconCL.getGridOperator().subtractBy(reconCL,tv_grad);

		temp1=TVgrad.getwTVvalueCL(reconCL);
		
		
		temp2=t*Gradnorm;
		delta=temp1-TV+temp2;
		
		while(delta>0.0f && t>tmin)
		{
			
			t=t*beta;
			reconCL.getGridOperator().addBy(reconCL, tv_grad);
			
			tv_grad.getGridOperator().multiplyBy(tv_grad,(float) beta);
					
			reconCL.getGridOperator().subtractBy(reconCL,tv_grad);

			temp1=TVgrad.getwTVvalueCL(reconCL); 
		
			temp2=t*Gradnorm;
			delta=temp1-TV+temp2;
		}
		//System.out.println("t="+t);
		step=t;
	}
	
	private static double Grid3DCLnorm(OpenCLGrid3D reconCL) {
		OpenCLGrid3D tempReconCL=new OpenCLGrid3D(reconCL);
		tempReconCL.
		getGridOperator().
		multiplyBy(tempReconCL,tempReconCL);
		double n=tempReconCL.getGridOperator().sum(tempReconCL);
		tempReconCL.release();
		return n;
	}
	/*
	private static double Grid3DCLnorm(OpenCLGrid3D reconCL) {
		reconCL.getGridOperator().multiplyBy(reconCL,reconCL);
		double n=reconCL.getGridOperator().sum(reconCL);
		reconCL.getGridOperator().sqrt(reconCL);
		return n;
	}
	*/
	
	private static double MSE(OpenCLGrid3D gridCL1, OpenCLGrid3D gridCL2){
		OpenCLGrid3D tempCL=new OpenCLGrid3D(gridCL1);
		tempCL.getGridOperator().subtractBy(tempCL, gridCL2);
		double err=Grid3DCLnorm(tempCL)/(gridCL1.getSize()[0]*gridCL1.getSize()[1]*gridCL1.getSize()[2]);
		tempCL.release();
		return err;
	}
	
	private static void initCLDatastructure(){
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		context = OpenCLUtil.getStaticContext();
		geo = conf.getGeometry();
		getProjectionMatrix();
		if(isResample)
			resetGeometry();
		
			width =  geo.getDetectorWidth();
			height =  geo.getDetectorHeight();
			maxProjs = geo.getProjectionStackSize();
			// create context		
			imgSizeX = geo.getReconDimensionX();
			imgSizeY = geo.getReconDimensionY();
			imgSizeZ = geo.getReconDimensionZ();
			spacingX = geo.getVoxelSpacingX();
			spacingY = geo.getVoxelSpacingY();
			spacingZ = geo.getVoxelSpacingZ();
			//originX = -geo.getOriginX();
			//originY = -geo.getOriginY();
			//originZ = -geo.getOriginZ();
			/*
			onesVol = new OpenCLGrid3D(new Grid3D(imgSizeX+50,imgSizeY+50,imgSizeZ+50));
			onesVol.setSpacing(spacingX,spacingY,spacingZ);
			onesVol.getGridOperator().fill(onesVol, 1);
		*/
			radius=(float)(radius*0.4/spacingX);
			
	}
	
	protected static void getMeasuredSinoCL() throws Exception
	{/*switch (tag){
		case 0:	//SheppLogan Phantom	
		Grid3D phan = new NumericalSheppLogan3D(imgSizeX,imgSizeY, imgSizeZ).getNumericalSheppLoganPhantom();
		phan.setSpacing(spacingX,spacingY,spacingZ);
		volCL=new OpenCLGrid3D(phan);
		ImagePlus phanGT=ImageUtil.wrapGrid3D(volCL, null);
		IJ.saveAs(phanGT, "Tiff", (path+"phanGT.tif"));
		
		sinoCL = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		sinoCL.getDelegate().prepareForDeviceOperation();		
		cbp.fastProjectRayDrivenCL(sinoCL,volCL);	
		sinoCL.show("sinoCL");
	case 1://real data
	*/
		
		getClinicalSinograms();

	
	}
	
	
	private static void getProjectionMatrix(){
		 String Rawpath="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\Data\\MagdeburgBrain";
		 String Rawname="ProjMatrix.bin";
		
		FileInfo fi=new FileInfo();
		fi.width=3;
		fi.height=4;
		fi.nImages=496;
		fi.offset=6;
		fi.intelByteOrder=true;
		fi.fileFormat=FileInfo.RAW;
		fi.fileType=FileInfo.GRAY64_FLOAT;
		fi.directory=Rawpath;
		fi.fileName=Rawname;
		FileOpener fopen=new FileOpener(fi);
		ImagePlus ProjMatrixImp=fopen.open(true);
		ProjGrid3D=ImageUtil.wrapImagePlus(ProjMatrixImp);
		projMats=new Projection[ProjGrid3D.getSize()[2]];
        
		SimpleMatrix tempMat=new SimpleMatrix(ProjGrid3D.getSize()[0],ProjGrid3D.getSize()[1]);//**********FIXME row and col
		for(int i=0;i<ProjGrid3D.getSize()[2];i++){
			for(int k=0;k<ProjGrid3D.getSize()[0];k++)
				for(int m=0;m<ProjGrid3D.getSize()[1];m++)
					tempMat.setElementValue(k, m, ProjGrid3D.getSubGrid(i).getAtIndex(k, m));			
			        projMats[i]=new Projection();
			        projMats[i].initFromP(tempMat);
				    //projMats[i].initFromP(tempMat.transposed());
		}
		geo.setProjectionMatrices(projMats);
		geo.setNumProjectionMatrices(ProjGrid3D.getSize()[2]);
		
		
	}
	
	private static void getClinicalSinograms(){
		 String sinopath="D:\\tasks\\FAU1\\Research_Limited angle reconstruction\\TVresults\\Data\\MagdeburgBrain";
		 String sinoname="projections.bin"; 
		FileInfo fi=new FileInfo();
		fi.width=1240;
		fi.height=960;
		fi.nImages=496;
		fi.offset=6;
		fi.intelByteOrder=true;
		fi.fileFormat=FileInfo.RAW;
		fi.fileType=FileInfo.GRAY32_FLOAT;
		fi.directory=sinopath;
		fi.fileName=sinoname;
		FileOpener fopen=new FileOpener(fi);
		ImagePlus ProjMatrixImp=fopen.open(false);
		Grid3D sino=ImageUtil.wrapImagePlus(ProjMatrixImp);
		
		if(isResample){
			sinogram=new Grid3D(sino.getSize()[0]/2,sino.getSize()[1]/2,sino.getSize()[2]/2);
		for(int i=0;i<sinogram.getSize()[2];i++)
			sinogram.setSubGrid(i, DownSampling(sino.getSubGrid(2*i)));
		}
		else
			sinogram=(Grid3D)sino.clone();
		//sinoCL=new OpenCLGrid3D(sinogram);
	}
	private static void resetGeometry()
	{
		geo.setDetectorHeight(geo.getDetectorHeight()/samplingFactor);
		geo.setDetectorWidth(geo.getDetectorWidth()/samplingFactor);
		geo.setNumProjectionMatrices(geo.getNumProjectionMatrices()/samplingFactor);
		geo.setPixelDimensionX(geo.getPixelDimensionX()*samplingFactor);
		geo.setPixelDimensionY(geo.getPixelDimensionY()*samplingFactor);
		geo.setProjectionStackSize(geo.getProjectionStackSize()/samplingFactor);
		projMats=new Projection[geo.getNumProjectionMatrices()];
		for(int i=0;i<geo.getNumProjectionMatrices();i++){
			projMats[i]=new Projection();
			projMats[i]=geo.getProjectionMatrices()[2*i];
			//System.out.println(tempProjMats[i].computeP().toString());
		}
		geo.setProjectionMatrices(projMats);
		
		
	}
	
	protected static void initialReconCL()
	{
		if(isInitial){
			ImagePlus imp=IJ.openImage(initialPath);
			
			  reconCL=new OpenCLGrid3D(ImageUtil.wrapImagePlus(imp));
			  reconCL.show("initial reconCL");
			  iter=IterIni;
		}
		else{
		reconCL=new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		reconCL.setSpacing(spacingX,spacingY,spacingZ);
		reconCL.getGridOperator().fill(reconCL, 0);
		if(isFDKinitial){
			double focalLength = geo.getSourceToDetectorDistance();
			double deltaU = geo.getPixelDimensionX();
			double deltaV = geo.getPixelDimensionY();
			Grid3D sinogram2=new Grid3D(sinogram);
			ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength, width, height, deltaU, deltaV);
			RamLakKernel ramK = new RamLakKernel(width, deltaU);
			for (int i = 0; i < geo.getNumProjectionMatrices(); ++i) 
				
			{
				cbFilter.applyToGrid(sinogram2.getSubGrid(i));
				//ramp
				for (int j = 0;j <height; ++j)
					ramK.applyToGrid(sinogram2.getSubGrid(i).getSubGrid(j));
			
			}
			
			for(int i=0;i<sinogram2.getSize()[2]*reconAngle/200.0;i++)
			{sinoP.release();
			sinoP=new OpenCLGrid2D(sinogram2.getSubGrid(i)) ;
			updBP.getGridOperator().fill(updBP, 0);
			cbbp.fastBackprojectPixelDrivenCL(sinoP,updBP, i);
			 updBP.getGridOperator().multiplyBy(updBP, beta);
			 reconCL.getGridOperator().addBy(reconCL, updBP);
			 //System.out.println(testqueue.getContext().getMemoryObjects().size()+"Here 12");
			}
			TVOperators.FOVmask(reconCL, radius);
			reconCL.show("FDKinitial");
		}
		}
	}
		
	
	
	protected static void createNormProj() throws Exception{
		/*OpenCLGrid3D onesVol = new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));		
		onesVol.getGridOperator().fill(onesVol, 1);
	OpenCLGrid2D normSino = new OpenCLGrid2D(new Grid2D(width,height));		
	cbp.fastProjectRayDrivenCL(normSino,onesVol,0);			
	onesVol.release();
	normSinomean=(float)normSino.getGridOperator().sum(normSino)/(normSino.getNumberOfElements());
	normSino.release();
	System.out.println("normSinomean="+normSinomean);*/
	normSinomean=176.f;
	}
	

/*
	protected static void createNormGrid2(int projIndex) throws Exception {
		
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid.release();
		normGrid=new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, projIndex);
		//normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		normGrid.getGridOperator().addBy(normGrid, (float)eps);
	}
	*/
	
	protected static void createNormGrid() throws Exception {
	    /* OpenCLGrid3D normGrid;
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid=new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, 0);
		normGridmean=(float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements());
		//normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		//normGrid.getGridOperator().addBy(normGrid, (float)eps);
		normGrid.release();
		c_sinoCL.release();
		System.out.println("normGridmean="+normGridmean);*/
		normGridmean=1.01897f;
	}
	private static void OutPutMediateResult( int iter, int i)
	{
			ImagePlus imprecon=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imprecon, "Tiff",(path+"_"+iter+"_"+i+"thImRecon.tif"));
			TVgrad.ComputeGradientCL(reconCL);
			ImagePlus imGrad=ImageUtil.wrapGrid3D(TVgrad.WmatrixCL, null);
			IJ.saveAs(imGrad, "Tiff",(path+"_"+iter+"_"+i+"thWmatrixCL.tif"));	
			ImagePlus impTVgrad=ImageUtil.wrapGrid3D(TVgrad.TVgradientCL, null);
			IJ.saveAs(impTVgrad, "Tiff", (path+"_"+iter+"_"+i+"thTVgradient.tif"));		
	}
	
	private static void OutPutError() throws IOException, InterruptedException{
		 
		if (iter % showInt== 0)
		{
			reconCL.show(iter+"_th iteration");				
		}
		if(isSave)
		if((iter<=10&&iter%saveInt==0)||(iter>10&&iter%5==0))
		//if(iter%saveInt==0)
		{
			ImagePlus imp2=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imp2, "Tiff", (path+iter+"thResult.tif"));
					}
		System.out.println("iter="+iter);
		//error =MSE(reconCL, reconCLpre);	
		//error2=MSE(reconCL, volCL);
		//System.out.println(iter+":  error=" + error);	
		//bw.write(iter+":  error=" + error+"\r\n");
		bw.flush();
		//reconCLpre.release();
	}
	
	private static final Object lock = new Object();

	public static void waitHere() throws InterruptedException{
	   synchronized(lock){
	      lock.wait();
	      lock.notifyAll();
	   } 
	}
	
	private static Grid2D DownSampling(Grid2D img){
		int x=img.getSize()[0];
		int y=img.getSize()[1];
		float val;
		Grid2D DownImg=new Grid2D(x/2,y/2);
		for(int i=0;i<x/2;i++){
			for(int j=0;j<y/2;j++){
				val=(img.getAtIndex(2*i, 2*j)+img.getAtIndex(2*i+1, 2*j)+img.getAtIndex(2*i, 2*j+1)+img.getAtIndex(2*i+1, 2*j+1))/4;
			DownImg.setAtIndex(i, j, val);	
			}
	}
	return DownImg;
	}
	
}

