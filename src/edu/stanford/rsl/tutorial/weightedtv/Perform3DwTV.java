package edu.stanford.rsl.tutorial.weightedtv;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;

import loci.poi.util.SystemOutLogger;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;

public class Perform3DwTV {
	static int maxIter=500;
	static int maxTVIter=10;
	static int showInt=10;
	static int saveInt=5;
	static double eps=1.0e-10;
	static double error=1,error2=1;
	static int iter=0;
	static boolean isSave=false;
	static boolean debug=false;

	
	

	static private String path="D:\\wTV3D\\";
	static private File outPutDir=new File(path+"log.txt");
	static private BufferedWriter bw;
	
	//geometry
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
	static double originX;
	static double originY;
	static double originZ;
	//
	static protected CLContext context = null;
	
	//SART
	static float beta=0.8f;
	protected static ConeBeamProjector cbp;
	protected static ConeBeamBackprojector cbbp;
	private static OpenCLGrid3D volCL;//Object
	private static OpenCLGrid3D reconCL;
	private static OpenCLGrid3D reconCLpre;//Current reconCL before SART and wTV iteration
	private static OpenCLGrid3D sinoCL;//measured sinogram
	private static OpenCLGrid3D updBP; 
	private static OpenCLGrid2D sinoP;
	protected static OpenCLGrid3D normSino;
	protected static OpenCLGrid3D normGrid;
	
	//WTV
	protected static TVGradient3D TVgrad;
	protected static double step=0;
	
	
	public static void main(String[] args) throws Exception {
		new ImageJ();
			
		
		
		if(!outPutDir.exists()){
		    outPutDir.getParentFile().mkdirs();
			outPutDir.createNewFile();
		}
		
		bw=new BufferedWriter(new FileWriter(outPutDir));
		
		initCLDatastructure();

		Grid3D phan = new NumericalSheppLogan3D(imgSizeX,imgSizeY, imgSizeZ).getNumericalSheppLoganPhantom();
		phan.setSpacing(spacingX,spacingY,spacingZ);
		volCL=new OpenCLGrid3D(phan);
		
		//****************************prepare for SART ******************************************************	
		cbp=new ConeBeamProjector();
		cbbp=new ConeBeamBackprojector();
		sinoP=new OpenCLGrid2D(new Grid2D(width,height));
		 updBP = new OpenCLGrid3D(volCL);
		
		getMeasuredSinoCL();		
		initialReconCL();
		createNormProj();		
		createNormGrid();  
	
		
		
		
		//***************wTV parameters*****************************
			
		
		TVgrad=new TVGradient3D(reconCL);
		
		TVgrad.initialWmatrixCL();
		
		
		//**********************************************************
	
		long t_start=System.currentTimeMillis();
		TVgrad.computeGradientCL(volCL);
		float pTV=volCL.getGridOperator().normL1(TVgrad.imgGradientCL);
		System.out.println("Perfect TV: "+pTV);
		
		//**********************************************************
		//Start iteration
		while(iter<=maxIter&&error>eps){
		   reconCLpre=new OpenCLGrid3D(reconCL);
		
					
			sartiterate();//SART iteration

			
			//reconCL.show(" reconCL");
			
			//wTV part
			wTViterate();
			
			TVgrad.wMatrixCLUpdate(reconCL);

			outPutError();	
			iter++;
		}
		
		long t_end=System.currentTimeMillis();
		System.out.println("time is "+(t_end-t_start)/1000.0);
		reconCL.show("final reconCL");
		ImagePlus imp3=ImageUtil.wrapGrid3D(reconCL, null);
		IJ.saveAs(imp3, "Tiff", (path+iter+"_"+"FinalReconCL.tif"));
		bw.close();
		
}
	private static void sartiterate() throws Exception{
	   OpenCLGrid2D tempSino;//=new OpenCLGrid2D(sinoP);
	   //normGrid=new OpenCLGrid3D(volCL);//***********************
		for(int projIndex=0;projIndex<maxProjs;projIndex++){
			
			cbp.fastProjectRayDrivenCL(sinoP, reconCL, projIndex);
			
			tempSino=new OpenCLGrid2D(sinoCL.getSubGrid(projIndex));		      
			sinoP.getGridOperator().subtractBy(sinoP,tempSino);

			tempSino.release();
			tempSino=new OpenCLGrid2D(normSino.getSubGrid(projIndex));
           
			sinoP.getGridOperator().divideBy(sinoP,tempSino);
			tempSino.release();			
            updBP.getGridOperator().fill(updBP, 0);  
           
            cbbp.fastBackprojectPixelDrivenCL(sinoP,updBP, projIndex);
         
            updBP.getGridOperator().multiplyBy(updBP, -beta);
            // createNormGrid2(projIndex);  //****************************
            updBP.getGridOperator().divideBy(updBP, normGrid);
            reconCL.getGridOperator().addBy(reconCL, updBP);

		}
		reconCL.getGridOperator().removeNegative(reconCL);
	}
	
	
	
	private static void wTViterate() throws IOException{
		OpenCLGrid3D tv_gradient;
		double wTV=TVgrad.getwTVvalueCL(reconCL);
		int i=0;
		double deltaTV=-1.0;
		while(i<maxTVIter){
			double preTV=wTV;
				tv_gradient=TVgrad.compute_wTV_Gradient(reconCL);

			tv_gradient.getGridOperator().divideBy(tv_gradient, tv_gradient.getGridOperator().max(tv_gradient));				
	
			wBTLsearch(tv_gradient);
			
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

	
	private static void wBTLsearch( OpenCLGrid3D tv_grad){//weighted TV
		double t=1.0,tmin=0.0000001;
		double alpha=0.3, beta=0.6;
		double delta=1.0f,temp1,temp2;
		
		double TV=TVgrad.getwTVvalueCL(reconCL);
		double Gradnorm=alpha*grid3DCLnorm(tv_grad);
	
		
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
	
		
	private static double grid3DCLnorm(OpenCLGrid3D reconCL) {
		OpenCLGrid3D tempReconCL=new OpenCLGrid3D(reconCL);
		tempReconCL.getGridOperator().multiplyBy(tempReconCL,tempReconCL);
		double n=reconCL.getGridOperator().sum(tempReconCL);
		tempReconCL.release();
		return n;
	}
	
	
	private static double mse(OpenCLGrid3D gridCL1, OpenCLGrid3D gridCL2){
		OpenCLGrid3D tempCL=new OpenCLGrid3D(gridCL1);
		tempCL.getGridOperator().subtractBy(tempCL, gridCL2);
		double err=grid3DCLnorm(tempCL)/(gridCL1.getSize()[0]*gridCL1.getSize()[1]*gridCL1.getSize()[2]);
		tempCL.release();
		return err;
	}
	
	private static void initCLDatastructure(){
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		geo = conf.getGeometry();
		width =  geo.getDetectorWidth();
		height =  geo.getDetectorHeight();
		maxProjs = geo.getProjectionStackSize();
		// create context
		context = OpenCLUtil.getStaticContext();
		imgSizeX = geo.getReconDimensionX();
		imgSizeY = geo.getReconDimensionY();
		imgSizeZ = geo.getReconDimensionZ();
		spacingX = geo.getVoxelSpacingX();
		spacingY = geo.getVoxelSpacingY();
		spacingZ = geo.getVoxelSpacingZ();
		originX = -geo.getOriginX();
		originY = -geo.getOriginY();
		originZ = -geo.getOriginZ();

	}
	
	protected static void getMeasuredSinoCL() throws Exception
	{
		sinoCL = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		sinoCL.getDelegate().prepareForDeviceOperation();		
		cbp.fastProjectRayDrivenCL(sinoCL,volCL);	
		sinoCL.show("sinoCL");
	}
	
	protected static void initialReconCL()
	{

		reconCL=new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		reconCL.setSpacing(spacingX,spacingY,spacingZ);
		reconCL.getGridOperator().fill(reconCL, 0);
	}
		
	
	
	protected static void createNormProj() throws Exception{
		OpenCLGrid3D onesVol = new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));		
		onesVol.getGridOperator().fill(onesVol, 1);
	normSino = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));		
	cbp.fastProjectRayDrivenCL(normSino,onesVol);			
		
	normSino.getGridOperator().addBy(normSino,(float) eps);
	//if(debug)normSino.show("normSino");
	}

	protected static void createNormGrid2(int projIndex) throws Exception {
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid.release();
		normGrid=new OpenCLGrid3D(volCL);
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, projIndex);
		//normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		normGrid.getGridOperator().addBy(normGrid, (float)eps);
	}
	
	
	protected static void createNormGrid() throws Exception {
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid=new OpenCLGrid3D(volCL);
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, 0);
		normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		//normGrid.getGridOperator().addBy(normGrid, (float)eps);
	}
	private static void outPutMediateResult( int iter, int i)
	{
			ImagePlus imprecon=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imprecon, "Tiff",(path+iter+"_"+i+"thImRecon.tif"));
			TVgrad.computeGradientCL(reconCL);
			ImagePlus imGrad=ImageUtil.wrapGrid3D(TVgrad.imgGradientCL, null);
			IJ.saveAs(imGrad, "Tiff",(path+iter+"_"+i+"thImGrad.tif"));	
			ImagePlus impTVgrad=ImageUtil.wrapGrid3D(TVgrad.tvGradientCL, null);
			IJ.saveAs(impTVgrad, "Tiff", (path+iter+"_"+i+"thTVgradient.tif"));		
	}
	
	private static void outPutError() throws IOException{
		
		if (iter % showInt== 0)
		{
			reconCL.show(iter+"_th iteration");				
		}
		if(isSave)
		if((iter<=100&&iter%saveInt==0)||(iter>100&&iter%50==0))
		{
			ImagePlus imp2=ImageUtil.wrapGrid3D(reconCL, null);
			IJ.saveAs(imp2, "Tiff", (path+iter+"thResult.tif"));	
		}
		error =mse(reconCL, reconCLpre);	
		error2=mse(reconCL, volCL);
		System.out.println(iter+":  error=" + error+" error2= "+error2);	
		bw.write(iter+":  error=" + error+" error2= "+error2+"\r\n");
		bw.flush();
		reconCLpre.release();
	}
}

