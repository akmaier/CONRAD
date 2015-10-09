package edu.stanford.rsl.tutorial.weightedtv;
/**
 * Here is an example to apply weighted total variation on 3D cone-beam limited angle tomography
 * @author Yixing Huang
 */

import java.io.IOException;
import com.jogamp.opencl.CLContext;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;

public class Perform3DWeightedTV {
	private int maxIter=500;
	public int maxTVIter=10;
	private int showInt=5;
	private double eps=1.0e-10;
	public double error=1,error2=1;
	private int iter=0;

	//geometry
	public int imgSizeX;
	public int imgSizeY;
	public int imgSizeZ;
	 protected Trajectory geo = null;
	 protected int maxProjs;
     protected int width;
	 protected int height;	
	 protected double spacingX;
	 protected double spacingY;
	 protected double spacingZ;
	 protected double originX;
	 protected double originY;
	 protected double originZ;
	//
	 protected CLContext context = null;
	
	//SART 
	private float beta=0.8f;
	
	public OpenCLGrid3D volCL;//Object
	public OpenCLGrid3D reconCL;
	private OpenCLGrid3D reconCLpre;//Current reconCL before SART and wTV iteration
	private OpenCLGrid3D sinoCL;//measured sinogram
	private OpenCLGrid3D updBP; 
	private OpenCLGrid2D sinoP;
	protected OpenCLGrid3D normSino;
	protected OpenCLGrid3D normGrid;
	
	//weighted TV
	public TVGradient3D TVgrad;
	protected double step=0;
	private  ConeBeamProjector cbp;
	private  ConeBeamBackprojector cbbp;
	
	public static void main(String[] args) throws Exception {
		new ImageJ();
			
		Perform3DWeightedTV tv3D=new Perform3DWeightedTV();
		
		
		tv3D.initOpenCLDataStructure();
		
		/**
		 * The ground truth phantom
		 */
		Grid3D phan = new NumericalSheppLogan3D(tv3D.imgSizeX,tv3D.imgSizeY, tv3D.imgSizeZ).getNumericalSheppLoganPhantom();
		//phan.setSpacing(spacingX,spacingY,spacingZ);
		tv3D.volCL=new OpenCLGrid3D(phan);
		
	
		tv3D.getMeasuredSinoCL();		
		tv3D.initialReconCL();
		tv3D.createNormProj();		
		tv3D.createNormGrid();  
			
		tv3D.TVgrad=new TVGradient3D(tv3D.reconCL);
		
		
		/**
		 * output the TV value of the ground truth phantom
		 */
		long t_start=System.currentTimeMillis();
		tv3D.TVgrad.computeImageGradientCL(tv3D.volCL);
		float pTV=tv3D.volCL.getGridOperator().normL1(tv3D.TVgrad.imgGradientCL);
		System.out.println("Perfect TV: "+pTV);
	
		
		while(tv3D.iter<=tv3D.maxIter&&tv3D.error>tv3D.eps){		   				
			tv3D.sartIterate();//SART iteration			
			tv3D.weightedTVIterate();//wTV iteration
			tv3D.outPutError();	
			tv3D.iter++;
		}
		
		long t_end=System.currentTimeMillis();
		System.out.println("time is "+(t_end-t_start)/1000.0);
		tv3D.reconCL.show("final reconCL");		
}
	
	/**
	 * SART part
	 * @throws Exception
	 */
	public void sartIterate() throws Exception{
	   OpenCLGrid2D tempSino;
	   reconCLpre=new OpenCLGrid3D(reconCL);
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
	
	
	/**
	 * weighted TV gradient descent part
	 * @throws IOException
	 */
	public void weightedTVIterate() throws IOException{
		OpenCLGrid3D tv_gradient;
		double wTV=TVgrad.getWeightedTVvalueCL(reconCL);
		int i=0;
		double deltaTV=-1.0;
		while(i<maxTVIter){
			double preTV=wTV;
				tv_gradient=TVgrad.computeWeightedTVGradient(reconCL);

			tv_gradient.getGridOperator().divideBy(tv_gradient, tv_gradient.getGridOperator().max(tv_gradient));				
	
			backTrackingLineSearch(tv_gradient);
			
			wTV=TVgrad.getWeightedTVvalueCL(reconCL);
			deltaTV=wTV-preTV;
			//System.out.println("iter="+iter+" i="+i+" L1: "+reconCL.getGridOperator().normL1(TVgrad.imgGradientCL)+" wTV="+wTV+" step="+step);
			System.out.println("iter="+iter+" i="+i+" wTV="+wTV+" step="+step);
			i++;
		}
		TVgrad.weightMatrixCLUpdate(reconCL);
	}

	/**
	 * using back tracking line search to find the step size for weighted TV gradient descent
	 * @param tv_grad
	 */
	private void backTrackingLineSearch( OpenCLGrid3D tvGradient){//weighted TV
		double t=1.0,tmin=0.0000001;
		double alpha=0.3, beta=0.6;
		double delta=1.0f,temp1,temp2;
		
		double TV=TVgrad.getWeightedTVvalueCL(reconCL);
		double Gradnorm=alpha*openCLGrid3DNorm(tvGradient);
		
		tvGradient.getGridOperator().multiplyBy(tvGradient,(float) t);
		reconCL.getGridOperator().subtractBy(reconCL,tvGradient);
		temp1=TVgrad.getWeightedTVvalueCL(reconCL);
		temp2=t*Gradnorm;
		delta=temp1-TV+temp2;
		
		while(delta>0.0f && t>tmin)
		{
			
			t=t*beta;
			reconCL.getGridOperator().addBy(reconCL, tvGradient);
			
			tvGradient.getGridOperator().multiplyBy(tvGradient,(float) beta);
					
			reconCL.getGridOperator().subtractBy(reconCL,tvGradient);

			temp1=TVgrad.getWeightedTVvalueCL(reconCL); 
		
			temp2=t*Gradnorm;
			delta=temp1-TV+temp2;
		}
		//System.out.println("t="+t);
		step=t;
	}
	
	/**
	 * L2 norm of OpenCLGrid3D	
	 * @param reconCL
	 * @return
	 */
	private double openCLGrid3DNorm(OpenCLGrid3D reconCL) {
		OpenCLGrid3D tempReconCL=new OpenCLGrid3D(reconCL);
		tempReconCL.getGridOperator().multiplyBy(tempReconCL,tempReconCL);
		double n=reconCL.getGridOperator().sum(tempReconCL);
		tempReconCL.release();
		return n;
	}
	
	/**
	 * mean square error
	 * @param gridCL1
	 * @param gridCL2
	 * @return
	 */
	private double meanSquareError(OpenCLGrid3D gridCL1, OpenCLGrid3D gridCL2){
		OpenCLGrid3D tempCL=new OpenCLGrid3D(gridCL1);
		tempCL.getGridOperator().subtractBy(tempCL, gridCL2);
		double err=openCLGrid3DNorm(tempCL)/(gridCL1.getSize()[0]*gridCL1.getSize()[1]*gridCL1.getSize()[2]);
		tempCL.release();
		return err;
	}
	
	/**
	 * initialize the geometry
	 */
	public void initOpenCLDataStructure(){
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
		
		
		cbp=new ConeBeamProjector();
		cbbp=new ConeBeamBackprojector();
		sinoP=new OpenCLGrid2D(new Grid2D(width,height));
		updBP = new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));

	}
	
	/**
	 * get the measured sinogram
	 * @throws Exception
	 */
	public void getMeasuredSinoCL() throws Exception
	{
		sinoCL = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));
		sinoCL.getDelegate().prepareForDeviceOperation();		
		cbp.fastProjectRayDrivenCL(sinoCL,volCL);	
		sinoCL.show("sinoCL");
	}
	
	/**
	 * initialize the reconstructed image as 0
	 */
	public void initialReconCL()
	{
		reconCL=new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));
		reconCL.setSpacing(spacingX,spacingY,spacingZ);
		reconCL.getGridOperator().fill(reconCL, 0);
	
	}
		
	
	/**
	 * compute the normalization projections
	 * @throws Exception
	 */
	public void createNormProj() throws Exception{
		OpenCLGrid3D onesVol = new OpenCLGrid3D(new Grid3D(imgSizeX,imgSizeY,imgSizeZ));		
		onesVol.getGridOperator().fill(onesVol, 1);
	normSino = new OpenCLGrid3D(new Grid3D(width,height,maxProjs));		
	cbp.fastProjectRayDrivenCL(normSino,onesVol);			
		
	normSino.getGridOperator().addBy(normSino,(float) eps);
	//if(debug)normSino.show("normSino");
	}

	/**
	 * compute the projIndex_th normalization grids for backprojection
	 * @param projIndex
	 * @throws Exception
	 */
	protected void createNormGrid(int projIndex) throws Exception {
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid.release();
		normGrid=new OpenCLGrid3D(volCL);
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, projIndex);
		//normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		normGrid.getGridOperator().addBy(normGrid, (float)eps);
	}
	
	/**
	 * compute the mean normalization grid for backprojection
	 * @throws Exception
	 */
	public void createNormGrid() throws Exception {
		OpenCLGrid2D c_sinoCL=new OpenCLGrid2D(new Grid2D(width,height));		
		c_sinoCL.getGridOperator().fill(c_sinoCL, 1.0f);
		normGrid=new OpenCLGrid3D(volCL);
		normGrid.getGridOperator().fill(normGrid, 0);
		cbbp.backprojectPixelDrivenCL(normGrid, c_sinoCL, 0);
		normGrid.getGridOperator().fill(normGrid, (float)normGrid.getGridOperator().sum(normGrid)/(normGrid.getNumberOfElements()));			
		//normGrid.getGridOperator().addBy(normGrid, (float)eps);
	}

	/**
	 * out put the result at each iteration
	 * @throws IOException
	 */
	public void outPutError() throws IOException{
		
		if (iter % showInt== 0)
		{
			reconCL.show(iter+"_th iteration");				
		}
	
		error =meanSquareError(reconCL, reconCLpre);	
		error2=meanSquareError(reconCL, volCL);
		System.out.println(iter+":  error=" + error+" error2= "+error2);	
	
		reconCLpre.release();
	}
}

