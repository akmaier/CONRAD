package edu.stanford.rsl.tutorial.weightedtv;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;

/**
 *This is an example to apply weighted total variation (wTV) on 2D fan-beam limited angle tomography.
 *wTV is according to [1].
 *[1]Candes. Enhancing Sparsity by Reweighted l1 Minimization
 * @author Yixing Huang
 *
 */
public class Perform2DWeightedTV {
	static boolean isOpencl=true;

	 private  int maxIter = 500;
	 public  int maxTVIter = 10;
	 private double epsilon = 1.0e-10;
	 public double error = 1.0;
	 public double error2;
	 private int iter = 0;
	 private double step=0;
	
	//SART
	 public Grid2D phan;
	 public Grid2D recon;
	 private Grid2D sinogram;
	 private Grid1D sinoDiff1D;
	 private Grid2D reconPre;
	 private Grid2D normSinogram;
	 private Grid2D normGrids;
	 private Grid2D localImageUpdate;
	//weighted TV
	public TVGradient TVGrad;
	//public Grid2D TVGradient;
	//Geometry
	public int imgSizeX;
	public int imgSizeY;
	private double gammaM;
	private double maxT;
	private double deltaT;
	private double focalLength;
	private double maxBeta;
	private double deltaBeta;
	private int spacingX=1,spacingY=1;
	private FanBeamProjector2D projector;
	private FanBeamBackprojector2D backProj;
	
	public static void main(String[] args) throws IOException {
		new ImageJ();
		Perform2DWeightedTV tv2D=new Perform2DWeightedTV();
		tv2D.imgSizeX = 256;
		tv2D.imgSizeY=tv2D.imgSizeX;
		tv2D.phan = new SheppLogan(tv2D.imgSizeX, false);
		tv2D.phan.show("The Phantom");
	   	
	   
	   tv2D.initialGeometry();
	   tv2D.initialSART();
	   tv2D.createNormProj();
       tv2D.createNormGrids();	
       
	
       //initial wTV
		tv2D.TVGrad = new TVGradient(tv2D.recon);

		while (tv2D.error > tv2D.epsilon && tv2D.iter <=tv2D.maxIter) {
		 tv2D.sartIterate();		 
		 tv2D.weightedTVIterate();

		 tv2D.TVGrad.weightMatrixUpdate(tv2D.recon);
		 tv2D.outPutError();
		 tv2D.iter++;
		}

		tv2D.recon.show("Reconstructed Image");
	}
	
	
	
/**
 * SART iteration part
 */
public void sartIterate(){
	reconPre=new Grid2D(recon);
	for (int theta = 0; theta < sinogram.getSize()[1]; theta++) {
		if(isOpencl)
			sinoDiff1D= projector.projectRayDriven1DCL(recon, theta);//get the projection of the current updated image at angle theta
		else
			sinoDiff1D = projector.projectRayDriven1D(recon, theta);
		  
		sinoDiff1D=(Grid1D)NumericPointwiseOperators.subtractedBy(sinoDiff1D, sinogram.getSubGrid(theta));//compare with the measured sinogram, get the difference

		NumericPointwiseOperators.divideBy(sinoDiff1D, normSinogram.getSubGrid(theta));

		if(isOpencl)
			localImageUpdate = backProj.backprojectPixelDriven1DCL(sinoDiff1D, theta);
		else
			localImageUpdate = backProj.backprojectPixelDriven1D(sinoDiff1D, theta);
	
		double stepSize = -1.0;

		NumericPointwiseOperators.multiplyBy(localImageUpdate,(float) stepSize);
		NumericPointwiseOperators.divideBy(localImageUpdate, normGrids);
	
       
		NumericPointwiseOperators.addBy(recon, localImageUpdate);
	
	}
	 recon.getGridOperator().removeNegative(recon);
}

/**
 * weighted TV gradient descent part
 */
public void weightedTVIterate()
{
	Grid2D TVGradient;
	double TV=TVGrad.getWeightedTVvalue(recon);
	int i=0;
	double deltaTV=-1.0;
	while(i<maxTVIter&& deltaTV<0){
		double preTV=TV;
		TVGradient=TVGrad.computewTVGradient(recon);
		NumericPointwiseOperators.divideBy(TVGradient, (float) TVGrad.maxValue);
		backTrackingLineSearch(TVGradient);//****************
		TV=TVGrad.getWeightedTVvalue(recon);
		
		deltaTV=TV-preTV;
		System.out.println(iter+" i="+i+" wTV="+TV+" step="+step);
		i++;				
	} 
}

/**
 * Using back tracking line search algorithm to find the step size for weighted TV gradient descent
 * @param grad
 */
private void backTrackingLineSearch(Grid2D grad){//weighted TV
	double t=1.0,tmin=1.0e-7;
	double alpha=0.3, beta=0.6;
	double delta=1.0f,temp1,temp2;
	double TV=TVGrad.getWeightedTVvalue(recon);
	double Gradnorm=alpha*grid2DNorm(grad);
	Grid2D temp=new Grid2D(grad);
	NumericPointwiseOperators.multiplyBy(temp,(float) t);
	temp1=TVGrad.getWeightedTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); 
	temp2=t*Gradnorm;
	delta=temp1-TV+temp2;
	while(delta>0.0f&& t>tmin)
	{
		t=t*beta;
		temp=(Grid2D)grad.clone();
		NumericPointwiseOperators.multiplyBy(temp,(float) t);
		temp1=TVGrad.getWeightedTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); //
		temp2=t*Gradnorm;
		delta=temp1-TV+temp2;
	}
	step=t;
	NumericPointwiseOperators.subtractBy(recon, temp);;
}

/**
 * The limited angle tomography geometry
 */
public void initialGeometry()
{
	gammaM = 10. * Math.PI / 180;
	// maxT = length of the detector array
	maxT =(int)(imgSizeX*1.5f);
	// deltaT = size of one detector element
	deltaT = 1.0;
	// focalLength = focal length
	focalLength = (maxT / 2.0 - 0.5) * deltaT / Math.tan(gammaM);
	// maxBeta = maximum rotation angle
	maxBeta =160. * Math.PI/180.0;
	// deltaBeta = step size between source positions
	deltaBeta = 1.0 * Math.PI / 180.0;
	projector = new FanBeamProjector2D(
			focalLength, maxBeta, deltaBeta, maxT, deltaT);
    backProj=new FanBeamBackprojector2D(focalLength,deltaT, deltaBeta, imgSizeX, imgSizeY);  
}

/**
 * Initial SART grids
 */
public void initialSART(){
    if(isOpencl)
	      sinogram = (Grid2D) projector.projectRayDrivenCL(phan);
    else
	   sinogram = (Grid2D) projector.projectRayDriven(phan);
    sinogram.show("sinogram");
    backProj.initSinogramParams(sinogram);
	
		recon = new Grid2D(imgSizeX, imgSizeY);
		recon.setSpacing(spacingX, spacingY);
		NumericPointwiseOperators.fill(recon, 0);// initialization
		
		
		sinoDiff1D = new Grid1D(sinogram.getSize()[0]);
		sinoDiff1D.setSpacing(deltaT);
		localImageUpdate = new Grid2D(imgSizeX, imgSizeY);
		localImageUpdate.setSpacing(spacingX, spacingY);
		localImageUpdate.setOrigin(imgSizeX/2,imgSizeY/2);
 
}

/**
 * compute the normalization projections
 */
public void createNormProj(){
	//projection normalization weights
	Grid2D C_phan=new Grid2D(imgSizeX,imgSizeY);//Constant grid with all values as 1;
	C_phan.setSpacing(spacingX,spacingY);
	NumericPointwiseOperators.fill(C_phan,1.0f);

	if(isOpencl)
		normSinogram=(Grid2D)projector.projectRayDrivenCL(C_phan);
	else
		normSinogram=(Grid2D)projector.projectRayDriven(C_phan);
	NumericPointwiseOperators.addBy(normSinogram,(float) epsilon);
	}

/**
 * compute normalization grids for backprojection
 */
public void createNormGrids(){
	//backprojection normalization weights
	Grid1D C_sino1D=new Grid1D(sinogram.getSize()[0]);
	C_sino1D.setSpacing(deltaT);
	NumericPointwiseOperators.fill(C_sino1D,1.0f);

	normGrids=new Grid2D(imgSizeX,imgSizeY);
	if(isOpencl)
		normGrids=backProj.backprojectPixelDriven1DCL(C_sino1D, 0);
	else
		normGrids=backProj.backprojectPixelDriven1D(C_sino1D, 0);
	
	NumericPointwiseOperators.addBy(normGrids,(float) epsilon);
	}

/**
 * output the difference between too iterations and output the difference from the ground truth	
 * @throws IOException
 */
public void outPutError() throws IOException{
	if (iter % 5== 0){
		recon.show(iter+"_th iteration");	
	}
	error =meanSquareError(recon, reconPre);	
	error2=meanSquareError(recon, phan);
	System.out.println(iter+":  error=" + error+" error2= "+error2);
}

/**
 * mean square error
 * @param imgGrid1
 * @param imgGrid2
 * @return
 */
private double meanSquareError(Grid2D imgGrid1, Grid2D imgGrid2) {
	double err = 0;
	Grid2D temp = new Grid2D(imgGrid1);
	NumericPointwiseOperators.subtractBy(temp, imgGrid2);
	err = grid2DNorm(temp);
	err=err / (temp.getSize()[0] * temp.getSize()[1]);
		
	return err;
}
	
/**
 * L2 norm
 * @param imgGrid
 * @return
 */
private double grid2DNorm(Grid2D imgGrid) {

	double d = 0;
	for (int row = 0; row < imgGrid.getSize()[0]; row++)
		for (int col = 0; col < imgGrid.getSize()[1]; col++)
			d = d + imgGrid.getAtIndex(row, col) * imgGrid.getAtIndex(row, col);
	return d;
}

	
}
