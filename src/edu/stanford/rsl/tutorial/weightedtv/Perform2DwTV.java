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

//import TVreconstruction.FanBeamProjector2D;
//import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;

/**
 *[1]Candes. Enhancing Sparsity by Reweighted l1 Minimization
 * 
 * 
 * @author Yixing Huang
 *
 */
public class Perform2DwTV {
	static boolean isOpencl=true;
	
	
	 private static String path="D:\\wTV\\";
	 private static File outPutDir=new File(path+"log.txt");
	 private static BufferedWriter bw;
	
	 static int maxIter = 500;
	 static int maxTVIter = 10;
	 static double epsilon = 1.0e-10;
	 static double error = 1.0;
	 static double error2;
	 static int iter = 0;
	 static double step=0;
	
	
	static Grid2D phan;
	static Grid2D sinogram;
	static Grid1D sinodiff1D;
	static Grid2D recon;
	static Grid2D recon_pre;
	static Grid2D wSinogram;
	static Grid2D wGrids;
	static Grid2D localImageUpdate;
	//wTV
	static TVGradient tvGrad;
	static Grid2D tv_Gradient;
	//Geometry
	static int imgSizeX;
	static int imgSizeY;
	static double gammaM;
	static double maxT;
	static double deltaT;
	static double focalLength;
	static double maxBeta;
	static double deltaBeta;
	static int spacingX=1,spacingY=1;
	static FanBeamProjector2D projector;
	static FanBeamBackprojector2D backproj;
	
	public static void main(String[] args) throws IOException {
		new ImageJ();
		
		imgSizeX = 256;
		imgSizeY=imgSizeX;
		phan = new SheppLogan(imgSizeX, false);
		phan.setSpacing(spacingX,spacingY);
		phan.show("The Phantom");
	   	
		if(!outPutDir.exists()){
		    outPutDir.getParentFile().mkdirs();
			outPutDir.createNewFile();
		}
		
		bw=new BufferedWriter(new FileWriter(outPutDir));
	   
	   initialGeometry();
	   initialSART();
	   createNormProj();
       createNormGrids();	
       
	
       //initial wTV
		tv_Gradient= new Grid2D(imgSizeX, imgSizeY);
		tv_Gradient.setSpacing(1, 1);
		tvGrad = new TVGradient(recon);
		tvGrad.initialWmatrix();
   

		
		while (error > epsilon && iter <=maxIter) {
		 sartIterate();		 
		 wTVIterate();
		 System.out.println(iter+"wTV="+tvGrad.getwTVvalue(recon));
		 tvGrad.wMatrixUpdate(recon);
		 outPutError();
		 iter++;
		}
		bw.close();
		recon.show("Reconstructed Image");
	}
	
	
	

private static void sartIterate(){
	recon_pre=new Grid2D(recon);
	for (int theta = 0; theta < sinogram.getSize()[1]; theta++) {
		if(isOpencl)
			sinodiff1D= projector.projectRayDriven1DCL(recon, theta);//get the projection of the current updated image at angle theta
		else
			sinodiff1D = projector.projectRayDriven1D(recon, theta);
		  
		sinodiff1D=(Grid1D)NumericPointwiseOperators.subtractedBy(sinodiff1D, sinogram.getSubGrid(theta));//compare with the measured sinogram, get the difference

		NumericPointwiseOperators.divideBy(sinodiff1D, wSinogram.getSubGrid(theta));

		if(isOpencl)
			localImageUpdate = backproj.backprojectPixelDriven1DCL(sinodiff1D, theta);
		else
			localImageUpdate = backproj.backprojectPixelDriven1D(sinodiff1D, theta);
	
		double stepSize = -1.0;

		NumericPointwiseOperators.multiplyBy(localImageUpdate,(float) stepSize);
		NumericPointwiseOperators.divideBy(localImageUpdate, wGrids);
	
       
		NumericPointwiseOperators.addBy(recon, localImageUpdate);
	
	}
	 recon.getGridOperator().removeNegative(recon);
}

private static void wTVIterate()
{
	double TV=tvGrad.getwTVvalue(recon);
	int i=0;
	double deltaTV=-1.0;
	while(i<maxTVIter&& deltaTV<0){
		double preTV=TV;
		tv_Gradient=tvGrad.computewTVGradient(recon);
		NumericPointwiseOperators.divideBy(tv_Gradient, (float) tvGrad.max_Value);
		wBTLsearch(tv_Gradient);//****************
		TV=tvGrad.getwTVvalue(recon);
		
		deltaTV=TV-preTV;
		System.out.println(iter+" i="+i+" wTV="+TV+" step="+step);
		i++;				
	} 
}

private static void wBTLsearch(Grid2D grad){//weighted TV
	double t=1.0,tmin=1.0e-7;
	double alpha=0.3, beta=0.6;
	double delta=1.0f,temp1,temp2;
	double TV=tvGrad.getwTVvalue(recon);
	double Gradnorm=alpha*grid2Dnorm(grad);
	Grid2D temp=new Grid2D(grad);
	NumericPointwiseOperators.multiplyBy(temp,(float) t);
	temp1=tvGrad.getwTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); 
	temp2=t*Gradnorm;
	delta=temp1-TV+temp2;
	while(delta>0.0f&& t>tmin)
	{
		t=t*beta;
		temp=(Grid2D)grad.clone();
		NumericPointwiseOperators.multiplyBy(temp,(float) t);
		temp1=tvGrad.getwTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); //
		temp2=t*Gradnorm;
		delta=temp1-TV+temp2;
	}
	step=t;
	NumericPointwiseOperators.subtractBy(recon, temp);;
}

private static void initialGeometry()
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
    backproj=new FanBeamBackprojector2D(focalLength,deltaT, deltaBeta, imgSizeX, imgSizeY);  
}

private static void initialSART(){
    if(isOpencl)
	      sinogram = (Grid2D) projector.projectRayDrivenCL(phan);
    else
	   sinogram = (Grid2D) projector.projectRayDriven(phan);
    sinogram.show("sinogram");
    backproj.initSinogramParams(sinogram);
	
		recon = new Grid2D(imgSizeX, imgSizeY);
		recon.setSpacing(spacingX, spacingY);
		NumericPointwiseOperators.fill(recon, 0);// initialization
		
		
		sinodiff1D = new Grid1D(sinogram.getSize()[0]);
		sinodiff1D.setSpacing(deltaT);
		localImageUpdate = new Grid2D(imgSizeX, imgSizeY);
		localImageUpdate.setSpacing(spacingX, spacingY);
		localImageUpdate.setOrigin(imgSizeX/2,imgSizeY/2);
 
}


	private static void createNormProj(){
		//projection normalization weights
		Grid2D C_phan=new Grid2D(imgSizeX,imgSizeY);//Constant grid with all values as 1;
		C_phan.setSpacing(spacingX,spacingY);
		NumericPointwiseOperators.fill(C_phan,1.0f);

		if(isOpencl)
			wSinogram=(Grid2D)projector.projectRayDrivenCL(C_phan);
		else
			wSinogram=(Grid2D)projector.projectRayDriven(C_phan);
		NumericPointwiseOperators.addBy(wSinogram,(float) epsilon);
	}

	private static void createNormGrids(){
		//backprojection normalization weights
		Grid1D C_sino1D=new Grid1D(sinogram.getSize()[0]);
		C_sino1D.setSpacing(deltaT);
		NumericPointwiseOperators.fill(C_sino1D,1.0f);

		wGrids=new Grid2D(imgSizeX,imgSizeY);
		if(isOpencl)
			wGrids=backproj.backprojectPixelDriven1DCL(C_sino1D, 0);
		else
			wGrids=backproj.backprojectPixelDriven1D(C_sino1D, 0);
	
		NumericPointwiseOperators.addBy(wGrids,(float) epsilon);
	}
	
	private static void outPutError() throws IOException{
		if (iter % 5== 0){
			recon.show(iter+"_th iteration");
			ImagePlus imp2=ImageUtil.wrapGrid(recon, null);
			IJ.saveAs(imp2, "Tiff", (path+iter+"thResult.tif"));	
		}
		error =mse(recon, recon_pre);	
		error2=mse(recon, phan);
		System.out.println(iter+":  error=" + error+" error2= "+error2);
		bw.write(iter+":  error=" + error+" error2= "+error2+"\r\n");
		bw.flush();	
	}

	private static double mse(Grid2D recon, Grid2D recon_data) {
		double err = 0;
		Grid2D temp = new Grid2D(recon);
		NumericPointwiseOperators.subtractBy(temp, recon_data);
		err = grid2Dnorm(temp);
		err=err / (temp.getSize()[0] * temp.getSize()[1]);
		
		return err;
	}
	

	private static double grid2Dnorm(Grid2D recon) {
		
		double d = 0;
		for (int row = 0; row < recon.getSize()[0]; row++)
			for (int col = 0; col < recon.getSize()[1]; col++)
				d = d + recon.getAtIndex(row, col) * recon.getAtIndex(row, col);
		return d;
	}

	
}
