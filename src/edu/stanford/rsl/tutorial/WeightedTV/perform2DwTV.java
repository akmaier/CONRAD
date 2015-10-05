package edu.stanford.rsl.tutorial.WeightedTV;

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
public class perform2DwTV {
	static boolean isOpencl=true;
	
	
	 private static String path="D:\\wTV\\";
	 private static File outPutDir=new File(path+"log.txt");
	 private static BufferedWriter bw;
	
	 static int maxIter = 500;
	 static int TVmaxIter = 10;
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
	static Grid2D Wsinogram;
	static Grid2D Wgrids;
	static Grid2D localImageUpdate;
	//wTV
	static TVgradient TVgrad;
	static Grid2D tv_gradient;
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
		tv_gradient= new Grid2D(imgSizeX, imgSizeY);
		tv_gradient.setSpacing(1, 1);
		TVgrad = new TVgradient(recon);
		TVgrad.initialWmatrix();
   

		
		while (error > epsilon && iter <=maxIter) {
		 SartIterate();		 
		 WTVIterate();
		 System.out.println(iter+"wTV="+TVgrad.getwTVvalue(recon));
		 TVgrad.WmatrixUpdate(recon);
		 OutputError();
		 iter++;
		}
		bw.close();
		recon.show("Reconstructed Image");
	}
	
	
	

private static void SartIterate(){
	recon_pre=new Grid2D(recon);
	for (int theta = 0; theta < sinogram.getSize()[1]; theta++) {
		if(isOpencl)
			sinodiff1D= projector.projectRayDriven1DCL(recon, theta);//get the projection of the current updated image at angle theta
		else
			sinodiff1D = projector.projectRayDriven1D(recon, theta);
		  
		sinodiff1D=(Grid1D)NumericPointwiseOperators.subtractedBy(sinodiff1D, sinogram.getSubGrid(theta));//compare with the measured sinogram, get the difference

		NumericPointwiseOperators.divideBy(sinodiff1D, Wsinogram.getSubGrid(theta));

		if(isOpencl)
			localImageUpdate = backproj.backprojectPixelDriven1DCL(sinodiff1D, theta);
		else
			localImageUpdate = backproj.backprojectPixelDriven1D(sinodiff1D, theta);
	
		double stepSize = -1.0;

		NumericPointwiseOperators.multiplyBy(localImageUpdate,(float) stepSize);
		NumericPointwiseOperators.divideBy(localImageUpdate, Wgrids);
	
       
		NumericPointwiseOperators.addBy(recon, localImageUpdate);
	
	}
	 recon.getGridOperator().removeNegative(recon);
}

private static void WTVIterate()
{
	double TV=TVgrad.getwTVvalue(recon);
	int i=0;
	double deltaTV=-1.0;
	while(i<TVmaxIter&& deltaTV<0){
		double preTV=TV;
		tv_gradient=TVgrad.ComputewTVgradient(recon);
		NumericPointwiseOperators.divideBy(tv_gradient, (float) TVgrad.max_Value);
		wBTLsearch(tv_gradient);//****************
		TV=TVgrad.getwTVvalue(recon);
		
		deltaTV=TV-preTV;
		System.out.println(iter+" i="+i+" wTV="+TV+" step="+step);
		i++;				
	} 
}

private static void wBTLsearch(Grid2D grad){//weighted TV
	double t=1.0,tmin=1.0e-7;
	double alpha=0.3, beta=0.6;
	double delta=1.0f,temp1,temp2;
	double TV=TVgrad.getwTVvalue(recon);
	double Gradnorm=alpha*Grid2Dnorm(grad);
	Grid2D temp=new Grid2D(grad);
	NumericPointwiseOperators.multiplyBy(temp,(float) t);
	temp1=TVgrad.getwTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); 
	temp2=t*Gradnorm;
	delta=temp1-TV+temp2;
	while(delta>0.0f&& t>tmin)
	{
		t=t*beta;
		temp=(Grid2D)grad.clone();
		NumericPointwiseOperators.multiplyBy(temp,(float) t);
		temp1=TVgrad.getwTVvalue((Grid2D)NumericPointwiseOperators.subtractedBy(recon,temp)); //
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
			Wsinogram=(Grid2D)projector.projectRayDrivenCL(C_phan);
		else
			Wsinogram=(Grid2D)projector.projectRayDriven(C_phan);
		NumericPointwiseOperators.addBy(Wsinogram,(float) epsilon);
	}

	private static void createNormGrids(){
		//backprojection normalization weights
		Grid1D C_sino1D=new Grid1D(sinogram.getSize()[0]);
		C_sino1D.setSpacing(deltaT);
		NumericPointwiseOperators.fill(C_sino1D,1.0f);

		Wgrids=new Grid2D(imgSizeX,imgSizeY);
		if(isOpencl)
			Wgrids=backproj.backprojectPixelDriven1DCL(C_sino1D, 0);
		else
			Wgrids=backproj.backprojectPixelDriven1D(C_sino1D, 0);
	
		NumericPointwiseOperators.addBy(Wgrids,(float) epsilon);
	}
	
	private static void OutputError() throws IOException{
		if (iter % 5== 0){
			recon.show(iter+"_th iteration");
			ImagePlus imp2=ImageUtil.wrapGrid(recon, null);
			IJ.saveAs(imp2, "Tiff", (path+iter+"thResult.tif"));	
		}
		error =MSE(recon, recon_pre);	
		error2=MSE(recon, phan);
		System.out.println(iter+":  error=" + error+" error2= "+error2);
		bw.write(iter+":  error=" + error+" error2= "+error2+"\r\n");
		bw.flush();	
	}

	private static double MSE(Grid2D recon, Grid2D recon_data) {
		double err = 0;
		Grid2D temp = new Grid2D(recon);
		NumericPointwiseOperators.subtractBy(temp, recon_data);
		err = Grid2Dnorm(temp);
		err=err / (temp.getSize()[0] * temp.getSize()[1]);
		
		return err;
	}
	

	private static double Grid2Dnorm(Grid2D recon) {
		
		double d = 0;
		for (int row = 0; row < recon.getSize()[0]; row++)
			for (int col = 0; col < recon.getSize()[1]; col++)
				d = d + recon.getAtIndex(row, col) * recon.getAtIndex(row, col);
		return d;
	}

	
}
