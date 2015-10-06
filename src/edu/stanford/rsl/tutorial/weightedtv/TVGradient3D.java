package edu.stanford.rsl.tutorial.weightedtv;

import ij.IJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;


public class TVGradient3D {
	public float eps = 0.1f;
	public float weps=0.001f;
	public Grid3D imgGradient;
	public Grid3D tvGradient;
	public double maxValue=0.0;
	public Grid3D weightMatrix;//weights for weighted TV

	//OpenCL
	public OpenCLGrid3D imgGradientCL;
	public OpenCLGrid3D tvGradientCL;
	public OpenCLGrid3D weightMatrixCL;
	TVOpenCLGridOperators tvOperators;

	
	private Grid3D onesTemp;
/**
 * constructor 	
 * @param img
 */
public TVGradient3D(Grid3D img){
	this.imgGradient=new Grid3D(img);
	this.tvGradient=new Grid3D(img);
    onesTemp=new Grid3D(weightMatrix);
	NumericPointwiseOperators.fill(onesTemp, 1.0f);
}
	
	
/**
 * constructor with OpenCL
 * @param imgCL
 */
public TVGradient3D(OpenCLGrid3D imgCL)
{
	this.tvGradientCL=new OpenCLGrid3D(imgCL);
	tvGradientCL.getGridOperator().fill(tvGradientCL, 0);
	weightMatrixCL=new OpenCLGrid3D(tvGradientCL);
	tvOperators=new TVOpenCLGridOperators();	
}

/**
 * constructor
 * @param size
 */
public TVGradient3D(int[] size)
{this.tvGradientCL=new OpenCLGrid3D(new Grid3D(size[0],size[1],size[2]));
tvGradientCL.getGridOperator().fill(tvGradientCL, 0);
weightMatrixCL=new OpenCLGrid3D(tvGradientCL);
tvOperators=new TVOpenCLGridOperators();	
}

/**
 * initial weight matrix as 1
 */
public void initialWeightMatrix(){
	weightMatrix=new Grid3D(tvGradient);
	NumericPointwiseOperators.fill(weightMatrix, 1.0f);
}

/**
 * initial weight matrix as 1 using OpenCL
 */
public void initialWeightMatrixCL(){
	
	weightMatrixCL.getGridOperator().fill(weightMatrixCL, 1.0f);
}

/**
 * compute image gradient
 * @param img
 */
public void computeImageGradient(Grid3D img){//Compute the gradient of the img
	this.imgGradient=new Grid3D(img);
	double Hdiff,Vdiff,Zdiff;
	for (int i = 0; i < imgGradient.getSize()[0]; i++) {
		for (int j = 0; j < imgGradient.getSize()[1]; j++) {
			for(int k=0;k<imgGradient.getSize()[2];k++){
			double fij = img.getAtIndex(i, j,k);
			double fijl = fij;
			double fiju = fij;
			double fijt=fij;
			if (i > 0)
				fijl = img.getAtIndex(i - 1, j,k);
			if (j > 0)
				fiju = img.getAtIndex(i, j - 1,k);
			if(k>0)
				fijt = img.getAtIndex(i, j,k-1);
			Hdiff=fij-fijl;
			Vdiff=fij-fiju;
			Zdiff=fij-fijt;
			this.imgGradient.setAtIndex(i, j,k, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff));
			}
		}
	}
}

/**
 * compute image gradient with OpenCL
 * @param imgCL
 */
public void computeImageGradientCL(OpenCLGrid3D imgCL){
	//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
	this.imgGradientCL=new OpenCLGrid3D(tvGradientCL);
	tvOperators.computeImageGradient(imgCL, imgGradientCL);
}

/**
 * update the weight matrix
 */
public void weightMatrixUpdate(){//Update the weights for weighted TV
	
	Grid3D gradient_temp=new Grid3D(this.imgGradient);
	NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
	weightMatrix=(Grid3D)NumericPointwiseOperators.dividedBy(onesTemp,gradient_temp );	
}

/**
 * update the weight matrix with OpenCL
 * @param imgCL
 */
public void weightMatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdate(imgCL, weightMatrixCL, weps);
}

/**
 * 
 * @param imgCL
 */
public void adaptiveWeightMatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().computeImageGradient(imgCL,this.tvGradientCL);
	TVOpenCLGridOperators.getInstance().computeAdaptiveWeightMatrixUpdate(tvGradientCL, this.weightMatrixCL, weps);
}
/**
 * update weight matrix for anisotropic weighted TV (AwTV)
 * @param imgCL
 */
public void anisotropicWeightedTVWeightMatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().computeDirectionalWeightedTVWeightMatrixUpdate(imgCL, weightMatrixCL, weps);
}

/**
 * update weight matrix, here only compute image gradient in each XY plane, not include Z direction
 * @param imgCL
 */
public void weightMatrixCLUpdate2(OpenCLGrid3D imgCL){//do TV in each Z slice
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdate2(imgCL, weightMatrixCL, weps);
}

/**
 * compute nonweighted TV value
 * @return
 */
public double getTVvalue()//Compute ComputeGradient(Grid2D img)
{
	double TV=0.0;
	/*for (int i = 0; i < gradient.getSize()[0]; i++) 
		for (int j = 0; j < gradient.getSize()[1]; j++) 
			for(int k=0;k<gradient.getSize()[2];k++)
			TV+=gradient.getAtIndex(i, j,k);*/
	TV=NumericPointwiseOperators.sum(imgGradient);
	return TV;
}

/**
 * compute weighted TV value
 * @return
 */
public double getWeightedTVvalue()//Compute ComputeGradient(Grid2D img)
{
	double wTV=0.0;
	
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(imgGradient, weightMatrix));

	return wTV;
}

/**
 * get weighted TV value with OpenCL
 * @param imgCL
 * @return
 */
public double getWeightedTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getWeightedTV(imgCL, weightMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

/**
 * get TV value in adaptive weighted TV
 * @param imgCL
 * @return
 */
public double getWeightedTVvalueCLAdaptive(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getAdaptiveWeightedTV(imgCL, weightMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

/**
 * compute directional weighted TV with OpenCL, in Y direction gradient has a large weight, B=100 for instance
 * @param imgCL
 * @return
 */
public double getDirectionalWeightedTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getDirectionalWeightedTV(imgCL, weightMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

/**
 * compute weighted TV value with OpenCL, here compute image gradient in each XY plane
 * @param imgCL
 * @return
 */
public double getWeightedTVvalueCL2(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getWeightedTV2(imgCL, weightMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}


public Grid3D computeTVgradient(Grid3D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.maxValue=0.0f;
		for (int i = 0; i < tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
				for(int k=0;k<imgGradient.getSize()[2];k++){
				double fijk = img.getAtIndex(i, j,k);
	
				double fl = fijk;
				double fr = fijk;
				double fu = fijk;
				double fd = fijk;
				double ft=fijk;
				double fb=fijk;
				double fld = fijk;
				double fru = fijk;
				double frt=fijk;
				double flb=fijk;
				double fdt=fijk;
				double fub=fijk;
				//Not at border
				if (i > 0)
					fl = img.getAtIndex(i - 1, j,k);
				if (i < tvGradient.getSize()[0] - 1)
					fr = img.getAtIndex(i + 1, j,k);
				if (j > 0)
					fu = img.getAtIndex(i, j - 1,k);
				if (j < tvGradient.getSize()[1] - 1)
					fd = img.getAtIndex(i, j + 1,k);
				if(k>0)
					ft=img.getAtIndex(i, j, k-1);
				if(k<tvGradient.getSize()[2]-1)
					fb=img.getAtIndex(i, j, k+1);
				if (i > 0 & j < tvGradient.getSize()[1] - 1)
					fld = img.getAtIndex(i - 1, j + 1,k);
				if (i < tvGradient.getSize()[0] - 1 & j > 0)
					fru = img.getAtIndex(i + 1, j - 1,k);
				if(k>0 & i<tvGradient.getSize()[0]-1)
					frt=img.getAtIndex(i+1, j, k-1);
				if(i>0 & k<tvGradient.getSize()[2]-1)
					flb=img.getAtIndex(i-1, j, k+1);
				if(j<(tvGradient.getSize()[1] - 1) & k>0)
					fdt=img.getAtIndex(i, j+1, k-1);				
				if(k<(tvGradient.getSize()[2] - 1) & j>0)
					fub=img.getAtIndex(i, j-1, k+1);
				
				double vij = (3 * fijk -  fl - fu-ft)
						/ Math.sqrt(eps + (fijk- fl) * (fijk - fl)+ (fijk - fu) * (fijk - fu)+(fijk-ft)*(fijk-ft))
						-  (fr - fijk)
						/ Math.sqrt(eps + (fr - fijk) * (fr - fijk)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						- (fd - fijk)
						/ Math.sqrt(eps + (fd - fijk) * (fd - fijk)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-(fb-fijk)
						/Math.sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fijk)*(fb-fijk));
				if (Math.abs(vij)>maxValue)
					maxValue=Math.abs(vij);
				tvGradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return tvGradient;
	}
	
/**
 * compute weighted TV gradient
 * @param img
 * @return
 */
	public Grid3D computeWeightedTVGradient(Grid3D img) {//weighted TV gradient
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		double vij;
		double wr,wd,wb;
		for (int i = 0; i < tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
				for(int k=0;k<imgGradient.getSize()[2];k++){
				double fijk = img.getAtIndex(i, j,k);
				double fl = fijk;
				double fr = fijk;
				double fu = fijk;
				double fd = fijk;
				double ft=fijk;
				double fb=fijk;
				double fld = fijk;
				double fru = fijk;
				double frt=fijk;
				double flb=fijk;
				double fdt=fijk;
				double fub=fijk;
				//Not at border
				if (i > 0)
					fl = img.getAtIndex(i - 1, j,k);
				if (i < tvGradient.getSize()[0] - 1)
				{
					fr = img.getAtIndex(i + 1, j,k);
					wr=weightMatrix.getAtIndex(i+1, j, k);}
				else
					wr=0;
				if (j > 0)
					fu = img.getAtIndex(i, j - 1,k);
				if (j < tvGradient.getSize()[1] - 1){
					fd = img.getAtIndex(i, j + 1,k);
					wd=weightMatrix.getAtIndex(i, j+1, k);}
				else
					wd=0;
				if(k>0)
					ft=img.getAtIndex(i, j, k-1);
				if(k<tvGradient.getSize()[2]-1){
					fb=img.getAtIndex(i, j, k+1);
					wb=weightMatrix.getAtIndex(i, j, k+1);
				}
				else
					wb=0;
				if (i > 0 & j < tvGradient.getSize()[1] - 1)
					fld = img.getAtIndex(i - 1, j + 1,k);
				if (i < tvGradient.getSize()[0] - 1 & j > 0)
					fru = img.getAtIndex(i + 1, j - 1,k);
				if(k>0 & i<tvGradient.getSize()[0]-1)
					frt=img.getAtIndex(i+1, j, k-1);
				if(i>0 & k<tvGradient.getSize()[2]-1)
					flb=img.getAtIndex(i-1, j, k+1);
				if(j<(tvGradient.getSize()[1] - 1) & k>0)
					fdt=img.getAtIndex(i, j+1, k-1);				
				if(k<(tvGradient.getSize()[2] - 1) & j>0)
					fub=img.getAtIndex(i, j-1, k+1);
				//Not at border           
				vij = weightMatrix.getAtIndex(i, j, k)*(3 * fijk -  fl - fu-ft)
						/ Math.sqrt(eps + (fijk- fl) * (fijk - fl)+ (fijk - fu) * (fijk - fu)+(fijk-ft)*(fijk-ft))
						- wr* (fr - fijk)
						/ Math.sqrt(eps + (fr - fijk) * (fr - fijk)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						-wd* (fd - fijk)
						/ Math.sqrt(eps + (fd - fijk) * (fd - fijk)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-wb*(fb-fijk)
						/Math.sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fijk)*(fb-fijk));
				if (Math.abs(vij)>maxValue)
					maxValue=Math.abs(vij);
				tvGradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return tvGradient;
	}
	
	/**
	 * compute weighted TV gradient with OpenCL
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeWeightedTVGradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradient(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	/**
	 * 
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeAdaptiveWeightedTVGradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeAdaptiveWeightedTVGradient(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	/**
	 * compute anisotropic weighted TV gradient
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeAnisotropicWeightedTVGradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeDirectionalWeightedTVGradient(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	/**
	 * compute wTV gradient, only in XY plane
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeWeightedTVGradient2(OpenCLGrid3D imgCL)//do TV in each Z slide
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradient2(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	/**
	 * update weight matrix for anisotropic weighted TV (AwTV) along Y direction
	 * @param imgCL
	 */
	public void weightMatrixCLUpdateY(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdateY(imgCL, weightMatrixCL, weps);
	}
	
	/**
	 * get weighted TV value for anisotropic weighted TV (AwTV) along Y direction
	 * @param imgCL
	 * @return
	 */
	public double getWeightedTVvalueCLY(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		tvOperators.getWeightedTVY(imgCL, weightMatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	/**
	 * get weighted TV gradient for anisotropic weighted TV (AwTV) along Y direction
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeWeightedTVGradientY(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradientY(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}

	/**
	 * update weight matrix for anisotropic weighted TV (AwTV) along X direction
	 * @param imgCL
	 */
	public void weightMatrixCLUpdateX(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdateX(imgCL, weightMatrixCL, weps);
	}
	
	/**
	 * get weighted TV value for anisotropic weighted TV (AwTV) along X direction
	 * @param imgCL
	 * @return
	 */
	public double getWeightedTVvalueCLX(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		tvOperators.getWeightedTVY(imgCL, weightMatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	/**
	 * get weighted TV gradient for anisotropic weighted TV (AwTV) along X direction
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid3D computeWeightedTVGradientX(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradientX(imgCL, weightMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
}

