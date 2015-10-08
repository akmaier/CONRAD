package edu.stanford.rsl.tutorial.weightedtv;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;



public class TVGradient {
	public double eps = 0.1;
	public float weps=0.001f;
	public float thresMax=0.1f;
	public float thresMin=0.01f;
	public Grid2D imageGradient;
	public Grid2D tvGradient;
	public double maxValue=0.0;
	public Grid2D weightMatrix;//weights for weighted TV

	//Parameters for 2 level weighted TV
	public int numK=1400;
	public float rou=0.00001f;
	
	public OpenCLGrid2D imgGradientCL;
	public OpenCLGrid2D TVGradientCL;
	public OpenCLGrid2D weightMatrixCL;
	TVOpenCLGridOperators tvOperators;
	
	//anisotropic weighted TV
	public float a=1.0f;
	public float b=1.0f;
	
	private Grid2D onesTemp;
	private Grid2D gradientTemp;
	
	/**
	 * constructor
	 * @param img
	 */
public TVGradient(Grid2D img)
{
	tvGradient=new Grid2D(img);
	this.imageGradient=new Grid2D(img);
	NumericPointwiseOperators.fill(tvGradient, 0);
	onesTemp=new Grid2D(img);
	NumericPointwiseOperators.fill(onesTemp, 1.0f);
	gradientTemp=new Grid2D(img);
	initialWeightMatrix();
}

/**
 * constructor with OpenCL grids
 * @param imgCL
 */
public TVGradient(OpenCLGrid2D imgCL){
	this.TVGradientCL=new OpenCLGrid2D(imgCL);
	TVGradientCL.getGridOperator().fill(TVGradientCL, 0);
	imgGradientCL=new OpenCLGrid2D(imgCL);
	TVGradientCL.getGridOperator().fill(imgGradientCL, 0);
	tvOperators=new TVOpenCLGridOperators();
	 initialWeightMatrixCL();
}

/**
 * initial weight matrix as 1
 */
public void initialWeightMatrix(){
	weightMatrix=new Grid2D(tvGradient);
	NumericPointwiseOperators.fill(weightMatrix, 1.0f);
}

/**
 * initial weight matrix as 1 with OpenCL
 */
public void initialWeightMatrixCL(){
	weightMatrixCL=new OpenCLGrid2D(TVGradientCL);
	weightMatrixCL.getGridOperator().fill(weightMatrixCL, 1.0f);
}

/**
 * compute image gradient
 * @param img
 */
public void computeImageGradient(Grid2D img){//Compute the gradient of the img

	double Hdiff,Vdiff;
	for (int i = 0; i < imageGradient.getSize()[0]; i++) {
		for (int j = 0; j < imageGradient.getSize()[1]; j++) {
			double fij = img.getAtIndex(i, j);
			double fijl = fij;
			double fiju = fij;
			if (i > 0)
				fijl = img.getAtIndex(i - 1, j);
			if (j > 0)
				fiju = img.getAtIndex(i, j - 1);
			Hdiff=fij-fijl;
			Vdiff=fij-fiju;
			this.imageGradient.setAtIndex(i, j, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff));
		}
	}
}

/**
 * Update the weight matrix, don't need to compute image first
 * @param img
 */
public void weightMatrixUpdate(Grid2D img){
	this.computeImageGradient(img);
	gradientTemp=(Grid2D)this.imageGradient.clone();
	NumericPointwiseOperators.addBy(gradientTemp, (float)this.weps);
	weightMatrix=(Grid2D)NumericPointwiseOperators.dividedBy(onesTemp,gradientTemp );	
}

/**
 *  Update the weight matrix, need to compute image first
 */
public void weightMatrixUpdate(){
	gradientTemp=(Grid2D)this.imageGradient.clone();
	NumericPointwiseOperators.addBy(gradientTemp, (float)this.weps);
	weightMatrix=(Grid2D)NumericPointwiseOperators.dividedBy(onesTemp,gradientTemp );	
}

/**
 * Update the weight matrix, here compute the image gradient only in XY plane, not in Z direction
 * @param img
 */
public void weightMatrixUpdate2(Grid2D img){
	this.computeImageGradient(img);
	Grid2D gradient_temp=new Grid2D(this.imageGradient);
	gradient_temp=SetThreshold(gradient_temp,0.01f);
	weightMatrix=(Grid2D)NumericPointwiseOperators.dividedBy(onesTemp,gradient_temp );	
}

/**
 * set the image gradient values below the threshold as 1
 * @param imgGradient
 * @param thres
 * @return
 */
private Grid2D SetThreshold(Grid2D imgGradient,float thres){
	for(int i=0;i<imgGradient.getSize()[0];i++)
		for(int j=0;j<imgGradient.getSize()[1];j++){
			if(imgGradient.getAtIndex(i, j)<thres)
				imgGradient.setAtIndex(i, j, 1);	
		}
	return imgGradient;
}

/**
 * compute the non-weighted TV value
 * @param img
 * @return
 */
public double getTVvalue(Grid2D img)//Compute ComputeGradient(Grid2D img)
{
	double TV=0.0;
	this.computeImageGradient(img);
	TV=NumericPointwiseOperators.sum(this.imageGradient);
	
	return TV;
}

/**
 * compute the weighted TV value
 * @param img
 * @return
 */
public double getWeightedTVvalue(Grid2D img)
{
	double wTV=0.0;
	this.computeImageGradient(img);
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(weightMatrix, imageGradient));
	return wTV;
}

/**
 * get the weighted TV value, need to compute the image gradient first
 * @return
 */
public double getWeightedTVvalue()//Compute ComputeGradient(Grid2D img) first
{
	double wTV=0.0;
	
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(weightMatrix, imageGradient));
	return wTV;
}

/**
 * compute the non-weighted TV gradient
 * @param img
 * @return
 */
public Grid2D computeTVGradient(Grid2D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.maxValue=0.0f;
		for (int i = 0; i < tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
				double fij = img.getAtIndex(i, j);
	
				double fijl = fij;
				double fijr = fij;
				double fiju = fij;
				double fijd = fij;
				double fijld = fij;
				double fijru = fij;
				//Not at border
				if (i > 0)
					fijl = img.getAtIndex(i - 1, j);
				if (i < tvGradient.getSize()[0] - 1)
					fijr = img.getAtIndex(i + 1, j);
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < tvGradient.getSize()[1] - 1)
					fijd = img.getAtIndex(i, j + 1);
				if (i > 0 & j < tvGradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < tvGradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
				double vij = (4 * fij - 2 * fijl - 2 * fiju)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fij - fiju) * (fij - fiju))
						- 2* (fijr - fij)
						/ Math.sqrt(eps + (fijr - fij) * (fijr - fij)+ (fijr - fijru) * (fijr - fijru))
						- 2* (fijd - fij)
						/ Math.sqrt(eps + (fijd - fij) * (fijd - fij)+ (fijd - fijld)*(fijd - fijld));
				if (Math.abs(vij)>maxValue)
					maxValue=Math.abs(vij);
				tvGradient.setAtIndex(i, j, (float) vij);

			}
		}
		return tvGradient;
	}
	
	/**
	 * compute the non-weighted TV gradient, in XY plane, not include the Z direction
	 * @param img
	 * @return
	 */
	public Grid2D computeTVGradient2(Grid2D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.maxValue=0;
		for (int i = 0; i< tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
				double fij = img.getAtIndex(i, j);
				double fijl = fij;
				double fijr = fij;
				double fiju = fij;
				double fijd = fij;
				double fijld = fij;
				double fijru = fij;
				//Not at border
				if (i > 0)
					fijl = img.getAtIndex(i - 1, j);
				if (i < tvGradient.getSize()[0] - 1)
					fijr = img.getAtIndex(i + 1, j);
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < tvGradient.getSize()[1] - 1)
					fijd = img.getAtIndex(i, j + 1);
				if (i > 0 & j < tvGradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < tvGradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
				double vij = (2* fij - fijr - fijd)
						/ Math.sqrt(eps + (fij - fijr) * (fij - fijr)+ (fij - fijd) * (fij - fijd))
						+(fij - fiju)
						/ Math.sqrt(eps + (fijru - fiju) * (fijru - fiju)+ (fij - fiju) * (fij - fiju))
						+  (fij - fijl)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fijl - fijld)*(fijl - fijld));
				if (Math.abs(vij)>maxValue)
					maxValue=Math.abs(vij);
				tvGradient.setAtIndex(i, j, (float) vij);

			}
		}
		return tvGradient;
	}

	/**
	 * compute weighted TV gradient
	 * @param img
	 * @return
	 */
	public Grid2D computewTVGradient(Grid2D img) {//weighted TV gradient
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		double vij;
		double wr,wd;
		this.maxValue=0;
		for (int i = 0; i < img.getSize()[0]; i++) {
			for (int j = 0; j < img.getSize()[1]; j++) {
				double fij = img.getAtIndex(i, j);

				double fijl = fij;
				double fijr = fij;
				double fiju = fij;
				double fijd = fij;
				double fijld = fij;
				double fijru = fij;
				//Not at border
				if (i > 0)
					fijl = img.getAtIndex(i - 1, j);
				if (i < tvGradient.getSize()[0] - 1){
					fijr = img.getAtIndex(i + 1, j);
				wr=this.weightMatrix.getAtIndex(i+1, j);
				}
				else
					wr=0.0f;
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < tvGradient.getSize()[1] - 1){
					fijd = img.getAtIndex(i, j + 1);
				wd=this.weightMatrix.getAtIndex(i, j+1);}
				else
					wd=0.0;
				if (i > 0 & j < tvGradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < tvGradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
			
				vij = this.weightMatrix.getAtIndex(i, j)*(4 * fij - 2 * fijl - 2 * fiju)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fij - fiju) * (fij - fiju))
						-wr* 2* (fijr - fij)
						/ Math.sqrt(eps + (fijr - fij) * (fijr - fij)+ (fijr - fijru) * (fijr - fijru))
						-wd* 2* (fijd - fij)
						/ Math.sqrt(eps + (fijd - fij) * (fijd - fij)+ (fijd - fijld)*(fijd - fijld));
				
				if (Math.abs(vij)>maxValue)
					maxValue=Math.abs(vij);
				tvGradient.setAtIndex(i, j, (float) vij);

			}
		}
		return tvGradient;
	}
	
	//*****************************************************************************************************use OpenCL
	/**
	 * compute image gradient use OpenCL
	 * @param imgCL
	 */
	public void computeImageGradientCL(OpenCLGrid2D imgCL){
		//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
		tvOperators.computeImageGradient2D(imgCL, imgGradientCL);
	}
	
	/**
	 * weight matrix update with OpenCL
	 * @param imgCL
	 */
	public void weightMatrixCLUpdate(OpenCLGrid2D imgCL){		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdate2D(imgCL, weightMatrixCL, weps);
	}

	/**
	 * 
	 * @param imgCL
	 * @return
	 */
	public double getWeightedTVvalue(OpenCLGrid2D imgCL)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		
		this.computeImageGradientCL(imgCL);
		weightMatrixCL.getGridOperator().multiplyBy(imgGradientCL, weightMatrixCL);
		wTV=imgGradientCL.getGridOperator().sum(imgGradientCL);
		return wTV;
	}
	
	/**
	 * compute weighted TV gradient
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid2D computeWeightedTVGradient(OpenCLGrid2D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradient2D(imgCL, weightMatrixCL, TVGradientCL);
	
		return this.TVGradientCL;
	}
	
	/**
	 * compute anisotropic image gradient along X direction
	 * @param imgCL
	 */
	public void computeImageGradientCLX(OpenCLGrid2D imgCL){
		//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
		tvOperators.computeImageGradient2DX(imgCL, imgGradientCL);
	}
	
	/**
	 * update weight matrix in anisotropic wTV along X direction
	 * @param imgCL
	 */
	public void weightMatrixCLUpdateX(OpenCLGrid2D imgCL){		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().computeWeightMatrixUpdate2DX(imgCL, weightMatrixCL, weps);
	}

	/**
	 * get weighted TV value in anisotropic wTV along X direction
	 * @param imgCL
	 * @return
	 */
	public double getWeightedTVvalueX(OpenCLGrid2D imgCL)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		
		this.computeImageGradientCLX(imgCL);
		weightMatrixCL.getGridOperator().multiplyBy(imgGradientCL, weightMatrixCL);
		wTV=imgGradientCL.getGridOperator().sum(imgGradientCL);
		return wTV;
	}
	
	/**
	 * compute weighted TV gradient in anisotropic TV along X direction
	 * @param imgCL
	 * @return
	 */
	public OpenCLGrid2D computeWeightedTVGradientX(OpenCLGrid2D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.computeWeightedTVGradient2DX(imgCL, weightMatrixCL, TVGradientCL);
	
		return this.TVGradientCL;
	}
	
	/**
	 * get the max value in the TV gradient with OpenCL
	 * @return
	 */
	public double getMaxTVGradientCLValue()
	{ maxValue=0;
	for(int i=0;i<TVGradientCL.getSize()[0];i++)
		for(int j=0;j<TVGradientCL.getSize()[1];j++)
			if(Math.abs(TVGradientCL.getAtIndex(i, j))>maxValue)
				maxValue=Math.abs(TVGradientCL.getAtIndex(i, j));
	 
	 return maxValue;
 }
	
	/**
	 * Xiaolin's 2 level weighted TV 
	 */
	public void weightMatrixUpdateL2(){//2Level 
		int Masize=tvGradient.getSize()[0]*tvGradient.getSize()[1];
			Grid1D tempGradient=new Grid1D(Masize);
		 
		int i,k;//the minimun value index in tagArray
		float temp;
		int[] index=new int[Masize];

		for( i=0;i<Masize;i++){
			index[i]=i;
			tempGradient.setAtIndex(i, imageGradient.getAtIndex(i/tvGradient.getSize()[1],i%tvGradient.getSize()[1]));
		}
		initialWeightMatrix();
		for( i=0;i<numK;i++){
			for(int j=i+1;j<Masize;j++){
			if(tempGradient.getAtIndex(j)>tempGradient.getAtIndex(i))
			{
				temp=tempGradient.getAtIndex(j);
				tempGradient.setAtIndex(j, tempGradient.getAtIndex(i));
				tempGradient.setAtIndex(i, temp);
				k=index[i];
				index[i]=index[j];
				index[j]=k;
			}
			}
			weightMatrix.setAtIndex(index[i]/tvGradient.getSize()[1], index[i]%tvGradient.getSize()[1], rou);
		}
		}
	
	/**
	 * compute image gradient in anisotropic wTV along X direction
	 */
	public void computeImageGradientX(Grid2D img){//Compute the gradient of the img

		double Hdiff,Vdiff;
		for (int i = 0; i < imageGradient.getSize()[0]; i++) {
			for (int j = 0; j < imageGradient.getSize()[1]; j++) {
				double fij = img.getAtIndex(i, j);
				double fl = fij,fl2=fij,fr=fij;
				double fiju = fij;
				if (i > 2&& i<imageGradient.getSize()[0]-2){
					fl = img.getAtIndex(i - 1, j);
					fl2=img.getAtIndex(i-2, j);
					fr=img.getAtIndex(i+1, j);
				}
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				Hdiff=a*fr+b*fij-b*fl-a*fl2;
				Vdiff=fij-fiju;
				this.imageGradient.setAtIndex(i, j, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff));
			}
		}
	}
	
	/**
	 * update the weight matrix in the anisotropic wTV along X
	 * @param img
	 */
	public void weightedMatrixUpdateX(Grid2D img){//Update the weights for weighted TV
		this.computeImageGradientX(img);
		gradientTemp=(Grid2D)this.imageGradient.clone();
		NumericPointwiseOperators.addBy(gradientTemp, (float)this.weps);
		weightMatrix=(Grid2D)NumericPointwiseOperators.dividedBy(onesTemp,gradientTemp );	
	}
	
	/**
	 * get weighted TV in anisotropic wTV along X
	 * @param img
	 * @return
	 */
	public double getWeightedTVvalueX(Grid2D img)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		this.computeImageGradientX(img);
		wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(weightMatrix, imageGradient));
		return wTV;
	}

	
	
	
}

