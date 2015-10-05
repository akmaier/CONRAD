package edu.stanford.rsl.tutorial.WeightedTV;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;



public class TVgradient {
	public double eps = 0.1;
	public float weps=0.001f;
	public float thresMax=0.1f;
	public float thresMin=0.01f;
	public Grid2D gradient;
	public Grid2D TVgradient;
	public double max_Value=0.0;
	public Grid2D Wmatrix;//weights for weighted TV
	public Grid2D ATVweightH,ATVweightV;
	//Parameters for 2 level weighted TV
	public int K=1400;
	public float rou=0.00001f;
	
	public OpenCLGrid2D imgGradientCL;
	public OpenCLGrid2D TVgradientCL;
	public OpenCLGrid2D WmatrixCL;
	TVOpenCLGridOperators TVOperators;
	
	//AwTV
	public float a=1.0f;
	public float b=1.0f;
	
	private Grid2D Ones_temp;
	private Grid2D gradient_temp;
public TVgradient(Grid2D img)
{
TVgradient=new Grid2D(img);
this.gradient=new Grid2D(img);
NumericPointwiseOperators.fill(TVgradient, 0);
Ones_temp=new Grid2D(img);
NumericPointwiseOperators.fill(Ones_temp, 1.0f);
gradient_temp=new Grid2D(img);
}

public TVgradient(OpenCLGrid2D imgCL){
	this.TVgradientCL=new OpenCLGrid2D(imgCL);
	TVgradientCL.getGridOperator().fill(TVgradientCL, 0);
	imgGradientCL=new OpenCLGrid2D(imgCL);
	TVgradientCL.getGridOperator().fill(imgGradientCL, 0);
	TVOperators=new TVOpenCLGridOperators();
}

public void initialWmatrix(){
	Wmatrix=new Grid2D(TVgradient);
	NumericPointwiseOperators.fill(Wmatrix, 1.0f);
}

public void initialWmatrixCL(){
	WmatrixCL=new OpenCLGrid2D(TVgradientCL);
	WmatrixCL.getGridOperator().fill(WmatrixCL, 1.0f);
}


public void ComputeGradient(Grid2D img){//Compute the gradient of the img

	double Hdiff,Vdiff;
	for (int i = 0; i < gradient.getSize()[0]; i++) {
		for (int j = 0; j < gradient.getSize()[1]; j++) {
			double fij = img.getAtIndex(i, j);
			double fijl = fij;
			double fiju = fij;
			if (i > 0)
				fijl = img.getAtIndex(i - 1, j);
			if (j > 0)
				fiju = img.getAtIndex(i, j - 1);
			Hdiff=fij-fijl;
			Vdiff=fij-fiju;
			this.gradient.setAtIndex(i, j, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff));
		}
	}
}



public void WmatrixUpdate(Grid2D img){//Update the weights for weighted TV
	this.ComputeGradient(img);
	gradient_temp=(Grid2D)this.gradient.clone();
	NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
	Wmatrix=(Grid2D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
}
public void WmatrixUpdate(){//Update the weights for weighted TV
	gradient_temp=(Grid2D)this.gradient.clone();
	NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
	Wmatrix=(Grid2D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
}

public void WmatrixUpdate2(Grid2D img){//Update the weights for weighted TV
	this.ComputeGradient(img);
	Grid2D gradient_temp=new Grid2D(this.gradient);
	gradient_temp=SetThreshold(gradient_temp,0.01f);
	Wmatrix=(Grid2D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
}

private Grid2D SetThreshold(Grid2D imgGradient,float thres){
	for(int i=0;i<imgGradient.getSize()[0];i++)
		for(int j=0;j<imgGradient.getSize()[1];j++){
			if(imgGradient.getAtIndex(i, j)<thres)
				imgGradient.setAtIndex(i, j, 1);
			
		}
	return imgGradient;
}


public double getTVvalue(Grid2D img)//Compute ComputeGradient(Grid2D img)
{
	double TV=0.0;

	this.ComputeGradient(img);
	TV=NumericPointwiseOperators.sum(this.gradient);
	
	return TV;
}

public double getwTVvalue(Grid2D img)//Compute ComputeGradient(Grid2D img)
{
	double wTV=0.0;
	this.ComputeGradient(img);
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(Wmatrix, gradient));
	return wTV;
}


public double getwTVvalue()//Compute ComputeGradient(Grid2D img)
{
	double wTV=0.0;
	
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(Wmatrix, gradient));
	return wTV;
}

public Grid2D ComputeTVgradient(Grid2D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.max_Value=0.0f;
		for (int i = 0; i < TVgradient.getSize()[0]; i++) {
			for (int j = 0; j < TVgradient.getSize()[1]; j++) {
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
				if (i < TVgradient.getSize()[0] - 1)
					fijr = img.getAtIndex(i + 1, j);
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < TVgradient.getSize()[1] - 1)
					fijd = img.getAtIndex(i, j + 1);
				if (i > 0 & j < TVgradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < TVgradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
				double vij = (4 * fij - 2 * fijl - 2 * fiju)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fij - fiju) * (fij - fiju))
						- 2* (fijr - fij)
						/ Math.sqrt(eps + (fijr - fij) * (fijr - fij)+ (fijr - fijru) * (fijr - fijru))
						- 2* (fijd - fij)
						/ Math.sqrt(eps + (fijd - fij) * (fijd - fij)+ (fijd - fijld)*(fijd - fijld));
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				TVgradient.setAtIndex(i, j, (float) vij);

			}
		}
		return TVgradient;
	}
	
	
	public Grid2D ComputeTVgradient2(Grid2D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.max_Value=0;
		for (int i = 0; i< TVgradient.getSize()[0]; i++) {
			for (int j = 0; j < TVgradient.getSize()[1]; j++) {
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
				if (i < TVgradient.getSize()[0] - 1)
					fijr = img.getAtIndex(i + 1, j);
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < TVgradient.getSize()[1] - 1)
					fijd = img.getAtIndex(i, j + 1);
				if (i > 0 & j < TVgradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < TVgradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
				double vij = (2* fij - fijr - fijd)
						/ Math.sqrt(eps + (fij - fijr) * (fij - fijr)+ (fij - fijd) * (fij - fijd))
						+(fij - fiju)
						/ Math.sqrt(eps + (fijru - fiju) * (fijru - fiju)+ (fij - fiju) * (fij - fiju))
						+  (fij - fijl)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fijl - fijld)*(fijl - fijld));
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				TVgradient.setAtIndex(i, j, (float) vij);

			}
		}
		return TVgradient;
	}

	public Grid2D ComputewTVgradient(Grid2D img) {//weighted TV gradient
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		double vij;
		double wr,wd;
		this.max_Value=0;
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
				if (i < TVgradient.getSize()[0] - 1){
					fijr = img.getAtIndex(i + 1, j);
				wr=this.Wmatrix.getAtIndex(i+1, j);
				}
				else
					wr=0.0f;
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				if (j < TVgradient.getSize()[1] - 1){
					fijd = img.getAtIndex(i, j + 1);
				wd=this.Wmatrix.getAtIndex(i, j+1);}
				else
					wd=0.0;
				if (i > 0 & j < TVgradient.getSize()[1] - 1)
					fijld = img.getAtIndex(i - 1, j + 1);
				if (i < TVgradient.getSize()[0] - 1 & j > 0)
					fijru = img.getAtIndex(i + 1, j - 1);
				
			
				vij = this.Wmatrix.getAtIndex(i, j)*(4 * fij - 2 * fijl - 2 * fiju)
						/ Math.sqrt(eps + (fij - fijl) * (fij - fijl)+ (fij - fiju) * (fij - fiju))
						-wr* 2* (fijr - fij)
						/ Math.sqrt(eps + (fijr - fij) * (fijr - fij)+ (fijr - fijru) * (fijr - fijru))
						-wd* 2* (fijd - fij)
						/ Math.sqrt(eps + (fijd - fij) * (fijd - fij)+ (fijd - fijld)*(fijd - fijld));
				
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				TVgradient.setAtIndex(i, j, (float) vij);

			}
		}
		return TVgradient;
	}
	
	//*****************************************************************************************************use OpenCL
	public void ComputeGradientCL(OpenCLGrid2D imgCL){
		//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
		TVOperators.compute_img_gradient2D(imgCL, imgGradientCL);
	}
	
	public void WmatrixCLUpdate(OpenCLGrid2D imgCL){		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update2D(imgCL, WmatrixCL, weps);
	}

	public double getwTVvalue(OpenCLGrid2D imgCL)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		
		this.ComputeGradientCL(imgCL);
		WmatrixCL.getGridOperator().multiplyBy(imgGradientCL, WmatrixCL);
		wTV=imgGradientCL.getGridOperator().sum(imgGradientCL);
		return wTV;
	}
	
	public OpenCLGrid2D compute_wTV_Gradient(OpenCLGrid2D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient2D(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	
	
	public void ComputeGradientCL_X(OpenCLGrid2D imgCL){
		//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
		TVOperators.compute_img_gradient2D_X(imgCL, imgGradientCL);
	}
	
	public void WmatrixCLUpdate_X(OpenCLGrid2D imgCL){		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update2D_X(imgCL, WmatrixCL, weps);
	}

	public double getwTVvalue_X(OpenCLGrid2D imgCL)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		
		this.ComputeGradientCL_X(imgCL);
		WmatrixCL.getGridOperator().multiplyBy(imgGradientCL, WmatrixCL);
		wTV=imgGradientCL.getGridOperator().sum(imgGradientCL);
		return wTV;
	}
	
	public OpenCLGrid2D compute_wTV_Gradient_X(OpenCLGrid2D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient2D_X(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	
	
	public double getMaxTVgradientCLValue()
	{ max_Value=0;
	for(int i=0;i<TVgradientCL.getSize()[0];i++)
		for(int j=0;j<TVgradientCL.getSize()[1];j++)
			if(Math.abs(TVgradientCL.getAtIndex(i, j))>max_Value)
				max_Value=Math.abs(TVgradientCL.getAtIndex(i, j));
	 
	 return max_Value;
 }
	
	public void L2WmatrixUpdate(){//2Level 
		int Masize=TVgradient.getSize()[0]*TVgradient.getSize()[1];
			Grid1D tempGradient=new Grid1D(Masize);
		 
		int i,k;//the minimun value index in tagArray
		float temp;
		int[] index=new int[Masize];

		for( i=0;i<Masize;i++){
			index[i]=i;
			tempGradient.setAtIndex(i, gradient.getAtIndex(i/TVgradient.getSize()[1],i%TVgradient.getSize()[1]));
		}
		initialWmatrix();
		for( i=0;i<K;i++){
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
			Wmatrix.setAtIndex(index[i]/TVgradient.getSize()[1], index[i]%TVgradient.getSize()[1], rou);
		}
		}
	
	
	public void ComputeGradient_X(Grid2D img){//Compute the gradient of the img

		double Hdiff,Vdiff;
		for (int i = 0; i < gradient.getSize()[0]; i++) {
			for (int j = 0; j < gradient.getSize()[1]; j++) {
				double fij = img.getAtIndex(i, j);
				double fl = fij,fl2=fij,fr=fij;
				double fiju = fij;
				if (i > 2&& i<gradient.getSize()[0]-2){
					fl = img.getAtIndex(i - 1, j);
					fl2=img.getAtIndex(i-2, j);
					fr=img.getAtIndex(i+1, j);
				}
				if (j > 0)
					fiju = img.getAtIndex(i, j - 1);
				Hdiff=a*fr+b*fij-b*fl-a*fl2;
				Vdiff=fij-fiju;
				this.gradient.setAtIndex(i, j, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff));
			}
		}
	}
	
	public void WmatrixUpdate_X(Grid2D img){//Update the weights for weighted TV
		this.ComputeGradient_X(img);
		gradient_temp=(Grid2D)this.gradient.clone();
		NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
		Wmatrix=(Grid2D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
	}
	
	public double getwTVvalue_X(Grid2D img)//Compute ComputeGradient(Grid2D img)
	{
		double wTV=0.0;
		this.ComputeGradient_X(img);
		wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(Wmatrix, gradient));
		return wTV;
	}

	
	
	
}

