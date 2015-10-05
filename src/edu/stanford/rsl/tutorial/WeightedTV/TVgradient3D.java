package edu.stanford.rsl.tutorial.WeightedTV;

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


public class TVgradient3D {
	public float eps = 0.1f;
	public float weps=0.001f;
	public Grid3D gradient;
	public Grid3D TVgradient;
	public double max_Value=0.0;
	public Grid3D Wmatrix;//weights for weighted TV

	//OpenCL
	public OpenCLGrid3D imgGradientCL;
	public OpenCLGrid3D TVgradientCL;
	public OpenCLGrid3D WmatrixCL;
	TVOpenCLGridOperators TVOperators;
	/*public TVgradient3D(Grid3D img)
{
TVgradient=new Grid3D(img);
NumericPointwiseOperators.fill(TVgradient, 0);
}*/

public TVgradient3D(OpenCLGrid3D imgCL)
{this.TVgradientCL=new OpenCLGrid3D(imgCL);
TVgradientCL.getGridOperator().fill(TVgradientCL, 0);
WmatrixCL=new OpenCLGrid3D(TVgradientCL);
TVOperators=new TVOpenCLGridOperators();	
}

public TVgradient3D(int[] size)
{this.TVgradientCL=new OpenCLGrid3D(new Grid3D(size[0],size[1],size[2]));
TVgradientCL.getGridOperator().fill(TVgradientCL, 0);
WmatrixCL=new OpenCLGrid3D(TVgradientCL);
TVOperators=new TVOpenCLGridOperators();	
}

public void initialWmatrix(){
	Wmatrix=new Grid3D(TVgradient);
	NumericPointwiseOperators.fill(Wmatrix, 1.0f);
}
public void initialWmatrixCL(){
	
	WmatrixCL.getGridOperator().fill(WmatrixCL, 1.0f);
}


public void ComputeGradient(Grid3D img){//Compute the gradient of the img
	this.gradient=new Grid3D(img);
	double Hdiff,Vdiff,Zdiff;
	for (int i = 0; i < gradient.getSize()[0]; i++) {
		for (int j = 0; j < gradient.getSize()[1]; j++) {
			for(int k=0;k<gradient.getSize()[2];k++){
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
			this.gradient.setAtIndex(i, j,k, (float)Math.sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff));
			}
		}
	}
}
public void ComputeGradientCL(OpenCLGrid3D imgCL){
	//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
	this.imgGradientCL=new OpenCLGrid3D(TVgradientCL);
	TVOperators.compute_img_gradient(imgCL, imgGradientCL);
}



public void WmatrixUpdate(){//Update the weights for weighted TV
	Grid3D Ones_temp=new Grid3D(Wmatrix);
	NumericPointwiseOperators.fill(Ones_temp, 1.0f);
	Grid3D gradient_temp=new Grid3D(this.gradient);
	NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
	Wmatrix=(Grid3D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
}

public void WmatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update(imgCL, WmatrixCL, weps);
}

public void AdaptiveWmatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL,this.TVgradientCL);
	TVOpenCLGridOperators.getInstance().compute_adaptive_Wmatrix_Update(TVgradientCL, this.WmatrixCL, weps);
}

public void AwTVWmatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_AwTV_Wmatrix_Update(imgCL, WmatrixCL, weps);
}
public void WmatrixCLUpdate2(OpenCLGrid3D imgCL){//do TV in each Z slide
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update2(imgCL, WmatrixCL, weps);
}

public double getTVvalue()//Compute ComputeGradient(Grid2D img)
{
	double TV=0.0;
	/*for (int i = 0; i < gradient.getSize()[0]; i++) 
		for (int j = 0; j < gradient.getSize()[1]; j++) 
			for(int k=0;k<gradient.getSize()[2];k++)
			TV+=gradient.getAtIndex(i, j,k);*/
	TV=NumericPointwiseOperators.sum(gradient);
	return TV;
}

public double getwTVvalue()//Compute ComputeGradient(Grid2D img)
{
	double wTV=0.0;
	/*for (int i = 0; i < gradient.getSize()[0]; i++) 
		for (int j = 0; j < gradient.getSize()[1]; j++)
			for(int k=0;k<gradient.getSize()[2];k++)
			wTV+=this.Wmatrix.getAtIndex(i, j,k)*this.gradient.getAtIndex(i, j,k);*/
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(gradient, Wmatrix));

	return wTV;
}

public double getwTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	TVOperators.getwTV(imgCL, WmatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getwTVvalueCL_adaptive(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	TVOperators.getwTV_adaptive(imgCL, WmatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getAwTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	TVOperators.getAwTV(imgCL, WmatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getwTVvalueCL2(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	TVOperators.getwTV2(imgCL, WmatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}
/*
public double getwTVvalueCL(OpenCLGrid3D imgCL)
{   this.ComputeGradientCL(imgCL);
	OpenCLGrid3D wTVTempGrid=new OpenCLGrid3D(this.imgGradientCL);
	wTVTempGrid.getGridOperator().multiplyBy(wTVTempGrid, WmatrixCL);
	double wTV=wTVTempGrid.getGridOperator().sum(wTVTempGrid);
	wTVTempGrid.release();
	return wTV;
}
*/

	public Grid3D ComputeTVgradient(Grid3D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.max_Value=0.0f;
		for (int i = 0; i < TVgradient.getSize()[0]; i++) {
			for (int j = 0; j < TVgradient.getSize()[1]; j++) {
				for(int k=0;k<gradient.getSize()[2];k++){
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
				if (i < TVgradient.getSize()[0] - 1)
					fr = img.getAtIndex(i + 1, j,k);
				if (j > 0)
					fu = img.getAtIndex(i, j - 1,k);
				if (j < TVgradient.getSize()[1] - 1)
					fd = img.getAtIndex(i, j + 1,k);
				if(k>0)
					ft=img.getAtIndex(i, j, k-1);
				if(k<TVgradient.getSize()[2]-1)
					fb=img.getAtIndex(i, j, k+1);
				if (i > 0 & j < TVgradient.getSize()[1] - 1)
					fld = img.getAtIndex(i - 1, j + 1,k);
				if (i < TVgradient.getSize()[0] - 1 & j > 0)
					fru = img.getAtIndex(i + 1, j - 1,k);
				if(k>0 & i<TVgradient.getSize()[0]-1)
					frt=img.getAtIndex(i+1, j, k-1);
				if(i>0 & k<TVgradient.getSize()[2]-1)
					flb=img.getAtIndex(i-1, j, k+1);
				if(j<(TVgradient.getSize()[1] - 1) & k>0)
					fdt=img.getAtIndex(i, j+1, k-1);				
				if(k<(TVgradient.getSize()[2] - 1) & j>0)
					fub=img.getAtIndex(i, j-1, k+1);
				
				double vij = (3 * fijk -  fl - fu-ft)
						/ Math.sqrt(eps + (fijk- fl) * (fijk - fl)+ (fijk - fu) * (fijk - fu)+(fijk-ft)*(fijk-ft))
						-  (fr - fijk)
						/ Math.sqrt(eps + (fr - fijk) * (fr - fijk)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						- (fd - fijk)
						/ Math.sqrt(eps + (fd - fijk) * (fd - fijk)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-(fb-fijk)
						/Math.sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fijk)*(fb-fijk));
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				TVgradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return TVgradient;
	}
	

	public Grid3D ComputewTVgradient(Grid3D img) {//weighted TV gradient
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		double vij;
		double wr,wd,wb;
		for (int i = 0; i < TVgradient.getSize()[0]; i++) {
			for (int j = 0; j < TVgradient.getSize()[1]; j++) {
				for(int k=0;k<gradient.getSize()[2];k++){
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
				if (i < TVgradient.getSize()[0] - 1)
				{
					fr = img.getAtIndex(i + 1, j,k);
					wr=Wmatrix.getAtIndex(i+1, j, k);}
				else
					wr=0;
				if (j > 0)
					fu = img.getAtIndex(i, j - 1,k);
				if (j < TVgradient.getSize()[1] - 1){
					fd = img.getAtIndex(i, j + 1,k);
					wd=Wmatrix.getAtIndex(i, j+1, k);}
				else
					wd=0;
				if(k>0)
					ft=img.getAtIndex(i, j, k-1);
				if(k<TVgradient.getSize()[2]-1){
					fb=img.getAtIndex(i, j, k+1);
					wb=Wmatrix.getAtIndex(i, j, k+1);
				}
				else
					wb=0;
				if (i > 0 & j < TVgradient.getSize()[1] - 1)
					fld = img.getAtIndex(i - 1, j + 1,k);
				if (i < TVgradient.getSize()[0] - 1 & j > 0)
					fru = img.getAtIndex(i + 1, j - 1,k);
				if(k>0 & i<TVgradient.getSize()[0]-1)
					frt=img.getAtIndex(i+1, j, k-1);
				if(i>0 & k<TVgradient.getSize()[2]-1)
					flb=img.getAtIndex(i-1, j, k+1);
				if(j<(TVgradient.getSize()[1] - 1) & k>0)
					fdt=img.getAtIndex(i, j+1, k-1);				
				if(k<(TVgradient.getSize()[2] - 1) & j>0)
					fub=img.getAtIndex(i, j-1, k+1);
				//Not at border           
				vij = Wmatrix.getAtIndex(i, j, k)*(3 * fijk -  fl - fu-ft)
						/ Math.sqrt(eps + (fijk- fl) * (fijk - fl)+ (fijk - fu) * (fijk - fu)+(fijk-ft)*(fijk-ft))
						- wr* (fr - fijk)
						/ Math.sqrt(eps + (fr - fijk) * (fr - fijk)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						-wd* (fd - fijk)
						/ Math.sqrt(eps + (fd - fijk) * (fd - fijk)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-wb*(fb-fijk)
						/Math.sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fijk)*(fb-fijk));
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				TVgradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return TVgradient;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	public OpenCLGrid3D compute_wTV_adaptive_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_adaptive_Gradient(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	
	public OpenCLGrid3D compute_AwTV_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_AwTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient2(OpenCLGrid3D imgCL)//do TV in each Z slide
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient2(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
	
	public void WmatrixCLUpdate_Y(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update_Y(imgCL, WmatrixCL, weps);
	}
	
	public double getwTVvalueCL_Y(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		TVOperators.getwTV_Y(imgCL, WmatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient_Y(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient_Y(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}

	
	public void WmatrixCLUpdate_X(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update_X(imgCL, WmatrixCL, weps);
	}
	
	public double getwTVvalueCL_X(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		TVOperators.getwTV_Y(imgCL, WmatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient_X(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		TVOperators.compute_wTV_Gradient_X(imgCL, WmatrixCL, TVgradientCL);
	
		return this.TVgradientCL;
	}
}

