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
	public Grid3D gradient;
	public Grid3D tvGradient;
	public double max_Value=0.0;
	public Grid3D wMatrix;//weights for weighted TV

	//OpenCL
	public OpenCLGrid3D imgGradientCL;
	public OpenCLGrid3D tvGradientCL;
	public OpenCLGrid3D wMatrixCL;
	TVOpenCLGridOperators tvOperators;
	/*public TVgradient3D(Grid3D img)
{
TVgradient=new Grid3D(img);
NumericPointwiseOperators.fill(TVgradient, 0);
}*/

public TVGradient3D(OpenCLGrid3D imgCL)
{this.tvGradientCL=new OpenCLGrid3D(imgCL);
tvGradientCL.getGridOperator().fill(tvGradientCL, 0);
wMatrixCL=new OpenCLGrid3D(tvGradientCL);
tvOperators=new TVOpenCLGridOperators();	
}

public TVGradient3D(int[] size)
{this.tvGradientCL=new OpenCLGrid3D(new Grid3D(size[0],size[1],size[2]));
tvGradientCL.getGridOperator().fill(tvGradientCL, 0);
wMatrixCL=new OpenCLGrid3D(tvGradientCL);
tvOperators=new TVOpenCLGridOperators();	
}

public void initialWmatrix(){
	wMatrix=new Grid3D(tvGradient);
	NumericPointwiseOperators.fill(wMatrix, 1.0f);
}
public void initialWmatrixCL(){
	
	wMatrixCL.getGridOperator().fill(wMatrixCL, 1.0f);
}


public void computeGradient(Grid3D img){//Compute the gradient of the img
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
public void computeGradientCL(OpenCLGrid3D imgCL){
	//TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL, this.imgGradientCL);
	this.imgGradientCL=new OpenCLGrid3D(tvGradientCL);
	tvOperators.compute_img_gradient(imgCL, imgGradientCL);
}



public void wMatrixUpdate(){//Update the weights for weighted TV
	Grid3D Ones_temp=new Grid3D(wMatrix);
	NumericPointwiseOperators.fill(Ones_temp, 1.0f);
	Grid3D gradient_temp=new Grid3D(this.gradient);
	NumericPointwiseOperators.addBy(gradient_temp, (float)this.weps);
	wMatrix=(Grid3D)NumericPointwiseOperators.dividedBy(Ones_temp,gradient_temp );	
}

public void wMatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update(imgCL, wMatrixCL, weps);
}

public void adaptiveWmatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_img_gradient(imgCL,this.tvGradientCL);
	TVOpenCLGridOperators.getInstance().compute_adaptive_Wmatrix_Update(tvGradientCL, this.wMatrixCL, weps);
}

public void awTVWmatrixCLUpdate(OpenCLGrid3D imgCL){
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_AwTV_Wmatrix_Update(imgCL, wMatrixCL, weps);
}
public void wMatrixCLUpdate2(OpenCLGrid3D imgCL){//do TV in each Z slide
	
	//this.ComputeGradientCL(imgCL);
	TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update2(imgCL, wMatrixCL, weps);
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
	wTV=NumericPointwiseOperators.sum(NumericPointwiseOperators.multipliedBy(gradient, wMatrix));

	return wTV;
}

public double getwTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getwTV(imgCL, wMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getwTVvalueCL_adaptive(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getwTV_adaptive(imgCL, wMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getAwTVvalueCL(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getAwTV(imgCL, wMatrixCL,tempZSum);
	double wTV=tempZSum.getGridOperator().sum(tempZSum);
   tempZSum.release();
	return wTV;
}

public double getwTVvalueCL2(OpenCLGrid3D imgCL)
{   
	OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
	tempZSum.getGridOperator().fill(tempZSum, 0);
	tvOperators.getwTV2(imgCL, wMatrixCL,tempZSum);
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

	public Grid3D computeTVgradient(Grid3D img) {
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		this.max_Value=0.0f;
		for (int i = 0; i < tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
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
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				tvGradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return tvGradient;
	}
	

	public Grid3D computewTVgradient(Grid3D img) {//weighted TV gradient
		//According to the paper:
		//Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT
		double vij;
		double wr,wd,wb;
		for (int i = 0; i < tvGradient.getSize()[0]; i++) {
			for (int j = 0; j < tvGradient.getSize()[1]; j++) {
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
				if (i < tvGradient.getSize()[0] - 1)
				{
					fr = img.getAtIndex(i + 1, j,k);
					wr=wMatrix.getAtIndex(i+1, j, k);}
				else
					wr=0;
				if (j > 0)
					fu = img.getAtIndex(i, j - 1,k);
				if (j < tvGradient.getSize()[1] - 1){
					fd = img.getAtIndex(i, j + 1,k);
					wd=wMatrix.getAtIndex(i, j+1, k);}
				else
					wd=0;
				if(k>0)
					ft=img.getAtIndex(i, j, k-1);
				if(k<tvGradient.getSize()[2]-1){
					fb=img.getAtIndex(i, j, k+1);
					wb=wMatrix.getAtIndex(i, j, k+1);
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
				vij = wMatrix.getAtIndex(i, j, k)*(3 * fijk -  fl - fu-ft)
						/ Math.sqrt(eps + (fijk- fl) * (fijk - fl)+ (fijk - fu) * (fijk - fu)+(fijk-ft)*(fijk-ft))
						- wr* (fr - fijk)
						/ Math.sqrt(eps + (fr - fijk) * (fr - fijk)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						-wd* (fd - fijk)
						/ Math.sqrt(eps + (fd - fijk) * (fd - fijk)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-wb*(fb-fijk)
						/Math.sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fijk)*(fb-fijk));
				if (Math.abs(vij)>max_Value)
					max_Value=Math.abs(vij);
				tvGradient.setAtIndex(i, j,k, (float) vij);
				}
			}
		}
		return tvGradient;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_wTV_Gradient(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	public OpenCLGrid3D compute_wTV_adaptive_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_wTV_adaptive_Gradient(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	public OpenCLGrid3D compute_AwTV_Gradient(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_AwTV_Gradient(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient2(OpenCLGrid3D imgCL)//do TV in each Z slide
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_wTV_Gradient2(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
	
	public void wMatrixCLUpdate_Y(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update_Y(imgCL, wMatrixCL, weps);
	}
	
	public double getwTVvalueCL_Y(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		tvOperators.getwTV_Y(imgCL, wMatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient_Y(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_wTV_Gradient_Y(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}

	
	public void wMatrixCLUpdate_X(OpenCLGrid3D imgCL){//mainly along Y
		
		//this.ComputeGradientCL(imgCL);
		TVOpenCLGridOperators.getInstance().compute_Wmatrix_Update_X(imgCL, wMatrixCL, weps);
	}
	
	public double getwTVvalueCL_X(OpenCLGrid3D imgCL)
	{   
		OpenCLGrid3D tempZSum=new OpenCLGrid3D(new Grid3D(imgCL.getSize()[0],imgCL.getSize()[1],1));
		tempZSum.getGridOperator().fill(tempZSum, 0);
		tvOperators.getwTV_Y(imgCL, wMatrixCL,tempZSum);
		double wTV=tempZSum.getGridOperator().sum(tempZSum);
	   tempZSum.release();
		return wTV;
	}
	
	public OpenCLGrid3D compute_wTV_Gradient_X(OpenCLGrid3D imgCL)
	{
		//TVOpenCLGridOperators.getInstance().compute_wTV_Gradient(imgCL, WmatrixCL, TVgradientCL);
		tvOperators.compute_wTV_Gradient_X(imgCL, wMatrixCL, tvGradientCL);
	
		return this.tvGradientCL;
	}
}

