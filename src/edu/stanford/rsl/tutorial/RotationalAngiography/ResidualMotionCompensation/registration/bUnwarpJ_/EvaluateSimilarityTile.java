package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.registration.bUnwarpJ_;

import java.awt.Rectangle;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class EvaluateSimilarityTile implements Runnable 
{

	/** float epsilon */
	private final double FLT_EPSILON = (double)Float.intBitsToFloat((int)0x33FFFFFF);
	/** current target image */
	final BSplineModel auxTarget;
	/** current source image */
	final BSplineModel auxSource;
	/** target mask */
	final Mask auxTargetMsk;
	/** source mask */
	final Mask auxSourceMsk;
	/** B-spline deformation in x */
	final BSplineModel swx;
	/** B-spline deformation in y */
	final BSplineModel swy;
	/** factor width */
	final double auxFactorWidth;
	/** factor height */
	final double auxFactorHeight;
	/** number of intervals between B-spline coefficients */
	final int intervals;
	/** similarity gradient */
	final double[] grad;
	/** evaluation results: image similarity value for the current rectangle and number of pixels that have been evaluated */
	final double[] result;
	/** rectangle containing the area of the image to be evaluated */
	final Rectangle rect;
	
	final double imageWeight = 2;
	final boolean PYRAMID = true;
	/**
	 * Evaluate similarity tile constructor 
	 * 
	 * @param auxTarget current target image
	 * @param auxSource current source image
	 * @param auxTargetMsk target mask
	 * @param auxSourceMsk source mask
	 * @param swx B-spline deformation in x
	 * @param swy B-spline deformation in y
	 * @param auxFactorWidth factor width
	 * @param auxFactorHeight factor height
	 * @param intervals number of intervals between B-spline coefficients
	 * @param grad similarity gradient (output)
	 * @param result output results: image similarity value for the current rectangle and number of pixels that have been evaluated
	 * @param rect rectangle containing the area of the image to be evaluated
	 */
	EvaluateSimilarityTile(final BSplineModel auxTarget,
						   final BSplineModel auxSource,
						   final Mask auxTargetMsk,
						   final Mask auxSourceMsk,
						   final BSplineModel swx,
						   final BSplineModel swy,
						   final double auxFactorWidth,
						   final double auxFactorHeight,
						   final int intervals,
						   final double[] grad,
						   final double[] result,
						   final Rectangle rect)
	{
		this.auxTarget = auxTarget;
		this.auxSource = auxSource;

		this.auxTargetMsk = auxTargetMsk;
		this.auxSourceMsk = auxSourceMsk;
		
		this.swx = swx;
		this.swy = swy;
		
		this.auxFactorWidth = auxFactorWidth;
		this.auxFactorHeight = auxFactorHeight;
				
		this.intervals = intervals;
		
		this.grad = grad;
		
		this.result = result;
	
		this.rect = rect;
	}

	//------------------------------------------------------------------
	/**
	 * Run method to evaluate the similarity of source and target images. 
	 * Only the part defined by the rectangle will be evaluated.
	 */
	public void runNCC() 
	{
		final int cYdim = intervals+3;
		final int cXdim = cYdim;
		final int Nk = cYdim * cXdim;
		final int twiceNk = 2 * Nk;
		
		// The rectangle marks the area of the image to be treated.
		int uv = rect.y * rect.width + rect.x;
		final int Ydim = rect.y + rect.height;
		final int Xdim = rect.x + rect.width;	
		
		int n = 0;		
		double sourceMean = 0.0, targetMean = 0.0;
		final double []I1D = new double[2]; // Space for the first derivatives of I1

		final double []targetCurrentImage = auxTarget.getCurrentImage();
		
		if(rect.height < 0 || rect.width < 0){
			double crossCorr = 0.0f;
			// Set result (image similarity value for the current rectangle
			// and number of pixels that have been evaluated)
			this.result[0] = crossCorr;		
			this.result[1] = n;			
			return;		 
		}
		
		double[][] valuesSource = new double[rect.height][rect.width];
		double[][] valuesTarget = new double[rect.height][rect.width];
		
		for (int v=rect.y; v<Ydim; v++){										
			for (int u=rect.x; u<Xdim; u++, uv++){
				// Check if this point is in the target mask
				if (auxTargetMsk.getValue(u/auxFactorWidth, v/auxFactorHeight)){
					// Compute value in the source image
					final double I2 = targetCurrentImage[uv];

					// Compute the position of this point in the target
					double x = swx.precomputed_interpolateI(u,v);
					double y = swy.precomputed_interpolateI(u,v);

					// Check if this point is in the source mask
					if (auxSourceMsk.getValue(x/auxFactorWidth, y/auxFactorHeight))
					{
						// Compute the value of the target at that point
						final double I1 = auxSource.prepareForInterpolationAndInterpolateIAndD(x, y, I1D, false, PYRAMID);							
					
						sourceMean += I2;
						targetMean += I1;
						valuesSource[v-rect.y][u-rect.x] = I2;
						valuesTarget[v-rect.y][u-rect.x] = I1;
						n++;						
					}					
				}					
			}
		}
		
		if(n == 0){
			System.out.println("failed n ");
			this.result[0] = 1/FLT_EPSILON; 
			this.result[1] = n; 
			return;
		}		
		//Compute mean value of source and target rectangle if n != 0
		sourceMean = sourceMean / n;
		targetMean = targetMean / n;
				
		double varianceSource = 0.0, varianceTarget = 0.0;
		n = 0;
		
		// Compute variance for source and target image
		for(int i = 0; i < valuesSource.length; i++){
			for(int j = 0; j < valuesSource[i].length; j++){
				varianceSource += Math.pow((valuesSource[i][j]-sourceMean), 2);
				varianceTarget += Math.pow((valuesTarget[i][j]-targetMean), 2);	
				n++;
			}
		}
		varianceSource = (n==0)?1:varianceSource / n;
		varianceTarget = (n==0)?1:varianceTarget / n;
		
		// Compute standard deviation for source and target image
		double stdDevSource = Math.sqrt(varianceSource);
		double stdDevTarget = Math.sqrt(varianceTarget);
		
		double temp = 0;
		double crossCorr = 0;
		double nccTemp = 0;
		n = 0;
		
		//reset to starting value
		uv = rect.y * rect.width + rect.x; 
		
		// Pre-computing the denominator
		double denom = 1.0f/(stdDevSource * stdDevTarget);	
		denom = (Double.isInfinite(denom))?1:denom;
		
		for (int v=rect.y; v<Ydim; v++){										
			for (int u=rect.x; u<Xdim; u++, uv++){				
				// Check if this point is in the target mask
				if (auxTargetMsk.getValue(u/auxFactorWidth, v/auxFactorHeight))
				{
					// Compute value in the source image
					final double I2 = targetCurrentImage[uv];

					// Compute the position of this point in the target
					double x = swx.precomputed_interpolateI(u,v);
					double y = swy.precomputed_interpolateI(u,v);

					// Check if this point is in the source mask
					if (auxSourceMsk.getValue(x/auxFactorWidth, y/auxFactorHeight))
					{
						// Compute the value of the target at that point
						final double I1 = auxSource.prepareForInterpolationAndInterpolateIAndD(x, y, I1D, false, PYRAMID);							
						final double I1dx = I1D[0], I1dy = I1D[1];		
						
						temp = (I2 - sourceMean) * (I1 - targetMean);
						//temp = Math.pow(temp, 2);
						nccTemp = -temp*denom;
						crossCorr += nccTemp;
						
						// Compute the derivative with respect to all the c coefficients
						// Cost of the derivatives = 16*(3 mults + 2 sums)
						// Current cost = 359 mults + 346 sums
						for (int l=0; l<4; l++)
							for (int m=0; m<4; m++)
							{
								if (swx.prec_yIndex[v][l]==-1 || swx.prec_xIndex[u][m]==-1) continue;

								// Note: It's the same to take the indexes and weightI from swx than from swy
								double weightI = swx.precomputed_getWeightI(l,m,u,v);

								int k = swx.prec_yIndex[v][l] * cYdim + swx.prec_xIndex[u][m];

								// Compute partial result
								// There's also a multiplication by 2 that I will
								// do later								
								double aux = -nccTemp* weightI;

								// Derivative related to X deformation
								grad[k]   += aux * I1dx;

								// Derivative related to Y deformation
								grad[k+Nk]+= aux * I1dy;
							}
						n++; // Another point has been successfully evaluated
					}
				}
			}
		}
		// Average the image related terms (now i do the 1/n outside)
		if(Double.isNaN(crossCorr)){
			System.out.println();
		}
		
		if (n!=0)
		{
			crossCorr *= imageWeight;
			double aux = imageWeight * 2.0; // This is the 2 coming from the
											   // derivative that I would do later
			for (int k=0; k<twiceNk; k++) 
				grad[k] *= aux;
		} 
		else{
			crossCorr = 1/FLT_EPSILON;
		}
		// Set result (image similarity value for the current rectangle
		// and number of pixels that have been evaluated)
		this.result[0] = crossCorr;		
		this.result[1] = n;
	} // end run method

 // end class EvaluateSimilarityTile

	public void run() 
	{	
		final int cYdim = intervals+3;
		final int cXdim = cYdim;
		final int Nk = cYdim * cXdim;
		final int twiceNk = 2 * Nk;
		
		double imageSimilarity = 0.0;
		
		// The rectangle marks the area of the image to be treated.
		int uv = rect.y * rect.width + rect.x;
		final int Ydim = rect.y + rect.height;
		final int Xdim = rect.x + rect.width;						
		
		// Loop over all points in the source image (rectangle)
		int n = 0;
		
		//System.out.println("Thread: " + Thread.currentThread().getName() + " - rect.Height: " + rect.height +  " - rect.width: " + rect.width);
		if(rect.height < 0 || rect.width < 0){
			System.out.println("check");
		}
		
		final double []I1D = new double[2]; // Space for the first derivatives of I1

		final double []targetCurrentImage = auxTarget.getCurrentImage();

		for (int v=rect.y; v<Ydim; v++)
		{										
			for (int u=rect.x; u<Xdim; u++, uv++) 
			{
				// Compute image term .....................................................

				// Check if this point is in the target mask
				if (auxTargetMsk.getValue(u/auxFactorWidth, v/auxFactorHeight))
				{
					// Compute value in the source image
					final double I2 = targetCurrentImage[uv];

					// Compute the position of this point in the target
					double x = swx.precomputed_interpolateI(u,v);
					double y = swy.precomputed_interpolateI(u,v);

					// Check if this point is in the source mask
					if (auxSourceMsk.getValue(x/auxFactorWidth, y/auxFactorHeight))
					{
						// Compute the value of the target at that point
						final double I1 = auxSource.prepareForInterpolationAndInterpolateIAndD(x, y, I1D, false, PYRAMID);							

						final double I1dx = I1D[0], I1dy = I1D[1];

						final double error = I2 - I1;
						final double error2 = error*error;							
						imageSimilarity += error2;
						
						

						// Compute the derivative with respect to all the c coefficients
						// Cost of the derivatives = 16*(3 mults + 2 sums)
						// Current cost = 359 mults + 346 sums
						for (int l=0; l<4; l++)
							for (int m=0; m<4; m++)
							{
								if (swx.prec_yIndex[v][l]==-1 || swx.prec_xIndex[u][m]==-1) continue;

								// Note: It's the same to take the indexes and weightI from swx than from swy
								double weightI = swx.precomputed_getWeightI(l,m,u,v);

								int k = swx.prec_yIndex[v][l] * cYdim + swx.prec_xIndex[u][m];

								// Compute partial result
								// There's also a multiplication by 2 that I will
								// do later
								double aux = -error * weightI;

								// Derivative related to X deformation
								grad[k]   += aux * I1dx;

								// Derivative related to Y deformation
								grad[k+Nk]+= aux * I1dy;
							}
						n++; // Another point has been successfully evaluated
					}
				}
			}
		}
		
		// Average the image related terms (now i do the 1/n outside)
		if (n!=0)
		{
			imageSimilarity *= imageWeight;
			double aux = imageWeight * 2.0; // This is the 2 coming from the
											   // derivative that I would do later
			for (int k=0; k<twiceNk; k++) 
				grad[k] *= aux;
		} 
		else
			imageSimilarity = 1/FLT_EPSILON;

		// Set result (image similarity value for the current rectangle
		// and number of pixels that have been evaluated)
		this.result[0] = imageSimilarity;		
		this.result[1] = n;			
		//System.out.println("rect.y: " + rect.y + " - rect.x: " + rect.x + " || Ydim: " + Ydim + " - Xdim: " + Xdim + " || ssdSimilarity: " + imageSimilarity );

	} // end run method 
}
 // end class EvaluateSimilarityTile

/*
for(int i = 0; i < valuesSource.length; i++){
	for(int j = 0; j < valuesSource[i].length; j++){
		temp = (valuesSource[i][j]-sourceMean)*(valuesTarget[i][j]-targetMean);
		crossCorr += temp*denom;
		
		final double I1dx = I1D[0], I1dy = I1D[1];
		
		for (int l=0; l<4; l++)
			for (int m=0; m<4; m++)
			{
				if (swx.prec_yIndex[i][l]==-1 || swx.prec_xIndex[j][m]==-1) continue;

				// Note: It's the same to take the indexes and weightI from swx than from swy
				double weightI = swx.precomputed_getWeightI(l,m,j,i);

				int k = swx.prec_yIndex[i][l] * cYdim + swx.prec_xIndex[j][m];

				// Compute partial result
				// There's also a multiplication by 2 that I will
				// do later
				double aux = -crossCorr * weightI;

				// Derivative related to X deformation
				grad[k]   += aux * I1dx;

				// Derivative related to Y deformation
				grad[k+Nk]+= aux * I1dy;
			}
		n++; // Another point has been successfully evaluated
	}
}*/
