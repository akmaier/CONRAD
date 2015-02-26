package TruncationPolynomial;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;
import ij.ImageJ;

public class TruncationCorrection {
	
	private static Grid2D getTruncatedSinogram(Grid2D sinogram, int truncationWidth) {
		
		Grid2D truncatedSinogram = new Grid2D(sinogram);
		
		for (int row = 0; row < sinogram.getHeight(); row++) {
			for (int col = 0; col < truncationWidth; col++) {
				
				truncatedSinogram.setAtIndex(col, row, 0.0f);
				truncatedSinogram.setAtIndex(truncatedSinogram.getWidth()-1-col, row, 0.0f);
				
			}
		}

		
		return truncatedSinogram;
	}
	
	public static void main(String[] args) {

		new ImageJ();
		
		int imageSize = 200;
		double deltaS = 200.0 / 201.0;
		
		int numTheta = 360;
		int truncationSize = 40;
		
		UniformCircleGrid2D cylinder = new UniformCircleGrid2D(imageSize, imageSize);
		ParallelProjector2D projector2d = new ParallelProjector2D(Math.PI, Math.PI/(numTheta-1), imageSize, deltaS);
		Grid2D sinogram = projector2d.projectRayDrivenCL(cylinder);
		sinogram.show("original sinogram");
		
		Grid2D truncatedSinogram = getTruncatedSinogram(sinogram, truncationSize);
		truncatedSinogram.show("Truncated Sinogram");

		ParallelBackprojector2D backprojector = new ParallelBackprojector2D(imageSize, imageSize, 1, 1);
		Grid2D backproj = backprojector.backprojectRayDriven(sinogram);
		backproj.show("original backprojected");
		
		Grid2D truncBackproj = backprojector.backprojectRayDriven(truncatedSinogram);
		truncBackproj.show("truncated backprojected");
		
		double[] l_double = new double[sinogram.getHeight()];
		double[] l = new double[sinogram.getHeight()];
		int[] thetaPrimes = new int[sinogram.getHeight()];
		int[] ses = new int[sinogram.getHeight()];
		for (int theta = 0; theta < numTheta; theta++) {
			
			double maximum = 0.0f;
			int curr_thetaPrime = 0;
			int curr_s = 0;
			for (int thetaPrime = theta-90; thetaPrime < theta+90; thetaPrime++) {
				if (thetaPrime < 0) {
					thetaPrime +=360;
					for (int s = 0;  s < sinogram.getWidth(); s++) {
						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime));
						if (val > maximum) {
							maximum = val;
							curr_thetaPrime = thetaPrime;
							curr_s = s;
						}
					}
					thetaPrime -= 360;
				} else if (thetaPrime >= 360) {
					thetaPrime -=360;
					for (int s = 0;  s < sinogram.getWidth(); s++) {
						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime));
						if (val > maximum) {
							maximum = val;
							curr_thetaPrime = thetaPrime;
							curr_s = s;
						}
					}
					thetaPrime += 360;
				} else {
					for (int s = 0;  s < sinogram.getWidth(); s++) {
						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime));
						if (val > maximum) {
							maximum = val;
							curr_thetaPrime = thetaPrime;
							curr_s = s;
						}
					}
				}
				
			}
			l_double[theta] = maximum;
			l[theta] = (int) Math.round(maximum);
			thetaPrimes[theta] = curr_thetaPrime;
			ses[theta] = curr_s;
		}
		
		System.out.println(0);
			
		
		
	}
	

}
