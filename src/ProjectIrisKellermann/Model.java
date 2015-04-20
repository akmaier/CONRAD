package ProjectIrisKellermann;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.ImageJ;

public class Model extends Grid2D {
	
	public Model(int width, int height)
	{
		super(width, height);
		
		DrawCircle(width/2, height/2, width/3, 1.0);
		
		
		
	}
	
	private void DrawCircle(int x, int y, int radius, double intensity)
	{
		for(int i = 0; i < this.getWidth(); ++i)
		{
			for(int j = 0; j < this.getHeight(); j++)
			{ 
				if((i-x)*(i-x) + (j-y)*(j-y) < (radius * radius))
				{
					putPixelValue(i,j,intensity);
				}
			}
		}
	}
}
