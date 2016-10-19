package edu.stanford.rsl.tutorial.praktikum;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.ImageJ;


public class Phantom extends Grid2D{

	public Phantom(int width, int height) {
		super(width, height);
		// TODO Auto-generated constructor stub
		
		//create square with intensity 0.2
		int edgeLength = width/8;
		for (int row =  edgeLength ; row < width/2 +  edgeLength ; row++) {
			for (int col =  edgeLength ; col < height/2 +  edgeLength ; col++) {
				this.setAtIndex(row, col, (0.2f));
			}
		}
		
		//create circle with intensity 0.5
		int xCenter = width/2 + width/8;
		int yCenter = height/2 + height/4;
		int radius = width/10;
		for (int row = xCenter - radius; row < xCenter + radius; row++){
			for (int col = yCenter - radius; col < yCenter + radius; col++){
				if (((row-xCenter)*(row-xCenter)) + ((col-yCenter)*(col-yCenter)) <= radius * radius){
					this.setAtIndex(row, col, (.5f));
				}
			}
		}
		
		//create triangle with intensity 0.7
		int count = 0;
		for (int row = 2 * height/8; row < height/2; row++){
			for (int col = 4 * width/8; col < width; col++){
				if(col >= (6 * width/8) - count && col <= (6 * width/8) + count){
					this.setAtIndex(col, row, (.7f));
				}
			}
			count++;
		}
	}
	
	public static void main(String[] args){
		ImageJ ui = new ImageJ();
		Phantom a = new Phantom(256, 256);
		
		ui.createImage(256,256);
		
		a.show();
	}

}
