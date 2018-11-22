/*
 * Copyright (C) 2010-2018 Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.tools;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;

/**
 * 3-D structuring element.
 * 
 * @author Maximilian Dankbar
 *
 */
public class StructuringElement3D {
	
	public static final String DIAMOND = "diamond";
	private boolean bgWhite = false;
	
	private ImagePlus contents;
	private ObjectWindow object;
	
/*
 * Constructors
 */
	/**
	 * 
	 * Structuring element with arbitrary shape: constructor takes ImagePlus which defines neighborhood with
	 * black and white pixels.
	 *
	 @param im ImagePlus defining the structuring element.  Auto-thresholded so only maximum pixels (255) are
	           seen as white.
	 *
	 @param bgWhite Defines whether the background is white (true) or black (false).
	 *
	 @return A StructElement based on the structure.
	 *
	 */
	public StructuringElement3D(ImagePlus im, boolean bgWhite) {

		setBgWhite(bgWhite);
		try {		
		    // Ensure we have an image with odd width and height.
    		if ( (im.getWidth() % 2 == 0) || (im.getHeight() % 2 == 0) ) {
	    		throw new Exception("Structuring elements must have odd-numbered height and width!");
		}

		// Make our ImagePlus contents an 8-bit image.
    	if(im.isHyperStack()){
    		contents = new ImagePlus("structuring element", im.getStack());
    	//2D
    	} else if(im.isProcessor()){ 
    		contents = new ImagePlus("structuring element", im.getProcessor());
    	} else {
    		throw new IllegalArgumentException("Image has to have a stack or processor");
    	}
		ImageConverter ic = new ImageConverter(contents);
		ic.convertToGray8();

		// Then threshold so only 255 pixels are white.  This 
		// should already be the case, but let's make sure, shall we?
		if(im.isHyperStack()){
			contents.getProcessor().threshold(254);
		} else {
			for(int i = 0; i < contents.getStackSize(); ++i){
				contents.getStack().getProcessor(i).threshold(254);
			}
		}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Construct a structuring element in a given shape. Use on if the class' constants for the type.
	 * @param type Type of the structuring element.
	 * @param radius
	 */
	public StructuringElement3D(String type, int radius){
		if(radius < 0){
			throw new IllegalArgumentException("Radius must not be negative");
		}
		
		if(type == DIAMOND){
			this.contents = makeDiamond3D(radius);
			setBgWhite(false);
		} else {
			throw new IllegalArgumentException("Invalid structuring element type: " + type);
		}
	}
	
	private ImagePlus makeDiamond3D(int radius){
		int size = radius * 2 + 1;
		ImageStack stack = new ImageStack(size, size);
		
		
		for(int i = 0; i < radius; ++i){
			ByteProcessor curSlice = makeDiamond(size, i);
			stack.addSlice(curSlice);
		}
		
		ByteProcessor centralSlice = makeDiamond(size, radius);
		stack.addSlice(centralSlice);
		
		for(int i = radius + 1; i < size; ++i){
			ByteProcessor curSlice = makeDiamond(size, size - i - 1);
			stack.addSlice(curSlice);
		}
		
		return new ImagePlus("structuring element", stack);
	}
	
	private ByteProcessor makeDiamond(int imageSize, int radius){
		ByteProcessor bp = new ByteProcessor(imageSize, imageSize);
		for(int y = imageSize/2 - radius; y <= imageSize/2 + radius; ++y){
			int dX = Math.abs(Math.abs(y - imageSize/2) - radius);
			for(int x = imageSize/2 - dX; x <= imageSize/2 + dX; ++x){
				bp.set(x, y, 255);
			}
		}
		
		if(!bgWhite){
			bp.invert();
		}
		
		return bp;
	}
	
	/**
	 * Set the background to white (true) or black (false).
	 */
	public void setBgWhite(boolean bgWhite) {
		this.bgWhite = bgWhite;
	}
	
	public boolean isBgWhite(){
		return bgWhite;
	}
	
	/**
	 * Store an internal representation of the object image to be operated on by 
	 * this structuring element.
	 *
	 @param im Image to be operated on.
	 @param symmetric Determines if boundary conditions are symmetric or if the edges are padded with background.
	 *
	 */
	public void setObject(ImagePlus im, boolean symmetric) {
		object = new ObjectWindow(im, this, bgWhite, symmetric);
	}
	
	public ImagePlus getContents(){
		return contents;
	}
	
	/**
	 * Return a list of pixel values corresponding to the object image's contents when the structuring element
	 * overlaps the specified coordinates.
	 *
	 */
	public int[] window(int x0, int y0, int z0) {
		return object.view(x0,y0, z0);			
	}
	
	/**
	 *
	 @return The size of the structuring element's foreground in pixels.
	 */
	public int getSize() {
		ImageStack stack = contents.getStack();
		int counter=0;
		
		for(int z = 0; z < contents.getStackSize(); ++z){
			ImageProcessor processor = stack.getProcessor(z + 1);
			int[][] se = processor.getIntArray();
			int sewidth=processor.getWidth(); int seheight=processor.getHeight();
			
			for (int x=0; x<sewidth; x++) {
				for (int y=0; y<seheight; y++) {
					if ((bgWhite && se[x][y]!=255)
							|| (!bgWhite && se[x][y] != 0)) {
						counter += 1;
					}
				}
			}
		}
		return counter;
	}
	
	
	/**
	 * Container for the object image which can return windowed views where the structuring element overlaps.
	 *
	 @author Adam DeConinck
	 @version 0.1
	 */
	class ObjectWindow {

		private int[][][] se;
		private int[][][] object;
		boolean bgWhite;
		private int sewidth, seheight, sedepth, width, height, depth, dx, dy, dz, size;
		boolean symmetric;

		/** 
		 * Construct an ObjectWindow based on the supplied image and structuring element.
		 *
		 @param im ImagePlus containing the object image.
		 @param s Structuring element used to generate the windows.
		 @param bg Background white.
		 @param sym Determines if boundary conditions are symmetric.
		 */
		public ObjectWindow(ImagePlus im, StructuringElement3D s, boolean bg, boolean sym) {
			ImagePlus sContents = s.getContents();
			se = new int[sContents.getStackSize()][][];
			object = new int[im.getStackSize()][][];
			
			ImageStack sStack = sContents.getStack();
			for(int i = 0; i < se.length; ++i){
				se[i] = sStack.getProcessor(i + 1).getIntArray();
			}
			
			ImageStack stack = im.getStack();
			for(int i = 0; i < object.length; ++i){
				object[i] = stack.getProcessor(i + 1).getIntArray();
			}

			bgWhite = bg;

			sewidth=sContents.getWidth(); seheight=sContents.getHeight();
			sedepth = sContents.getStackSize();
			width=im.getWidth(); height=im.getHeight(); setDepth(im.getStackSize());

			dx=sewidth/2; // int division truncates
			dy=seheight/2;
			dz=sedepth/2;

			symmetric = sym;
			size = s.getSize();
		}

		/**
		 * Produce a list of pixel values which are overlapped by the structuring element when centered at a given
		 * coordinate.
		 */
		public int[] view(int x0, int y0, int z0) {

			int[] result = new int[size];
			int k=0;

			for(int z = -dz; z <= dz; ++z){
				for (int x=-dx; x<=dx; x++) {
					for (int y=-dx; y<=dx; y++) {
						// Check if we are in a SE foreground pixel.
						if ((bgWhite && se[z+dz][x+dx][y+dy]!=255) 
								|| !bgWhite && se[z+dz][x+dx][y+dy]!=0) {
							// Coordinates in the object image.
							int xc=x0+x; int yc=y0+y; int zc = z0+z;
							
							// Check z boundary conditions.
							if (zc<0) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									zc=depth+zc;
								}
							} else if (zc >= depth) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									zc=zc-depth;
								}
							}
	
							// Check x boundary conditions.
							if (xc<0) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									xc=width+xc;
								}
							} else if (xc >= width ) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									xc=xc-width;
								}
							}
							
							// Check y boundary conditions.
							if (yc<0) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									yc=height+yc;
								}
							} else if (yc >= height ) {
								if (!(symmetric)) {
									result[k]=bgWhite ? 255 : 1;
									k=k+1;
									continue;
								} else {
									yc=yc-height;
								}
							}
							
							// Add this pixel to the result window.
							result[k]=object[zc][xc][yc];
							k=k+1;
						}
						
					}
				}
			}
			return result;
		}
		
		public void setDepth(int depth){
			this.depth = depth;
		}
	}
}
