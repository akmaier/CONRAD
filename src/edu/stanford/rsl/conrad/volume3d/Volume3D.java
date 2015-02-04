package edu.stanford.rsl.conrad.volume3d;


import ij.ImagePlus;
import ij.ImageStack;
import edu.stanford.rsl.conrad.utils.ImageUtil;


/**
 * 3D-FFTable Volume based on C-code from Lars Wigstroem. This volume class is able to describe a 3D grid with uniform spacing in each spatial dimension. One sub class of this volume is CUDAVolume3D which models the same data in CUDA memory. 
 * The corresponding operator is then CUDAVolumeOperator. Note that instantiation should be performed using the respective VolumeOperator in order to enable compatibility to CUDA.
 * 
 * @author akmaier
 * @see edu.stanford.rsl.conrad.cuda.CUDAVolume3D
 * @see edu.stanford.rsl.conrad.volume3d.VolumeOperator
 */
public class Volume3D {

	public int     dimensions;        /* #dimensions               */
	public int[]   size  = new int [MAX_DIM];     /* #points in each dimension */
	public float[] spacing = new float [MAX_DIM];      /* size of points [m, s, ..] */
	public int   in_dim;            /* inner dimension, i.e. complex or real  */ 
	public float [] [] [] data;        /* buffer holding the data   */
	/* points at first element   */

	public final static int MAX_DIM = 3;
	protected final static boolean DEBUG_FLAG = true;

/**
 * returns 1 if the volume is real and 2 if it is complex
 * @return the internal dimension of the volume.
 */
	public int getInternalDimension(){
		return in_dim;
	}
	
	/**
	 * Creates an empty volume. Use of this constructor is generally discouraged. It may result in incompatibility with CUDA.
	 * @param size2 sizes in each direction
	 * @param dim2 physical dimension in each direction
	 * @param inDim internal Dimension
	 */
	public Volume3D(int[] size2, float[] dim2, int inDim) {
		/* set attributes */

		this.dimensions = 3;

		for (int dim_loop=0; dim_loop<this.dimensions; dim_loop++) {
			this.size[dim_loop] = size2[dim_loop];
			this.spacing[dim_loop]  = dim2[dim_loop];
		}
		this.in_dim = inDim;

		/* allocate memory for data buffer */
		data = new float [size[0]][size[1]][size[2]*in_dim];		

	}
	
	/**
	 * Creates a Volume3D Object which supports 3-D filtering operations. Call of this constructor is discouraged. Use the VolumeOperator instead.
	 * 
	 * @param image the ImagePlus
	 * @param mirror size of the area which is mirrored to reduce FFT artifacts.
	 * @param cuty number of pixels to be cut from the original volume along y direction
	 * @param uneven is set if the original volume has an uneven number of slices / projections.
	 */
	public Volume3D(ImagePlus image, int mirror, int cuty, boolean uneven){
		dimensions = 3;
		in_dim = 1;
		spacing = new float [3];
		spacing[2] = (float) image.getCalibration().pixelWidth;
		spacing[1] = (float) image.getCalibration().pixelHeight;
		spacing[0] = (float) image.getCalibration().pixelDepth;
		size = new int[3];
		int width = image.getWidth();
		int height = image.getHeight();
		int depth = image.getStackSize();
		size[2]=image.getWidth()+(2*mirror);
		size[1]=image.getHeight()+(2*mirror)-(2*cuty);
		if (!uneven){
			size[0]=image.getStackSize()+(2*mirror);
		} else {
			size[0]=image.getStackSize()+(2*mirror)-1;
		}
		data = new float[size[0]][size[1]][size[2]];
		for (int h = 0; h< depth; h++){
			for (int j = cuty; j< height - cuty; j++){
				for (int i = 0; i< width; i++){
					data[mirror + h][mirror +j-cuty][mirror +i] = ((float [])image.getStack().getPixels(h+1))[(width*j)+i];
				}
				for (int i = 0; i< mirror; i++){
					data[mirror + h][mirror +j-cuty][i] = ((float [])image.getStack().getPixels(h+1))[(width*j)+(mirror)-i-1];
					data[mirror + h][mirror +j-cuty][width + mirror + i] = ((float [])image.getStack().getPixels(h+1))[(width*j)+width-i];
				}
			}
			for (int j = 0; j< mirror; j++){
				for (int i = 0; i < size[2]; i++){
					data[mirror + h][j][i] = data[mirror + h][(2*mirror)-j-1][i];
					data[mirror + h][size[1]-mirror+j][i] = data[mirror+h][size[1]-(mirror)-j-1][i];
				}
			}
		}
		int offset = 0;
		if (uneven) offset = 1;
		for (int h=0;h< mirror; h++){
			for(int j = 0; j < size[1]; j++){
				for (int i = 0; i < size[2]; i++){
					data[h][j][i] = data[(2*mirror)-h-1][j][i];
					if (size[0]-mirror+h-offset < size[0]) {
						data[size[0]-mirror+h-offset][j][i] = data[size[0]-mirror-h-1-offset][j][i];
					}
				}
			}
		}
		//this.getImagePlus("Mirrored").show();
	}
	
	/**
	 * Creates an ImagePlus to visualize the contents of this Volume.
	 * @param title the title of the ImagePlus
	 * @return the ImagePlus
	 */
	public ImagePlus getImagePlus(String title){
		return getImagePlus(title, 0, 0, false);
	}
	
	/**
	 * Method to create an ImagePlus from this Volume3D. Parameters are set to remove the mirrored boundary.
	 * @param title the title for the ImagePlus
	 * @param mirror the width of the mirrored boundary. Should match the parameters used to create the volume.
	 * @param cuty the number of pixels which were cut along the y axis during the creation of the volume.
	 * @param uneven true, if the original number of slices / projections was odd.
	 * @return the ImagePlus
	 */
	public ImagePlus getImagePlus(String title, int mirror, int cuty, boolean uneven){
		ImagePlus revan = new ImagePlus();
		int width = size[2]-(2*mirror);
		int height = size[1]-(2*mirror)+(2*cuty);
		int depth;
		int offset;
		if (uneven) {
			depth = size[0]-(2*mirror)+1;
			offset = 1;
		} else {
			depth = size[0]-(2*mirror);
			offset = 0;
		}
		ImageStack stack = new ImageStack(width, height, depth);
		System.out.println("Image Plus with internal Dimension: " + in_dim);
		for (int h = mirror; h < size[0]-mirror+offset; h++){
			float [] pixels = new float[width*height];
			for (int j = mirror-cuty; j< size[1]-mirror+cuty; j++){
				for (int i = mirror; i< size[2]-mirror; i++){
					pixels[(j-mirror+cuty)*(width)+(i-mirror)] = 
						data[h][j][i*in_dim];
				}
			}
			stack.setPixels(pixels, h-mirror+1);
		}
		stack.setColorModel(ImageUtil.getDefaultColorModel());
		revan.setStack(title, stack);
		return revan;
	}




	/**
	 * prints the dimensions of the volume to STDOUT.
	 */
	public void printSize(){
		for (int i =0; i < dimensions; i++) {
			System.out.print(size[i]+ " ");
		}
		System.out.println("Inner Dimension: " + in_dim);
		if(DEBUG_FLAG) {
			System.out.println(data.length + " " + data[0].length + " " + data[0][0].length);
		}
	}
	

	/**
	 * Releases the memory for this volume. References to data are set to null to improve garbage collection. In the CUDA version, the memory on the card is freed.
	 */
	public void destroy(){
		spacing = null;
		size = null;
		data = null;
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/