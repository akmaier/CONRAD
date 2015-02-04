package edu.stanford.rsl.conrad.utils;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import ij.ImagePlus;
import ij.ImageStack;

/**
 * Class to buffer ImageProcessors which were processed in a parallel manner.
 * Can also be used to sort the ImageProcessors efficiently.
 * @author akmaier
 *
 */
public class ImageGridBuffer {

	ArrayList<Grid2D> images;
	ArrayList<Integer> indices;
	private boolean debug = false;

	/**
	 * creates a new ImageProcessorBuffer object.
	 */
	public ImageGridBuffer (){
		images = new ArrayList<Grid2D>();
		indices = new ArrayList<Integer>();
	}

	public synchronized void set(Grid3D grid){
		for (int i = 0; i < grid.getSize()[2]; i++){
			add(grid.getSubGrid(i), i);
		}
	}
	
	/**
	 * adds the Image at index i; Previous entry will be overwritten.
	 * @param image
	 * @param index
	 */
	public synchronized void add(Grid2D image, int index){
		Integer key = new Integer(index);
		if (!indices.contains(key)){ // Insert
			if (debug) System.out.println("ImageProcessorBuffer: Duplicate Index replacing " + key);
			images.add(image);
			indices.add(key);
		} else { // Overwrite
			for (int i = 0; i < indices.size(); i++){
				if (indices.get(i).equals(key)){
					if (debug) System.out.println("ImageProcessorBuffer: Duplicate Index replacing " + key);
					images.set(i, image);
					break;
				}
			}
		}
	}

	/**
	 * Returns the internal index of the given external index.
	 * It is used to access the arrayLists.
	 * 
	 * @param externalIndex the external index position
	 * @return the position in the internal ArrayList.
	 */
	private synchronized int internalIndex(int externalIndex){
		int internalIndex = -1;
		if (debug) System.out.println("ImageProcessorBuffer: Requested Index " + externalIndex + "; current size = " +indices.size());
		for (int i = 0; i < indices.size(); i++){
			if (indices.get(i) != null) {
				if (indices.get(i).intValue() == externalIndex){
					internalIndex = i;
					if (debug) System.out.println("ImageProcessorBuffer: Index delivered " + externalIndex + "; current size = " +indices.size());
					break;
				}
			} else {
				System.out.println("ImageProcessorBuffer internal error.");
				System.exit(-1);
			}
		}
		return internalIndex;
	}

	/**
	 * Returns the ImageProcessor at index index.
	 * Returns null if the index is not found.
	 * @param index the index
	 * @return the ImageProcessor
	 */
	public synchronized Grid2D get(int index){
		Grid2D revan = null;
		int pos = internalIndex(index);
		if (pos != -1) revan = images.get(pos);
		return revan;
	}

	/**
	 * Removes the image at the given index.
	 * @param index the index.
	 */
	public synchronized void remove (int index){
		int pos = internalIndex(index);
		images.set(pos, null);
	}

	/**
	 * Returns a sorted array of the buffered ImageProcessors 
	 * @return the array.
	 */
	public synchronized Grid2D [] toArray(){
		if (debug) System.out.println("ImageProcessorBuffer: Creating array with length " + images.size());
		Grid2D[] array = new Grid2D[images.size()];
		for (int i = 0; i< array.length; i++){
			int index = indices.get(i).intValue();
			array[index] = images.get(i);
		}
		return array;
	}
	
	/**
	 * Returns a sorted ImagePlus of the buffered ImageProcessors 
	 * @return the ImagePlus.
	 */
	public synchronized ImagePlus toImagePlus(String title){
		if (debug) System.out.println("ImageProcessorBuffer: Creating ImagePlus with length " + images.size());
		ImagePlus image = new ImagePlus();
		ImageStack stack = new ImageStack(images.get(0).getWidth(), images.get(0).getHeight(), images.size());
		stack.setColorModel(ImageUtil.getDefaultColorModel());
		for (int i = 0; i< images.size(); i++){
			int index = indices.get(i).intValue();
			stack.setPixels(images.get(i).getBuffer(), index + 1);
		}
		image.setStack(title, stack);
		return image;
	}

	/**
	 * returns the number of stored ImageProcessors
	 * @return the number
	 */
	public synchronized int size(){
		return images.size();
	}



}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/