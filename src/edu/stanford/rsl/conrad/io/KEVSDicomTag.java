/*
 * Copyright (C) 2021  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.IOException;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.process.FloatProcessor;

public class KEVSDicomTag extends AdditionalDicomTag {
	public int tags [] = {5443733, 5443734, 5443735, 5443738};
	public static ArrayList<float []> KEVS;
	private boolean debug = false;
	
	public boolean readTag(int tag, int elementLength) throws IOException{
		int id = 0;
		boolean found = false;
		for (int i = 0; i < tags.length; i++)
		{
			if (tag == tags[i]){
				id = i;
				found = true;
				break;
			}
		}
		
		if (found) {
			if (KEVS == null) {
				KEVS = new ArrayList<float []>();
				KEVS.add(null);
				KEVS.add(null);
				KEVS.add(null);
				KEVS.add(null);
			}
			KEVS.set(id, getImage(elementLength));
			if (debug){
				FloatProcessor proc = new FloatProcessor(512,512);
				proc.setPixels(KEVS.get(id));
				ImagePlus img = new ImagePlus();
				img.setProcessor(proc);
				img.show("ID " + id);
			}
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Reads pixeldata from the DICOM field.
	 * @param elementLength the length in bytes
	 * @return the image data in floating point format.
	 * @throws IOException
	 */
	public float [] getImage(int elementLength) throws IOException{
		float [] image = new float [elementLength/2];
		for (int i = 0; i < image.length; i++){
			image[i] = decoder.getShort();
			// Data is stored in signed format, hence we need to fix this here.
			if (image[i] > Short.MAX_VALUE) image[i] -= Short.MAX_VALUE*2;
		}
		return image;
	}
	
	
}
