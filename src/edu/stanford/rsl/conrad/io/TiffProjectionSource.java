/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import ij.io.TiffDecoder;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.junit.Test;

import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.FileUtil;

/**
 * Wrapper for a tif reader using imageJ. Do not use this file format, if you have big files. The tif decoder wraps the stream into a RandomAccessStream which copies everything for later access. Use of this file format will double the memory usage for reading!
 * @author akmaier
 *
 */
public class TiffProjectionSource extends FileProjectionSource {

	String fileName;
	
	@Override
	public void initStream (String filename) throws IOException{
		if (!(filename.toLowerCase().endsWith("tif") || filename.toLowerCase().endsWith("tiff"))) throw new RuntimeException("TiffProjectionSource:Not a tiff file!");
		File file = new File(filename);
		fileName = file.getName();
		if (fileName==null)
			return;
		BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file));
		initStream(inputStream);
	}

	public void initStream(InputStream inputStream) throws IOException{
		TiffDecoder td = new TiffDecoder(inputStream, fileName);
		fi = td.getTiffInfo()[0];
		if (fi!=null && fi.width>0 && fi.height>0 && fi.offset>0) {
			init();
		} else {
			throw new IOException("Format does not match");
		}
	}
	
	@Test
	public void testProjectionSource(){
		// Test to read a large TIFF File. Result is that the memory access increases twice as much as it should to to ImageJ's RandomAccessStream! 
		// akmaier
		try {
			String filenameString = FileUtil.myFileChoose(".tif", false);
			ProjectionSource test = FileProjectionSource.openProjectionStream(filenameString);
			while(test.getNextProjection() != null){
				System.out.println("Read Projection " + test.getCurrentProjectionNumber());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
