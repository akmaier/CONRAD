package edu.stanford.rsl.conrad.io;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.junit.Test;

import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.DicomDecoder;
import edu.stanford.rsl.conrad.utils.FileUtil;



public class DicomProjectionSource extends FileProjectionSource {

	public void initStream (String filename) throws IOException{
		if (!(filename.toLowerCase().endsWith("dcm") || filename.toLowerCase().endsWith("ima"))) throw new RuntimeException("DicomProjectionSource:Not a dicom file!");
		File file = new File(filename);
		String fileName = file.getName();
		if (fileName==null)
			return;
		BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file));
		initStream(inputStream);
	}
	
	public void initStream(InputStream inputStream) throws IOException{
		DicomDecoder dd = new DicomDecoder(inputStream);
		fi = dd.getFileInfo();
		if (fi!=null && fi.width>0 && fi.height>0 && fi.offset>0) {
			init();
		} else {
			throw new IOException("Format does not match");
		}
	}
	
	@Test
	public void testProjectionSource(){
		// Test to read a large DICOM File. Result should be that the memory consumption does not increase.
		// Which is the case in the current implementation.
		// akmaier
		try {
			String filenameString = FileUtil.myFileChoose(".IMA", false);
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
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/