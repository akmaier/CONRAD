package edu.stanford.rsl.conrad.io;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import org.junit.Test;

import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.FileUtil;
import ij.IJ;
import ij.io.FileInfo;

/**
 * Class to read Dennerlein's Format. It's basically raw float data with a 6 byte header containing x,y, and z.
 * 
 * @author akmaier
 *
 */
public class DennerleinProjectionSource extends FileProjectionSource {

	protected boolean debug = false;
	
	@Override
	public void initStream (String filename) throws IOException {
		fi = getHeaderInfo(filename);
		init();
	}

	/**
	 * Reads the header information from the file into a fileinfo object
	 * @param filename the filename
	 * @return the FileInfo
	 * @throws IOException
	 */
	public FileInfo getHeaderInfo( String filename ) throws IOException {
		if (IJ.debugMode) CONRAD.log("Entering Nrrd_Reader.readHeader():");
		FileInfo fi = new FileInfo();
		File file = new File(filename);
		fi.fileName=file.getName();
		fi.directory = file.getParent() + "/";
		// NB Need RAF in order to ensure that we know file offset
		RandomAccessFile input = new RandomAccessFile(fi.directory+fi.fileName,"r");
		fi.fileType = FileInfo.GRAY8;  // just assume this for the mo    
		fi.fileFormat = FileInfo.RAW;
		fi.nImages = 1;

		byte [] header = new byte [6];
		input.read(header);
		fi.width = (int) convertToUnsignedShort(header, 0);
		fi.height = (int) convertToUnsignedShort(header, 1);
		fi.nImages = (int) convertToUnsignedShort(header, 2);

		CONRAD.log("Dennerlein Reading image with " + fi.nImages + " frames with " + fi.width + "x" + fi.height + " resolution");
		fi.compression = FileInfo.COMPRESSION_NONE;

		fi.fileType=FileInfo.GRAY32_FLOAT;
		// exception for projection matrix data
		if (fi.width == 3 && fi.height == 4){
			fi.width = 4;
			fi.height = 3;
			fi.fileType=FileInfo.GRAY64_FLOAT;
		}
		
		if (fi.width == 4 && fi.height == 3){
			fi.fileType=FileInfo.GRAY64_FLOAT;
		}
		fi.offset = 6;
		fi.intelByteOrder = true;

		input.close();		
		return (fi);
	}

	@Test
	public void testProjectionSource(){
		// Test to read a large Dennerlein file. Result should be that the memory consumption does not increase.
		// Which is the case in the current implementation.
		// akmaier
		try {
			String filenameString = FileUtil.myFileChoose(".bin", false);
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
