/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;


import ij.IJ;
import ij.io.FileInfo;
import ij.io.TiffDecoder;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;


/**
 * Wrapper for a zip reader using imageJ. 
 * Do not use this file format in combination with tiff files, if you have big files. 
 * The tiff decoder wraps the stream into a RandomAccessStream which copies everything for later access. Use of this file format will double the memory usage for reading!
 * @author akmaier
 *
 */
public class ZipProjectionSource extends FileProjectionSource {

	boolean DICOMMultiFrameMode = false;
	ZipInputStream zis = null;
	
	public void initStream (String filename) throws IOException{
		zis = new ZipInputStream(new FileInputStream(filename));
		is = zis;
		if (zis==null) {
			throw new IOException("Could not read zip header");
		}
		ZipEntry entry = zis.getNextEntry();
		if (entry==null) throw new IOException("Zip was empty");
		String name = entry.getName();
		if (name.endsWith(".tif")) {
			TiffDecoder td = new TiffDecoder(zis, name);
			if (IJ.debugMode) td.enableDebugging();
			FileInfo [] info = td.getTiffInfo();
			fi = info[0];
		} else if (name.endsWith(".ima")) {
			DicomProjectionSource dicomSource = new DicomProjectionSource();
			dicomSource.initStream(zis);
			fi = dicomSource.fi;
		} else if (name.endsWith(".dcm")) {
			DICOMMultiFrameMode = true;
			DicomProjectionSource dicomSource = new DicomProjectionSource();
			dicomSource.initStream(zis);
			fi = dicomSource.fi;
		} else {
			zis.close();
			throw new IOException("No matching fileformat found in zip file");
		}


		File f = new File(filename);
		fi.fileFormat = FileInfo.ZIP_ARCHIVE;
		fi.fileName = f.getName();
		fi.directory = f.getParent()+File.separator;

		if (fi!=null && fi.width>0 && fi.height>0 && fi.offset>0) {
			init();
		} else {
			throw new IOException("Format does not match");
		}
		zis.close();
		zis = new ZipInputStream(new FileInputStream(filename));
		is = zis;
		if(!DICOMMultiFrameMode) zis.getNextEntry();
	}	

	@Override
	public Grid2D getNextProjection(){
		if (!DICOMMultiFrameMode){
			return super.getNextProjection();
		} else {
			try {
				ZipEntry entry = zis.getNextEntry();
				if (entry == null){
					zis.close();
					return null;
				}
				DicomProjectionSource dicomSource = new DicomProjectionSource();
				dicomSource.initStream(zis);
				fi = dicomSource.fi;
				Grid2D ip = dicomSource.getNextProjection();
				currentIndex++;
				if (showProgress){
					IJ.showProgress((0.0 + currentIndex) / fi.nImages);
				}
				return ip;
			} catch (IOException e1) {
				e1.printStackTrace();
				throw new RuntimeException("Error while reading projection " + this.currentIndex);
			}
		}
	}
	
}
