package edu.stanford.rsl.conrad.io;
// FlexibleFileOpener
// ------------------
// Class to allow plugins ImageJ to semi-transparently access
// compressed (GZIP, ZLIB) raw image data.
// 
// - It can add a GZIPInputStream or ZInputStream onto the
// stream provided to File opener
// - Can also specify a pre-offset to jump to before FileOpener sees the 
// stream.  This allows one to read compressed blocks from a file that
// has not been completely compressed. 
//
// NB GZIP is not the same as ZLIB
// GZIP has a longer header; the compression algorithm is identical

// (c) Gregory Jefferis 2007
// Department of Zoology, University of Cambridge
// jefferis@gmail.com
// All rights reserved
// Source code released under Lesser Gnu Public License v2

import ij.IJ;
import ij.io.FileInfo;
import ij.io.FileOpener;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.util.zip.GZIPInputStream;

import com.jcraft.jzlib.ZInputStream;

public class FlexibleFileOpener extends FileOpener {
	
	public static final int UNCOMPRESSED = 0;
	public static final int GZIP = 1;
	public static final int ZLIB = 2;
	
	int gunzipMode=UNCOMPRESSED;
	// the offset that will be skipped before FileOpener sees the stream
	long preOffset=0; 
	
	public FlexibleFileOpener(FileInfo fi) {
		this(fi,fi.fileName.toLowerCase().endsWith(".gz")?GZIP:UNCOMPRESSED,0);
	}
	public FlexibleFileOpener(FileInfo fi, int gunzipMode) {
		this(fi,gunzipMode,0);
	}
	
	public FlexibleFileOpener(FileInfo fi, int gunzipMode, long preOffset) {
		super(fi);
		this.gunzipMode=gunzipMode;
		this.preOffset=preOffset;
	}
	
	public InputStream createInputStream(FileInfo fi) throws IOException, MalformedURLException {
		// use the method in the FileOpener class to generate an input stream
		InputStream is=super.createInputStream(fi);
	
		// Skip if required
		if (preOffset!=0) is.skip(preOffset);

		// Just return original input stream if uncompressed
		if (gunzipMode==UNCOMPRESSED) return is;

		//  else put a regular GZIPInputStream on top 
		/** BEGIN legacy code, kept for reference
		* NB should only do this if less than 138s because that will take care of things automatically.
		* if(gunzipMode==GZIP){
		*	boolean lessThan138s = IJ.getVersion().compareTo("1.38s")<0;
		*	if(lessThan138s) return new GZIPInputStream(is);
		*	else return is;
		* }
		* END of legacy code.
		* 2018-09-19:  	According to
		* 				http://www.cas.miamioh.edu/~meicenrd/anatomy/Ch14_IndependentInvestigation/ImageJ/ij-docs/ij-docs/notes.html
		* 				version 1.38s and beyond should handle gzip automatically.
		* 				However, the library class {@link ij.io.FileOpener} only recognizes files of .gz and .gzip ending as GZIP compressed.
		* 				This is not the case for .nrrd files, which have encoding information in their header.
		* 				Consequently, a GZIPInputStream is needed here regardless of version.
		*/
		if(gunzipMode==GZIP) return new GZIPInputStream(is,50000);
		
		// or put a ZInputStream on top (from jzlib)
		if(gunzipMode==ZLIB) return new ZInputStream(is);
		
		// fallback
		throw new IOException("Incorrect GZIP mode: "+gunzipMode);
	}
}