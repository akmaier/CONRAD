

// (c) Gregory Jefferis 2007
// Department of Zoology, University of Cambridge
// jefferis@gmail.com
// All rights reserved
// Source code released under Lesser Gnu Public License v2

// v0.1 2007-04-02
// - First functional version can write single channel image (stack)
// to raw/gzip encoded monolithic nrrd file
// - Writes key spatial calibration information	including
//   spacings, centers, units, axis mins

// TODO
// - Support for multichannel images, time data
// - option to write a detached header instead of detached nrrd file

// NB this class can be used to create detached headers for other file types
// See 

import ij.plugin.PlugIn;

import edu.stanford.rsl.conrad.io.NrrdFileWriter;


                          
public class Nrrd_Writer extends NrrdFileWriter implements PlugIn{
	public void run(String arg) {
		super.run(arg);
	}
}


