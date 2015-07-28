

// (c) Gregory Jefferis 2007
// Department of Zoology, University of Cambridge
// jefferis@gmail.com
// All rights reserved
// Source code released under Lesser Gnu Public License v2

// TODO
// - Support for multichannel images
//   (problem is how to figure out they are multichannel in the absence of 
//   other info - not strictly required by nrrd format)
// - time datasets
// - line skip (only byte skip at present)
// - calculating spacing information from axis mins/cell info

// Compiling:
// You must compile Nrrd_Writer.java first because this plugin
// depends on the NrrdFileInfo class declared in that file

import ij.plugin.PlugIn;

import edu.stanford.rsl.conrad.io.NrrdFileReader;


/**
 * ImageJ plugin to read a file in Gordon Kindlmann's NRRD 
 * or 'nearly raw raster data' format, a simple format which handles
 * coordinate systems and data types in a very general way.
 * See <A HREF="http://teem.sourceforge.net/nrrd">http://teem.sourceforge.net/nrrd</A>
 * and <A HREF="http://flybrain.stanford.edu/nrrd">http://flybrain.stanford.edu/nrrd</A>
 */

public class Nrrd_Reader extends NrrdFileReader implements PlugIn 
{
	public void run(String arg) {
		super.run(arg);
	}
}
