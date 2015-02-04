/*
 * Copyright (C) 2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.IOException;

import ij.io.FileInfo;
import edu.stanford.rsl.conrad.utils.DicomDecoder;

public class AdditionalDicomTag {
	public DicomDecoder decoder;
	public int tag = 0x00180050;
	public FileInfo fi;

	public boolean readTag(int tag, int elementLength) throws IOException{
		if (tag == this.tag ) {
			String spacing = decoder.getString(elementLength);
			fi.pixelDepth = decoder.s2d(spacing);
			decoder.addInfo(tag, spacing);
			return true;
		} else {
			return false;
		}
	}

}
