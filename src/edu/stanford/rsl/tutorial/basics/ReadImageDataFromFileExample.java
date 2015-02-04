package edu.stanford.rsl.tutorial.basics;

import ij.ImageJ;
import ij.io.FileInfo;
import ij.io.FileOpener;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class ReadImageDataFromFileExample {

	public static void main(String[] args) {
		new ImageJ();
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY16_UNSIGNED;
		fI.height = 256;
		fI.width = 256;
		fI.nImages = 256;
		fI.intelByteOrder = true;
		
		fI.directory = args[0];
		fI.fileName = args[1];		
		
		FileOpener fO = new FileOpener(fI);
				
		Grid3D grid = ImageUtil.wrapImagePlus(fO.open(false));
		
		NumericPointwiseIteratorND pIter = new NumericPointwiseIteratorND(grid);
		while (pIter.hasNext())
		{
			if ( pIter.get() == 0 )
			{
				pIter.getNext();
			} else {
				pIter.setNext(1000+pIter.get());
			}
				
		}
		grid.show("Skull");
	   
	}

}
