package edu.stanford.rsl.conrad.io;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import org.junit.Test;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.DicomDecoder;
import edu.stanford.rsl.conrad.utils.FileUtil;



public class DicomProjectionSource extends FileProjectionSource {

	protected boolean directoryMode = false;
	protected ArrayList<Grid2D> directoryImages;
	protected int current = 0;

	public void initStream (String filename, boolean skipCheck) throws IOException{

		if (!(filename.toLowerCase().endsWith("dcm") || filename.toLowerCase().endsWith("ima")||(skipCheck))) {
			File dir = new File (filename);
			directoryMode = dir.isDirectory();
			if (!directoryMode) throw new RuntimeException("DicomProjectionSource:Not a dicom file!");
		}
		if (directoryMode) {
			File dir = new File(filename);
			String [] files = dir.list();
			//Arrays.sort(files);
			Arrays.sort(files, new Comparator<String>() {
		        public int compare(String o1, String o2) {
		            return extractInt(o1) - extractInt(o2);
		        }

		        int extractInt(String s) {
		            String num = s.replaceAll("\\D", "");
		            // return 0 if no digits found
		            return num.isEmpty() ? 0 : Integer.parseInt(num);
		        }
		    });
			directoryImages = new ArrayList<Grid2D>();
			for (int i= 0; i < files.length; i++){
				DicomProjectionSource singleFileSource = new DicomProjectionSource();
				singleFileSource.initStream(filename + File.separator + files[i], true);
				Grid2D nextProj = singleFileSource.getNextProjection();
				if (KEVSDicomTag.KEVS.get(0) != null){
					// we found KEVS
					MultiChannelGrid2D multi = new MultiChannelGrid2D(nextProj.getWidth(), nextProj.getHeight(), 5);
					multi.setChannel(0, nextProj);
					for (int j = 0; j < 4; j++){
						Grid2D channel = new Grid2D(KEVSDicomTag.KEVS.get(j), nextProj.getWidth(), nextProj.getHeight());
						multi.setChannel(j+1, channel);
					}
					directoryImages.add(multi);
				} else {
					directoryImages.add(nextProj);
				}
				singleFileSource.close();
			}
		} else {
			File file = new File(filename);
			String fileName = file.getName();
			if (fileName==null)
				return;
			BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file));
			initStream(inputStream);
		}
	}

	public Grid2D getNextProjection(){
		if (!directoryMode) {
			return super.getNextProjection();
		} else {
			Grid2D next = (current < directoryImages.size())? directoryImages.get(current) : null;
			current++;
			currentIndex++;
			return next;
		}
	}

	public void initStream (String filename) throws IOException{
		initStream(filename, false);
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