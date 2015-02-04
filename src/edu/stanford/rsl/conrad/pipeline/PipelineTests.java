package edu.stanford.rsl.conrad.pipeline;


import org.junit.Test;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.HorizontalFlippingTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.utils.FileUtil;

public class PipelineTests {

	@Test
	public void testSink(){
		// This reads all data from the source and discards the projection in the sink.
		// This should work without memory leakage.
		// akmaier
		String filename;
		try {
			filename = FileUtil.myFileChoose(".IMA", false);
			BufferedProjectionSink sink = new DevNullSink();
			ProjectionSource source = FileProjectionSource.openProjectionStream(filename);
			Grid2D image = source.getNextProjection();
			while (image != null){
				System.out.println("Running projection: " + source.getCurrentProjectionNumber());
				sink.process(image, source.getCurrentProjectionNumber());
				image = source.getNextProjection();
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	@Test
	public void testPipeline(){
		String filename;
		try {
			filename = FileUtil.myFileChoose(".IMA", false);
			BufferedProjectionSink sink = new DevNullSink();
			ImageFilteringTool[] tools = {new HorizontalFlippingTool()};
			ProjectionSource source = FileProjectionSource.openProjectionStream(filename);
			ParallelImageFilterPipeliner pipeliner = new ParallelImageFilterPipeliner(source, tools, sink);
			pipeliner.project();
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