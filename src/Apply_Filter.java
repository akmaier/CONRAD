/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
import java.lang.annotation.Annotation;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.HideOnUIAnnotation;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.filtering.opencl.BilateralFiltering3DTool;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.ImagePlusProjectionDataSource;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.pipeline.ParallelProjectionDataSinkFeeder;
import edu.stanford.rsl.conrad.reconstruction.ReconstructionFilter;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;

/**
 * Filters the projection data with a 2-D filter which is selected via the ImageJ GUI.
 * 
 * @author Andreas Maier
 *
 */
public class Apply_Filter implements PlugIn {

	public Apply_Filter(){
		
	}
	
	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		ImageFilteringTool [] filters = ImageFilteringTool.getFilterTools();
		
		// Hide filtering tools which are annotated with HideOnUI 
		filters = sortImageFilteringTools(filters);
		
		ImagePlus projections = IJ.getImage();
		ImageFilteringTool filter = (ImageFilteringTool) JOptionPane.showInputDialog(null, "Select filter: ", "Filter Selection", JOptionPane.PLAIN_MESSAGE, null, filters, filters[0]);
		try {
			filter.configure();
			if (filter instanceof IndividualImageFilteringTool) {
				ImagePlusDataSink sink = new ImagePlusDataSink();
				sink.configure();
				ImagePlusProjectionDataSource pSource = new ImagePlusProjectionDataSource();
				pSource.setImage(ImageUtil.wrapImagePlus(projections, true));
				ImageFilteringTool [] filts = {filter};
				ParallelImageFilterPipeliner filteringPipeline = new ParallelImageFilterPipeliner(pSource, filts, sink);
				try {
					filteringPipeline.project();
				} catch (Exception e) {
					e.printStackTrace();
				}
				ImagePlus revan = ImageUtil.wrapGrid3D(sink.getResult(), filter.toString());
				revan.setTitle(filter.getToolName());
				ImageUtil.applyConradImageCalibration(revan, filter instanceof ReconstructionFilter);
				revan.show();
			} 
			if (filter instanceof MultiProjectionFilter){
				MultiProjectionFilter filt = (MultiProjectionFilter) filter;
				ImagePlusDataSink imp = new ImagePlusDataSink();
				imp.configure();
				filt.setSink(imp);
				filt.start();
				ParallelProjectionDataSinkFeeder.projectParallel(ImageUtil.wrapImagePlus(projections, true), filt, true);
				ImagePlus revan = ImageUtil.wrapGrid3D(imp.getResult(), filter.toString());
				revan.setTitle(filt.getToolName());
				ImageUtil.applyConradImageCalibration(revan, filter instanceof ReconstructionFilter);
				revan.show();
			}
			
			if (filter instanceof BilateralFiltering3DTool) {
				Grid3D source = ImageUtil.wrapImagePlus(projections, false);
				Grid3D result = ((BilateralFiltering3DTool) filter).process(source);
				ImagePlus revan = ImageUtil.wrapGrid3D(result, filter.toString());
				revan.setTitle(filter.getToolName());
				ImageUtil.applyConradImageCalibration(revan, filter instanceof ReconstructionFilter);
				revan.show();
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			IJ.log(e.getLocalizedMessage());
		}
	
	}
	
	
	/**
	 * Hide image filtering tools annotated with HideOnUI on their class declaration
	 * @param filters to sort
	 * @return sorted filters as array
	 */
	private ImageFilteringTool[] sortImageFilteringTools(ImageFilteringTool[] filteringTools) {
		List<ImageFilteringTool> toShow = new ArrayList<ImageFilteringTool>();
		for (ImageFilteringTool tool : filteringTools) {
			if (!tool.getClass().isAnnotationPresent((Class<? extends Annotation>) HideOnUIAnnotation.class)) {
				toShow.add(tool);
			}
		}
		
		ImageFilteringTool[] filteredTools = toShow.toArray(new ImageFilteringTool[toShow.size()]);
		return filteredTools;
	}

}


