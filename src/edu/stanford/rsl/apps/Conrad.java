package edu.stanford.rsl.apps;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.io.Opener;
import ij.io.RoiDecoder;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;



import edu.stanford.rsl.apps.gui.roi.EvaluateROI;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.OpenCLParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.motion.ParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.io.IndividualFilesProjectionDataSink;
import edu.stanford.rsl.conrad.io.VTKVectorField;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.renderer.ParallelProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.phantom.renderer.SliceParallelVolumePhantomRenderer;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FloatArrayUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class Conrad {

	public static PhantomRenderer phantom;
	public static AnalyticPhantom4D phantom4D;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		printStartup();
		if (args.length == 0){
			printSynopsis();
		} else {
			String command = args[0];
			if (command.equals("help")){
				if (args.length != 2){
					printListOfCommands();
				} else {
					printHelp(args[1]);
				}
			} else if (command.equals("execute")){
				if (args.length == 5){
					executePipeline(args[1], args[2], args[3], args[4]);
				} else {
					System.out.println("Wrong number of arguments for 'execute'. (n = " + args.length + ")\n");
					printHelp(command);
				}
			} else if (command.equals("evaluate")){
				if (args.length == 6){
					evaluateROI(args[1], args[2], args[3], args[4], Integer.parseInt(args[5]));
				} else {
					System.out.println("Wrong number of arguments for 'evaluate'. (n = " + args.length + ")\n");
					printHelp(command);
				}
			} else if (command.equals("render")){
				if (args.length == 3){
					renderPhantom(args[1], args[2], -1);
				} else if (args.length == 4){
					renderPhantom(args[1], args[2], Integer.parseInt(args[3]));
				} else {
					System.out.println("Wrong number of arguments for 'render'. (n = " + args.length + ")\n");
					printHelp(command);
				}
			} else if (command.equals("motionfield")){
				if (args.length == 6){
					createMotionfield(args[1], args[2], Double.parseDouble(args[3]), Double.parseDouble(args[4]), args[5]);
				}else {
					System.out.println("Wrong number of arguments for 'motionfield'. (n = " + args.length + ")\n");
					printHelp(command);
				}
			} else if (command.equals("compareVTKvectorfields")){
				if (args.length == 4){
					compareVTKVectorFields(args[1], args[2], args[3]);
				}else {
					System.out.println("Wrong number of arguments for 'compareVTKvectorfields'. (n = " + args.length + ")\n");
					printHelp(command);
				}
			} else {
				System.out.println("Command was not recognized.\n");
				printSynopsis();
			}
		}
	}
	
	private static void compareVTKVectorFields(String configfile, String file1, String file2){
		System.out.print("Reading configuration ... ");
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(configfile));
		System.out.println("done");
		try {
			System.out.print("Reading vector field 1 ... ");
			float[] motionfield1 = VTKVectorField.readFromFile3D(file1);
			System.out.println("done");
			System.out.print("Reading vector field 2 ... ");
			float[] motionfield2 = VTKVectorField.readFromFile3D(file2);
			System.out.println("done");
			System.out.println("Root Mean Square Error: " + Math.sqrt(FloatArrayUtil.computeMeanSquareError(motionfield1, motionfield2)));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	private static void createMotionfield(String configFile, String outfile, double fromTime, double toTime, String interpolationType){
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(configFile));
		try {
			System.out.print("Reading phantom ... ");
			if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION) != null) {
				FileInputStream fis;

				fis = new FileInputStream(Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION));

				ObjectInputStream ois = new ObjectInputStream(fis);

				phantom4D = (AnalyticPhantom4D) ois.readObject();
				ois.close();
			} else {
				phantom4D = (AnalyticPhantom4D) UserUtil.queryPhantom("Select phantom:", "Phantom Selection");
				phantom4D.configure();
			}
			System.out.println("done");
			// phantom loaded.
			// write vtkmesh to file.
			
			Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
			if (("true").equals(Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.RENDER_PHANTOM_VOLUME_AUTO_CENTER))) {
				traj.setOriginToPhantomCenter(phantom4D);
			}
			
			float [] motionfield = null;

			Configuration config = Configuration.getGlobalConfiguration();

			System.out.println("Creating motion field ... ");
			long time = System.currentTimeMillis();
			
			if (interpolationType.equals("ParzenCPU")) {
				motionfield = new float[config.getGeometry().getReconDimensionX()*config.getGeometry().getReconDimensionY()*config.getGeometry().getReconDimensionZ()*3];
				int zPitch = config.getGeometry().getReconDimensionX() *config.getGeometry().getReconDimensionY();
				int yPitch = config.getGeometry().getReconDimensionX();
				for (int z = 0; z<config.getGeometry().getReconDimensionZ(); z++){
					System.out.println("Computing slice " + z);
					for (int y = 0; y<config.getGeometry().getReconDimensionY(); y++){
						for (int x = 0; x<config.getGeometry().getReconDimensionX(); x++){
							PointND currentVoxel = new PointND(
									General.voxelToWorld(x, config.getGeometry().getVoxelSpacingX(), config.getGeometry().getOriginX()),
									General.voxelToWorld(y, config.getGeometry().getVoxelSpacingY(), config.getGeometry().getOriginY()),
									General.voxelToWorld(z, config.getGeometry().getVoxelSpacingZ(), config.getGeometry().getOriginZ()));
							PointND destination = phantom4D.getPosition(currentVoxel, fromTime, toTime);
							SimpleVector direction = destination.getAbstractVector();
							direction.subtract(currentVoxel.getAbstractVector());
							if(direction.getElement(1) == Double.POSITIVE_INFINITY){
								System.out.println("Infinity");
							}
							int idx = z*zPitch+y*yPitch+x;
							motionfield[idx*3] =(float) direction.getElement(0);
							motionfield[idx*3+1] =(float) direction.getElement(1);
							motionfield[idx*3+2] =(float) direction.getElement(2);
						}
					}
				}
			}
			
			if (interpolationType.equals("ParzenGPU")) { 
				CLContext context = OpenCLUtil.createContext();
				CLDevice device = context.getMaxFlopsDevice();
				ParzenWindowMotionField motion = (ParzenWindowMotionField) phantom4D.getMotionField();
				OpenCLParzenWindowMotionField cl = new OpenCLParzenWindowMotionField(motion, context, device);
				motionfield = cl.getMotionFieldAsArray(fromTime, toTime);
			}
			
			if (interpolationType.equals("ParzenGPUZFilter")) { 
				CLContext context = OpenCLUtil.createContext();
				CLDevice device = context.getMaxFlopsDevice();
				ParzenWindowMotionField motion = (ParzenWindowMotionField) phantom4D.getMotionField();
				OpenCLParzenWindowMotionField cl = new OpenCLParzenWindowMotionField(motion, context, device);
				motionfield = cl.getMotionFieldAsArrayReduceZ(fromTime, toTime);
			}
			
			if (interpolationType.equals("ParzenGPUZFilterGridXY")) {
				CLContext context = OpenCLUtil.createContext();
				CLDevice device = context.getMaxFlopsDevice();
				ParzenWindowMotionField motion = (ParzenWindowMotionField) phantom4D.getMotionField();
				OpenCLParzenWindowMotionField cl = new OpenCLParzenWindowMotionField(motion, context, device);
				int factor = 16; 
				motionfield = cl.getMotionFieldAsArrayReduceZGridXY(fromTime, toTime, config.getGeometry().getReconDimensionX()/factor, config.getGeometry().getReconDimensionY()/factor);
			}
			
			if (interpolationType.equals("ParzenGPURBC")) { 
				CLContext context = OpenCLUtil.createContext();
				CLDevice device = context.getMaxFlopsDevice();
				ParzenWindowMotionField motion = (ParzenWindowMotionField) phantom4D.getMotionField();
				OpenCLParzenWindowMotionField cl = new OpenCLParzenWindowMotionField(motion, context, device);
				motionfield = cl.getMotionFieldAsArrayRandomBallCover(fromTime, toTime, 500000);
			}

			time = System.currentTimeMillis() - time;
			System.out.print("Writing to file ... ");
			VTKVectorField.writeToFile3D(outfile, motionfield);
			System.out.println("done");

			System.out.println("Interpolation took: " + time);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}



	private static void renderPhantom(String configFile, String outfile, int frame){
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(configFile));
		try {
			IndividualFilesProjectionDataSink sink = new IndividualFilesProjectionDataSink();
			sink.setDirectory(outfile);
			sink.setPrefix("phantom.");
			sink.setFormat(IndividualFilesProjectionDataSink.Float32Bit);
			sink.setLittleEndian(true);
			sink.configured();
			if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION) != null) {
				FileInputStream fis = new FileInputStream(Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.SPLINE_4D_LOCATION));
				ObjectInputStream ois = new ObjectInputStream(fis);
				phantom = new ParallelProjectionPhantomRenderer();
				AnalyticPhantom phan = (AnalyticPhantom4D) ois.readObject();
				ParallelProjectionPhantomRenderer para = (ParallelProjectionPhantomRenderer) phantom;
				para.configure(phan, Configuration.getGlobalConfiguration().getDetector()); 	
				ois.close();
			} else {
				phantom = (PhantomRenderer) UserUtil.chooseObject("Select Phantom: ", "Phantom Selection", PhantomRenderer.getPhantoms(), PhantomRenderer.getPhantoms()[0]);
				phantom.configure();

			}
			long time = System.currentTimeMillis();
			if (frame == -1) {
				Thread thread = new Thread(new Runnable(){public void run(){phantom.createPhantom();}});
				thread.start();
				ParallelImageFilterPipeliner pipeliner = new ParallelImageFilterPipeliner(phantom, new ImageFilteringTool[]{}, sink);
				pipeliner.project(true);
				sink.getResult();
			} else {
				SliceParallelVolumePhantomRenderer renderer = (SliceParallelVolumePhantomRenderer) phantom;
				renderer.getModelWorker().workOnSlice(frame);
				sink.process(renderer.getModelWorker().getImageProcessorBufferValue().get(frame),frame);
				sink.close();
				sink.getResult();
			}
			time = System.currentTimeMillis() - time;
			System.out.println("Runtime: " + time/1000.0 + " s");
			System.out.println("Saving to " + outfile);
			System.out.println("All done.\nThanks for using Conrad today. As far as we know, Conrad has never had an undetected error.");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static void evaluateROI(String configFile, String method, String roiFile, String dataFile, int slice){
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(configFile));
		try {
			System.out.println("Evaluating: " + dataFile);
			Opener open = new Opener();
			ImagePlus imp = open.openImage(dataFile);
			imp.setPosition(slice+1);
			RoiDecoder rd = new RoiDecoder(roiFile);
			Roi roi = rd.getRoi();
			EvaluateROI eval = (EvaluateROI) Class.forName(method).newInstance();
			eval.setRoi(roi);
			eval.setImage(imp);
			eval.setConfigured(true);
			eval.evaluate();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	private static void executePipeline(String sinkName, String configFile,  String inFile, String outFile){
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(configFile));
		try {
			ProjectionSource pSource = FileProjectionSource.openProjectionStream(inFile);
			ImageFilteringTool [] filters = Configuration.getGlobalConfiguration().getFilterPipeline();
			File out = new File(outFile);
			boolean isConfigured = true;
			for (int i = 0; i < filters.length; i++){
				if (!filters[i].isConfigured()) isConfigured = false;
			}
			if (isConfigured) {
				try {
					BufferedProjectionSink sink = (BufferedProjectionSink) Class.forName(sinkName).newInstance();
					System.out.println("Running pipeline with " + sink.getName());
					sink.configure();
					ParallelImageFilterPipeliner filteringPipeline = new ParallelImageFilterPipeliner(pSource, filters, sink);
					long time = System.currentTimeMillis();
					try {
						filteringPipeline.project();
					} catch (Exception e) {
						e.printStackTrace();
					}
					sink.getResult();
					time = System.currentTimeMillis() - time;
					System.out.println("Runtime: " + time/1000.0 + " s");
					//if (sink instanceof ReconstructionFilter) {
					//	ReconstructionFilter recon = (ReconstructionFilter) sink;
					//	if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) recon.applyHounsfieldScaling();
					//}
					ImagePlus result = ImageUtil.wrapGrid3D(sink.getResult(),"Result of " + configFile);	
					// save to out-file; much simpler than I expected!
					System.out.println("Saving to " + outFile);
					IJ.saveAs(result, out.getName(), out.getAbsolutePath());
					System.out.println("All done.\nThanks for using Conrad today. As far as we know, Conrad has never had an undetected error.");
				} catch (InstantiationException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IllegalAccessException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (ClassNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			} else {
				System.out.println("Pipeline is not configured.");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private static void printHelp(String command){
		if (command.equals("execute")) {
			System.out.println("Usage: Conrad execute projection-sink config-file data-file out-file\n\nThis command will apply the pipeline defined in the config-file to the data-file and store the result in out-file. The out-file type is determined by its extension.\nUse the ReconstructionPipelineFrame in conrad.gui to create a config-file.\n");
			BufferedProjectionSink [] sinks = BufferedProjectionSink.getProjectionDataSinks();
			System.out.println("projection-sink has to be one of this list:\n");
			for (BufferedProjectionSink sink: sinks) {
				System.out.println(sink.getClass().getName() +"\t" + sink.getName());
			}
		} else if (command.equals("help")) {
			printListOfCommands();
		} else if (command.equals("evaluate")) {
			System.out.println("Usage: Conrad evaluate config-file method roi-file data-file slice\n\nThis command will apply the method with the parameters defined in the config-file to the data-file \nUse the ReconstructionPipelineFrame in conrad.gui to create a config-file.\n");
			EvaluateROI [] methods = EvaluateROI.knownMethods();
			System.out.println("method has to be one of this list:\n");
			for (EvaluateROI method: methods) {
				System.out.println(method.getClass().getName() +"\t" + method.toString());
			}
		} else if (command.equals("render")) {
			System.out.println("Usage: Conrad render config-file outfile [frame]\n\nThis command will render a phantom and save it in the out-file. The out-file type is determined by its extension.\nUse the ReconstructionPipelineFrame in conrad.gui to create a config-file.\n");
		} else if (command.equals("motionfield")) {
			System.out.println("Usage: Conrad motionfield config-file outfile timeFrom timeTo\n\nThis command will generate the motionfield of a phantom and save it in the out-file. The out-file type is determined by its extension.\nUse the ReconstructionPipelineFrame in conrad.gui to create a config-file.\n");
		} else if (command.equals("compareVTKvectorfields")) {
			System.out.println("Usage: Conrad compareVTKvectorfields config-file file1 file2\n\nThis command will compute the root mean square error between the two given vector fields.\n");
		} else {
			System.out.println("Command '" + command + "' unknown.\n");
			printListOfCommands();
		}
	}

	private static void printListOfCommands(){
		System.out.println("List of available commands:\n");
		System.out.println("help    - print this screen");
		System.out.println("execute - execute a pipeline on a specified data set");
		System.out.println("evaluate - evaluate an roi in a given dataset");
		System.out.println("render - create projection data");
		System.out.println("motionfield - generate a motionfield in vtk format");
		System.out.println("compareVTKvectorfields - compare two vector fields in vtk format");
	}

	private static void printStartup(){
		System.out.println("CONRAD Commandline Tool\n" + CONRAD.VersionString +"\n");
	}

	private static void printSynopsis(){
		System.out.println("Usage:\njava edu.stanford.rsl.apps.Conrad command\n\nUse command 'help' for a list of commands.\n\n");		
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
