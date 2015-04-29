/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.motion.compensation;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.OpenCLParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.motion.ParzenWindowMotionField;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;

public class OpenCLMotionCompensatedBackProjector extends OpenCLBackProjector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3263500016065695870L;
	protected MotionField motion = null;
	protected double referenceTime = 0;
	protected CLBuffer<FloatBuffer> motionParameters = null;

	/**
	 * @return the motion
	 */
	public MotionField getMotion() {
		return motion;
	}

	/**
	 * @param motion the motion to set
	 */
	public void setMotion(MotionField motion) {
		this.motion = motion;
	}



	protected String readCompleteRessourceAsString(String resource) throws IOException{
		InputStream inStream = OpenCLMotionCompensatedBackProjector.class.getResourceAsStream(resource);
		BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
		String content = "";
		String line = br.readLine();
		while (line != null){
			content += line + "\n";
			line = br.readLine();
		};
		return content;
	}

	@Override
	protected void createProgram() throws IOException{
		// initialize the program
		if (program==null || !program.getContext().equals(this.context)){
			String motionProgram = "";
			if (motion instanceof ParzenWindowMotionField){
				motionProgram = readCompleteRessourceAsString("applyMotionDenseMotionField.cl");
			}
			String backprojectProgram = readCompleteRessourceAsString("generalMotionCompensatedBackprojectCL.cl");
			program = context.createProgram(motionProgram + backprojectProgram).build();
		}
	}

	protected void initMotionParameters(double referenceTime, double currentTime){
		Configuration config = Configuration.getGlobalConfiguration();
		if (motion instanceof ParzenWindowMotionField){
			if (motionParameters != null) {
				motionParameters.release();
				motionParameters = null;
			}
			ParzenWindowMotionField motion = (ParzenWindowMotionField) this.motion;
			OpenCLParzenWindowMotionField cl;
			try {
				cl = new OpenCLParzenWindowMotionField(motion, context, device);
				int factor = 16; 
				motionParameters = cl.getMotionFieldAsArrayReduceZGridXY(referenceTime, currentTime, OpenCLUtil.iDivUp(config.getGeometry().getReconDimensionX(),factor), OpenCLUtil.iDivUp(config.getGeometry().getReconDimensionY(),factor), commandQueue, true);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}


	}

	@Override
	protected synchronized void projectSingleProjection(int projectionNumber, int dimz){
		// load projection matrix
		initProjectionMatrix(projectionNumber);

		// Correct for constant part of distance weighting + For angular sampling
		double D =  getGeometry().getSourceToDetectorDistance();
		float projectionMultiplier = (float)(10 * D*D * 2* Math.PI * getGeometry().getPixelDimensionX() / getGeometry().getNumProjectionMatrices());

		initProjectionData(projections.get(projectionNumber));
		//System.out.println("Uploading projection " + projectionNumber);
		if (!largeVolumeMode) {
			projections.remove(projectionNumber);
		}
		// backproject for each slice
		// OpenCL Grids are only two dimensional!
		int reconDimensionZ = dimz;
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();


		initMotionParameters(referenceTime, ((float)projectionNumber)/(projections.size()-1));

		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(volumePointer)
		.putArg(getGeometry().getReconDimensionX())
		.putArg(getGeometry().getReconDimensionY())
		.putArg(reconDimensionZ)
		.putArg((int) lineOffset)
		.putArg((float) voxelSpacingX)
		.putArg((float) voxelSpacingY)
		.putArg((float) voxelSpacingZ)
		.putArg((float) offsetX)
		.putArg((float) offsetY)
		.putArg((float) offsetZ)
		.putArg(projectionTex)
		.putArg(projectionMatrix)
		.putArg(projectionMultiplier)
		.putArg(motionParameters);

		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));

		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {getGeometry().getReconDimensionX(), getGeometry().getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}

		// Call the OpenCL kernel, writing the results into the volume which is pointed at
		commandQueue
		.putWriteImage(projectionTex, true)
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		//.finish()
		//.putReadBuffer(dOut, true)
		.finish();
	}

	/**
	 * @return the referenceTime
	 */
	public double getReferenceTime() {
		return referenceTime;
	}
	
	@Override
	public String getToolName(){
		return "Motion Compensated OpenCL Backprojector";
	}

	/**
	 * @param referenceTime the referenceTime to set
	 */
	public void setReferenceTime(double referenceTime) {
		this.referenceTime = referenceTime;
	}
	
	@Override
	protected synchronized void unload() {
		if(motionParameters!=null && !motionParameters.isReleased())
			motionParameters.release();
		super.unload();
	}

}
