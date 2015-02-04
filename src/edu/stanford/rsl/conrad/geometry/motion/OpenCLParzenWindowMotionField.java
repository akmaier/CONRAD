package edu.stanford.rsl.conrad.geometry.motion;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

/** 
 * General Class to map any of the ParzenWindowMotionField classes to GPU. We use the original class to create
 * the raster points. The expensive motionfield evaluation is then done on GPU.
 * @author akmaier
 *
 */
public class OpenCLParzenWindowMotionField extends ParzenWindowMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -322305647900119904L;
	ParzenWindowMotionField originalMotionField;
	CLContext context = null;
	CLDevice device = null;
	CLProgram program;
	static int bpBlockSize[] = {32, 16};


	public static CLBuffer<FloatBuffer> generateTimeSamplingPoints(float tIndex, int elementCountU, int elementCountV, CLContext context, CLDevice device){
		// prepare sampling points
		CLBuffer<FloatBuffer> samplingPoints = context.createFloatBuffer(elementCountU * elementCountV*3, Mem.READ_ONLY);
		for (int j = 0; j < elementCountV; j++){
			for (int i = 0; i < elementCountU; i++){
				samplingPoints.getBuffer().put(i*(1.0f / elementCountU));
				samplingPoints.getBuffer().put(j*(1.0f / elementCountV));
				samplingPoints.getBuffer().put(tIndex);
				//System.out.println(i*(1.0f / elementCountU) + " " +j*(1.0f / elementCountV)+ " "+t*(1.0f / elementCountT) );
			}
		}
		samplingPoints.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(samplingPoints, true).finish();
		clc.release();
		return samplingPoints;
	}

	public OpenCLParzenWindowMotionField(ParzenWindowMotionField motion, CLContext context, CLDevice device) throws IOException {
		super(motion.sigma);
		originalMotionField = motion;
		this.device = device;
		this.context = context;
		InputStream programFile = OpenCLParzenWindowMotionField.class.getResourceAsStream("MotionField.cl");
		program = context.createProgram(programFile).build();

	}

	@Override
	public
	PointND[] getRasterPoints(double time) {
		return originalMotionField.getRasterPoints(time);
	}

	private void fillBuffer(CLBuffer<IntBuffer> searchIdx, int[] searchIndicies) {
		IntBuffer buffer = searchIdx.getBuffer();
		for(int i= 0; i < searchIndicies.length; i++){
			buffer.put(searchIndicies[i]);
		}
		buffer.rewind();
	}

	private void fillBuffer(CLBuffer<FloatBuffer> clbuffer, PointND[] points){
		FloatBuffer buffer = clbuffer.getBuffer();
		for(int i= 0; i < points.length; i++){
			buffer.put((float) points[i].get(0));
			buffer.put((float) points[i].get(1));
			buffer.put((float) points[i].get(2));
		}
		buffer.rewind();
	}

	private void fillBuffer(CLBuffer<FloatBuffer> clbuffer, ArrayList<PointND> points){
		FloatBuffer buffer = clbuffer.getBuffer();
		for(int i= 0; i < points.size(); i++){
			buffer.put((float) points.get(i).get(0));
			buffer.put((float) points.get(i).get(1));
			buffer.put((float) points.get(i).get(2));
		}
		buffer.rewind();
	}

	private void fillBuffer(CLBuffer<FloatBuffer> search,
			float[] searchCandidates) {
		FloatBuffer buffer = search.getBuffer();
		for(int i= 0; i < searchCandidates.length; i++){
			buffer.put((float) searchCandidates[i]);
		}
		buffer.rewind();
	}

	@Override
	public ArrayList<PointND> getPositions(double from, double to, PointND ... initialPositions){
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();

		// Allocate buffers
		CLBuffer<FloatBuffer> inputPoint = context.createFloatBuffer(input.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> outputPoint = context.createFloatBuffer(output.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(initialPositions.length*3, Mem.READ_WRITE);

		fillBuffer(inputPoint, input);
		fillBuffer(outputPoint, output);
		fillBuffer(motionField, initialPositions);

		CLKernel kernel = program.createCLKernel("evaluateParzen1D");


		int[] realLocalSize = {Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0])};
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {(int) (Math.ceil(initialPositions.length/(double)realLocalSize[0]))*realLocalSize[0]}; 


		long time = System.currentTimeMillis();
		queue.putWriteBuffer(inputPoint, true)
		.putWriteBuffer(outputPoint, true)
		.putWriteBuffer(motionField, true);

		kernel
		.putArg(inputPoint)
		.putArg(outputPoint)
		.putArg(motionField)
		.putArg(initialPositions.length)
		.putArg(output.length)
		.putArg((float) sigma);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize[0], realLocalSize[0])
		.putReadBuffer(motionField, true)
		.finish();

		time = System.currentTimeMillis() - time;

		//System.out.println("Kernel execution times: " + time);

		ArrayList<PointND> result = new ArrayList<PointND>();
		motionField.getBuffer().rewind();
		for(int i=0; i < initialPositions.length; i++){
			PointND newPointND = new PointND (initialPositions[i].get(0) + motionField.getBuffer().get(), initialPositions[i].get(1) +motionField.getBuffer().get(), initialPositions[i].get(2) +motionField.getBuffer().get());
			result.add(newPointND);
		}
		// release buffers
		inputPoint.release();
		outputPoint.release();
		motionField.release();
		queue.release();
		kernel.release();
		return result;
	}

	public float [] getMotionFieldAsArrayNN(double from, double to){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();

		// Allocate buffers
		CLBuffer<FloatBuffer> inputPoint = context.createFloatBuffer(input.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> outputPoint = context.createFloatBuffer(output.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3, Mem.WRITE_ONLY);

		fillBuffer(inputPoint, input);
		fillBuffer(outputPoint, output);


		CLKernel kernel = program.createCLKernel("evaluateNN");


		// determine local worksize. "getMaxWorkGroupSize()" returns the overall number of workers! Thus the product over all elements of bpBlockSize
		// must be smaller then "getMaxWorkGroupSize()"
		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {geom.getReconDimensionX(), geom.getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}


		long time = System.currentTimeMillis();
		queue.putWriteBuffer(inputPoint, true)
		.putWriteBuffer(outputPoint, true);

		double originx = geom.getOriginX();
		kernel.rewind();
		kernel
		.putArg(inputPoint)
		.putArg(outputPoint)
		.putArg(motionField)
		.putArg(geom.getReconDimensionX())
		.putArg(geom.getReconDimensionY())
		.putArg(0)//dummy integer - will be overwritten in loop
		.putArg(output.length)
		.putArg((float) sigma)
		.putArg((float) geom.getVoxelSpacingX())
		.putArg((float) geom.getVoxelSpacingY())
		.putArg((float) geom.getVoxelSpacingZ())
		.putArg((float) originx)
		.putArg((float) geom.getOriginY())
		.putArg((float) geom.getOriginZ());
		
		for(int i=0; i < geom.getReconDimensionZ();i++){
			System.out.println("Computing Slice " + i);
			kernel.setArg(5, i);
			queue.put2DRangeKernel(kernel, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
			.finish();

		}

		queue.putReadBuffer(motionField, true).finish();
		time = System.currentTimeMillis() - time;

		System.out.println("Kernel execution times: " + time);

		float [] result = new float [geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3];
		motionField.getBuffer().rewind();
		for(int i=0; i < result.length; i++){
			result[i] = motionField.getBuffer().get();
		}
		// release buffers
		inputPoint.release();
		outputPoint.release();
		motionField.release();
		queue.release();
		kernel.release();
		return result;
	}

	public float [] getMotionFieldAsArray(double from, double to){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();

		// Allocate buffers
		CLBuffer<FloatBuffer> inputPoint = context.createFloatBuffer(input.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> outputPoint = context.createFloatBuffer(output.length * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3, Mem.WRITE_ONLY);

		fillBuffer(inputPoint, input);
		fillBuffer(outputPoint, output);


		CLKernel kernel = program.createCLKernel("evaluateParzen");


		// determine local worksize. "getMaxWorkGroupSize()" returns the overall number of workers! Thus the product over all elements of bpBlockSize
		// must be smaller then "getMaxWorkGroupSize()"
		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {geom.getReconDimensionX(), geom.getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}


		long time = System.currentTimeMillis();
		queue.putWriteBuffer(inputPoint, true)
		.putWriteBuffer(outputPoint, true);

		kernel.rewind();
		kernel
		.putArg(inputPoint)
		.putArg(outputPoint)
		.putArg(motionField)
		.putArg(geom.getReconDimensionX())
		.putArg(geom.getReconDimensionY())
		.putArg(0)//dummy integer will be overwritten inside loop
		.putArg(output.length)
		.putArg((float) sigma)
		.putArg((float) geom.getVoxelSpacingX())
		.putArg((float) geom.getVoxelSpacingY())
		.putArg((float) geom.getVoxelSpacingZ())
		.putArg((float) geom.getOriginX())
		.putArg((float) geom.getOriginY())
		.putArg((float) geom.getOriginZ());
		
		for(int i=0; i < geom.getReconDimensionZ();i++){
			System.out.println("Computing Slice " + i);
			kernel.setArg(5, i);
			queue.put2DRangeKernel(kernel, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
			.finish();

		}

		queue.putReadBuffer(motionField, true).finish();
		time = System.currentTimeMillis() - time;

		System.out.println("Kernel execution times: " + time);

		float [] result = new float [geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3];
		motionField.getBuffer().rewind();
		for(int i=0; i < result.length; i++){
			result[i] = motionField.getBuffer().get();
		}
		// release buffers
		inputPoint.release();
		outputPoint.release();
		motionField.release();
		queue.release();
		kernel.release();
		return result;
	}

	public float [] getMotionFieldAsArrayReduceZ(double from, double to){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();

		// Allocate buffers
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3, Mem.WRITE_ONLY);



		CLKernel kernel = program.createCLKernel("evaluateParzen");


		// determine local worksize. "getMaxWorkGroupSize()" returns the overall number of workers! Thus the product over all elements of bpBlockSize
		// must be smaller then "getMaxWorkGroupSize()"
		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {geom.getReconDimensionX(), geom.getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}

		ArrayList<ArrayList<PointND>> inputCandidates = new ArrayList<ArrayList<PointND>>();
		ArrayList<ArrayList<PointND>> outputCandidates = new ArrayList<ArrayList<PointND>>();

		double zMargin = (geom.getVoxelSpacingZ()*geom.getReconDimensionZ()*0.1) + 6 * sigma;
		for(int i=0; i < geom.getReconDimensionZ();i++){
			inputCandidates.add(new ArrayList<PointND>());
			outputCandidates.add(new ArrayList<PointND>());
			double zValue = i * geom.getVoxelSpacingZ() + geom.getOriginZ();
			for (int j=0; j < input.length; j++){
				double distance = Math.abs(input[j].get(2) - zValue);
				//System.out.println(distance + " " + zValue + " " +input[j].get(2));
				if (distance < zMargin){
					inputCandidates.get(i).add(input[j]);
					outputCandidates.get(i).add(output[j]);
				}
			}
		}

		long time = System.currentTimeMillis();
		double total = 0;

		// put static kernel arguments before loop
		kernel
		.setArg(2,motionField)
		.setArg(3,geom.getReconDimensionX())
		.setArg(4,geom.getReconDimensionY())
		.setArg(7,(float) sigma)
		.setArg(8,(float) geom.getVoxelSpacingX())
		.setArg(9,(float) geom.getVoxelSpacingY())
		.setArg(10,(float) geom.getVoxelSpacingZ())
		.setArg(11,(float) geom.getOriginX())
		.setArg(12,(float) geom.getOriginY())
		.setArg(13,(float) geom.getOriginZ());
		
		for(int i=0; i < geom.getReconDimensionZ();i++){

			if (inputCandidates.get(i).size()==0) continue;
			CLBuffer<FloatBuffer> inputPoint = context.createFloatBuffer(inputCandidates.get(i).size() * 3, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> outputPoint = context.createFloatBuffer(outputCandidates.get(i).size() * 3, Mem.READ_ONLY);
			fillBuffer(inputPoint, inputCandidates.get(i));
			fillBuffer(outputPoint, outputCandidates.get(i));

			queue.putWriteBuffer(inputPoint, true)
			.putWriteBuffer(outputPoint, true);

			//variable kernel arguments inside loop
			kernel
			.setArg(0,inputPoint)
			.setArg(1, outputPoint)
			.setArg(5, i)
			.setArg(6, inputCandidates.get(i).size());
			
			System.out.println("Computing Slice " + i + " (" + inputCandidates.get(i).size() + " points)");
			total += inputCandidates.get(i).size();

			queue.put2DRangeKernel(kernel, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
			.finish();

			inputPoint.release();
			outputPoint.release();
		}

		System.out.println("Evaluation quota: " + total/(input.length*geom.getReconDimensionZ()));

		queue.putReadBuffer(motionField, true).finish();
		time = System.currentTimeMillis() - time;
		System.out.println("Kernel execution times: " + time);

		float [] result = new float [geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3];
		motionField.getBuffer().rewind();
		for(int i=0; i < result.length; i++){
			result[i] = motionField.getBuffer().get();
		}
		// release buffers
		motionField.release();
		queue.release();
		kernel.release();
		return result;
	}

	public float [] getMotionFieldAsArrayReduceZGridXY(double from, double to, int nx, int ny){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();
		CLBuffer<FloatBuffer> motionField = getMotionFieldAsArrayReduceZGridXY(from, to, nx, ny, queue, true);
		queue.putReadBuffer(motionField, true).finish();
		float [] result = new float [geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3];
		motionField.getBuffer().rewind();
		for(int i=0; i < result.length; i++){
			result[i] = motionField.getBuffer().get();
		}
		// release buffers
		motionField.release();
		queue.release();
		return result;
	}

	public CLBuffer<FloatBuffer> getMotionFieldAsArrayReduceZGridXY(double from, double to, int nx, int ny, CLCommandQueue queue, boolean print){
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// Allocate buffers
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3, Mem.WRITE_ONLY);
		CLKernel kernel = program.createCLKernel("evaluateParzenLocal");


		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));

		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {geom.getReconDimensionX(), geom.getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}

		ArrayList<ArrayList<PointND>> inputCandidates = new ArrayList<ArrayList<PointND>>();
		ArrayList<ArrayList<PointND>> outputCandidates = new ArrayList<ArrayList<PointND>>();

		int total = 0;
		double zMargin = (geom.getVoxelSpacingZ()*geom.getReconDimensionZ()*0.1) + 6 * sigma;

		for(int i=0; i < geom.getReconDimensionZ();i++){
			inputCandidates.add(new ArrayList<PointND>());
			outputCandidates.add(new ArrayList<PointND>());
			double zValue = i * geom.getVoxelSpacingZ() + geom.getOriginZ();
			for (int j=0; j < input.length; j++){
				double distance = Math.abs(input[j].get(2) - zValue);
				if (distance < zMargin){
					inputCandidates.get(i).add(input[j]);
					outputCandidates.get(i).add(output[j]);
				}
			}
		}

		long time = System.currentTimeMillis();
		double stepSizeX =  (geom.getReconDimensionX() * geom.getVoxelSpacingX()) / (nx);
		double stepSizeY =  (geom.getReconDimensionY() * geom.getVoxelSpacingY()) / (ny);
		
		// set all static kernel parameters
		kernel
		.setArg(4,motionField)
		.setArg(5,geom.getReconDimensionX())
		.setArg(6,geom.getReconDimensionY())
		.setArg(7,nx*ny)
		.setArg(9,(float) sigma)
		.setArg(10,(float) geom.getVoxelSpacingX())
		.setArg(11,(float) geom.getVoxelSpacingY())
		.setArg(12,(float) geom.getVoxelSpacingZ())
		.setArg(13,(float) geom.getOriginX())
		.setArg(14,(float) geom.getOriginY())
		.setArg(15,(float) geom.getOriginZ());
		
		for(int i=0; i < geom.getReconDimensionZ();i++){
			double zValue = i * geom.getVoxelSpacingZ() + geom.getOriginZ();
			float [] searchCandidates = new float[nx*ny*3];
			int [] searchIndicies = new int[nx*ny*2];
			ArrayList<PointND> localList = new ArrayList<PointND>();
			ArrayList<PointND> localList2 = new ArrayList<PointND>();
			for (int x = 0; x < nx; x++){
				for (int y = 0; y < ny; y++){
					int idx = y*nx+x;
					searchCandidates[idx*3] = (float) ((stepSizeX / 2.0) + stepSizeX*x);
					searchCandidates[idx*3+1] = (float) ((stepSizeY / 2.0) + stepSizeY*y);
					searchCandidates[idx*3+2] = (float) zValue;
					searchIndicies[idx*2] = localList.size();
					int count = 0;
					for (int k = 0 ; k<inputCandidates.get(i).size(); k++){
						// compute euclidean distance;
						double distance = Math.pow(searchCandidates[idx*3] - inputCandidates.get(i).get(k).get(0),2);
						distance += Math.pow(searchCandidates[idx*3+1] - inputCandidates.get(i).get(k).get(1),2);
						distance += Math.pow(searchCandidates[idx*3+2] - inputCandidates.get(i).get(k).get(2),2);
						distance = Math.sqrt(distance);
						if (distance < 2 * stepSizeX + sigma * 6){
							localList.add(inputCandidates.get(i).get(k));
							localList2.add(outputCandidates.get(i).get(k));
							count++;
						}
					}
					searchIndicies[idx*2+1] = count;
				}
			}



			if (localList.size() == 0) continue;
			CLBuffer<FloatBuffer> search = context.createFloatBuffer(searchCandidates.length, Mem.READ_ONLY);
			CLBuffer<IntBuffer> searchIdx = context.createIntBuffer(searchIndicies.length, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> local = context.createFloatBuffer(localList.size() * 3, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> local2 = context.createFloatBuffer(localList2.size() * 3, Mem.READ_ONLY);
			fillBuffer(search, searchCandidates);
			fillBuffer(searchIdx, searchIndicies);
			fillBuffer(local, localList);
			fillBuffer(local2, localList2);


			queue.putWriteBuffer(search, true)
			.putWriteBuffer(searchIdx, true)
			.putWriteBuffer(local, true)
			.putWriteBuffer(local2, true);

			total += localList.size();	
			if (print) System.out.println("Computing Slice " + i + " (" + inputCandidates.get(i).size() + " points, local size "+localList.size()+")");

			kernel
			.setArg(0,search)
			.setArg(1,searchIdx)
			.setArg(2,local)
			.setArg(3,local2)
			.setArg(8,i);

			queue.put2DRangeKernel(kernel, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
			.finish();

			search.release();
			searchIdx.release();
			local.release();
			local2.release();
		}

		if (print) System.out.println("Evaluation quota [0.1 %]: " + total*1000/(input.length*geom.getReconDimensionZ()*nx*ny));


		time = System.currentTimeMillis() - time;
		if (print) System.out.println("Kernel execution times: " + time);

		// release buffers
		kernel.release();
		return motionField;
	}

	
	public float [] getMotionFieldAsArrayRandomBallCover(double from, double to, int n){

		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND[] input = getRasterPoints(from);
		PointND[] output = getRasterPoints(to);

		// create command queue on device.
		CLCommandQueue queue = device.createCommandQueue();

		// Allocate buffers
		CLBuffer<FloatBuffer> motionField = context.createFloatBuffer(geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3, Mem.WRITE_ONLY);
		CLKernel kernel = program.createCLKernel("evaluateParzenLocal");

		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));

		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {geom.getReconDimensionX(), geom.getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}

		float [] searchCandidates = new float[n*3];
		int [] searchIndicies = new int[n*2];
		// Select Random Points
		for (int x = 0; x < n; x++){
			int idx = x;
			int random = (int)(Math.random() * input.length);
			searchCandidates[idx*3] = (float) input[random].get(0);
			searchCandidates[idx*3+1] = (float) input[random].get(1);
			searchCandidates[idx*3+2] = (float) input[random].get(2);
		}
		// Compute average distance estimate
		double avgDist = 0;
		for (int x = 0; x < 100; x++){
			double mindist = Double.MAX_VALUE;
			int x2 = (int) (Math.random() * n);
			for (int y = 0; y < n; y++){
				double dist = Math.sqrt(Math.pow(searchCandidates[x2*3] - searchCandidates[(y)*3],2)
						+Math.pow(searchCandidates[x2*3+1] - searchCandidates[(y)*3+1],2)
						+Math.pow(searchCandidates[x2*3+2] - searchCandidates[(y)*3+2],2));
				if (dist == 0) continue;
				mindist =Math.min(dist, mindist);
			}
			avgDist = Math.max(mindist, avgDist);
		}
		avgDist += Math.min(6*sigma, avgDist);
		System.out.println("Average Distance:" + avgDist + " Sigma: " + sigma);
		// Compute Point lists
		ArrayList<PointND> localList = new ArrayList<PointND>();
		ArrayList<PointND> localList2 = new ArrayList<PointND>();
		for (int x = 0; x < n; x++){
			int idx = x;
			searchIndicies[idx*2] = localList.size();
			int count = 0;
			for (int k = 0 ; k<input.length; k++){
				// compute euclidean distance;
				double distance = Math.pow(searchCandidates[idx*3] - input[k].get(0),2);
				distance += Math.pow(searchCandidates[idx*3+1] - input[k].get(1),2);
				distance += Math.pow(searchCandidates[idx*3+2] - input[k].get(2),2);
				distance = Math.sqrt(distance);
				// selection needs to be independent of sigma
				if (distance < avgDist){
					localList.add(input[k]);
					localList2.add(output[k]);
					count++;
				}
			}
			searchIndicies[idx*2+1] = count;
		}
		if (localList.size() == 0) return null;
		CLBuffer<FloatBuffer> search = context.createFloatBuffer(searchCandidates.length, Mem.READ_ONLY);
		CLBuffer<IntBuffer> searchIdx = context.createIntBuffer(searchIndicies.length, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> local = context.createFloatBuffer(localList.size() * 3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> local2 = context.createFloatBuffer(localList2.size() * 3, Mem.READ_ONLY);
		fillBuffer(search, searchCandidates);
		fillBuffer(searchIdx, searchIndicies);
		fillBuffer(local, localList);
		fillBuffer(local2, localList2);

		queue.putWriteBuffer(search, true)
		.putWriteBuffer(searchIdx, true)
		.putWriteBuffer(local, true)
		.putWriteBuffer(local2, true);

		long time = System.currentTimeMillis();

		// set static kernel variables
		kernel.rewind();
		kernel
		.putArg(search)
		.putArg(searchIdx)
		.putArg(local)
		.putArg(local2)
		.putArg(motionField)
		.putArg(geom.getReconDimensionX())
		.putArg(geom.getReconDimensionY())
		.putArg(n)
		.putArg(0)// dummy slice variable - will be overwritten inside loop
		.putArg((float) sigma)
		.putArg((float) geom.getVoxelSpacingX())
		.putArg((float) geom.getVoxelSpacingY())
		.putArg((float) geom.getVoxelSpacingZ())
		.putArg((float) geom.getOriginX())
		.putArg((float) geom.getOriginY())
		.putArg((float) geom.getOriginZ());
		
		for(int i=0; i < geom.getReconDimensionZ();i++){				

			System.out.println("Computing Slice " + i + " (" + input.length + " points, local size "+localList.size()+")");
			
			kernel.setArg(8, i);
			queue.put2DRangeKernel(kernel, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
			.finish();
		}
		double total = localList.size();
		System.out.println("Evaluation quota: " + total/(input.length*n));


		queue.putReadBuffer(motionField, true).finish();
		time = System.currentTimeMillis() - time;
		System.out.println("Kernel execution times: " + time);

		float [] result = new float [geom.getReconDimensionX()*geom.getReconDimensionY()*geom.getReconDimensionZ()*3];
		motionField.getBuffer().rewind();
		for(int i=0; i < result.length; i++){
			result[i] = motionField.getBuffer().get();
		}
		// release buffers
		search.release();
		searchIdx.release();
		local.release();
		local2.release();
		motionField.release();
		queue.release();
		kernel.release();
		return result;
	}

	/**
	 * @return the originalMotionField
	 */
	public ParzenWindowMotionField getOriginalMotionField() {
		return originalMotionField;
	}

	/**
	 * @param originalMotionField the originalMotionField to set
	 */
	public void setOriginalMotionField(ParzenWindowMotionField originalMotionField) {
		this.originalMotionField = originalMotionField;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */