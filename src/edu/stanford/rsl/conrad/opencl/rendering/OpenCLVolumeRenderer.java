package edu.stanford.rsl.conrad.opencl.rendering;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Timer;
import java.util.TimerTask;


import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImagePlus;
import ij.process.ImageProcessor;

public class OpenCLVolumeRenderer extends OpenCLTextureRendering {

	/**
	 * 
	 */
	private static final long serialVersionUID = 369749150640130620L;

	private boolean multiFrameMode = false;
	private ArrayList<CLImage3d<IntBuffer>> clImages;
	private byte[][] volumes;
	private int nFrames=1;
	private int time = 0;

	public OpenCLVolumeRenderer(byte[] volumeData, int sizeX, int sizeY,
			int sizeZ, boolean stereoMode) {
		super(volumeData, sizeX, sizeY, sizeZ, stereoMode);
		/*transferFunc = new float[]
				{ 
				0.0f, 0.0f, 0.0f, 0.0f, 
				1.0f, 1.0f, 1.0f, 1.0f, 
				0.0f, 0.0f, 0.0f, 0.0f,
				};
*/
		volumes = new byte[1][volumeData.length];
		volumes[0]=volumeData;
		clImages = new ArrayList<CLImage3d<IntBuffer>>(1);
		start();

	}

	private double [] minmax = null;

	public OpenCLVolumeRenderer(ImagePlus image){
		super (new byte[image.getWidth() * image.getHeight() * image.getStackSize() / image.getNFrames()], image.getWidth(), image.getHeight(), image.getStackSize() / image.getNFrames(), false);

		transferFunc = new float[]
				{ 
				0.0f, 0.0f, 0.0f, 0.0f, 
				1.0f, 1.0f, 1.0f, 1.0f, 
				0.0f, 0.0f, 0.0f, 0.0f,
				};
		
		minmax = ImageUtil.minAndMaxOfImageProcessor(image);
		minmax[1] *= 1.05;
		
		if (image.getNFrames() == 1)
			multiFrameMode = false;
		else
			multiFrameMode = true;


		nFrames = image.getNFrames();
		// Create the image array
		clImages = new ArrayList<CLImage3d<IntBuffer>>(image.getNFrames());
		// rewrite imagePlus container into nFrames single byte volumes
		volumes = new byte [image.getNFrames()][volumeSize[2]*volumeSize[1]*volumeSize[0]];
		for (int n = 0; n < image.getNFrames(); n++) {
			for (int k = 0; k < volumeSize[2]; k++) {
				ImageProcessor img = image.getStack().getProcessor((n*volumeSize[2])+k + 1);
				for (int i = 0; i < image.getWidth(); i++){
					for (int j = 0; j < image.getHeight(); j++){
						setVoxel(volumes[n], i,j,k, img.getPixelValue(i, j));
					}	
				}
			}
		}
		
		initialized = false;
		//System.out.println("init");
		frame.setTitle("Volume: " + image.getTitle());	



	}


	public void setVoxel(byte [] h_volume, int i, int j, int k, double value){
		h_volume[((((int)volumeSize[1] * k) + j) * (int)volumeSize[0]) + i] = (byte) (((value - minmax[0]) / (minmax[1] - minmax[0])) * 256);
	}

	/**
	 * Entry point for this sample.
	 * 
	 * @param args not used
	 */
	public static void main(String args[])
	{
		startSample("Bucky.raw", 32, 32, 32, false);

		// Other input files may be obtained from http://www.volvis.org
		//startSample("mri_ventricles.raw", 256, 256, 124, false);
		//startSample("backpack8.raw", 512, 512, 373, false);
		//startSample("foot.raw", 256, 256, 256, false);
	}

	/**
	 * Starts this sample with the data that is read from the file
	 * with the given name. The data is assumed to have the 
	 * specified dimensions.
	 * 
	 * @param fileName The name of the volume data file to load
	 * @param sizeX The size of the data set in X direction
	 * @param sizeY The size of the data set in Y direction
	 * @param sizeZ The size of the data set in Z direction
	 * @param stereoMode Whether stereo mode should be used
	 */
	private static void startSample(
			String fileName, final int sizeX, final int sizeY, 
			final int sizeZ, final boolean stereoMode)
	{
		// Try to read the specified file
		byte data[] = null;
		try
		{
			int size = sizeX * sizeY * sizeZ; 
			FileInputStream fis = new FileInputStream(fileName);
			data = new byte[size];
			fis.read(data);
			fis.close();
		}
		catch (IOException e)
		{
			System.err.println("Could not load input file");
			e.printStackTrace();
			return;
		}

		// Start the sample with the data that was read from the file
		final byte volumeData[] = data;
		SwingUtilities.invokeLater(new Runnable()
		{
			public void run()
			{
				new OpenCLVolumeRenderer(
						volumeData, sizeX, sizeY, sizeZ, stereoMode);
			}
		});
	}

	/**
	 * Stops the animator and calls System.exit() in a new Thread.
	 * (System.exit() may not be called synchronously inside one 
	 * of the JOGL callbacks)
	 */
	protected void runExit()
	{
		if (pbo != 0)
		{
			unload();
			//gl.glDeleteBuffers(1, new int[]{ pbo }, 0);
			pbo = 0;
		}
		new Thread(new Runnable()
		{
			public void run()
			{
				animatorL.stop();
				if (animatorR != null)
				{
					animatorR.stop();
				}

				//System.exit(0);
			}
		}).start();
	}

	private Timer playTimer = null;
	private long frameDuration = 37;

	@Override
	protected JPanel createControlPanel(){
		JPanel controlPanel = super.createControlPanel();
		if (multiFrameMode) {
			JPanel panel = null;
			JSlider slider = null;
			// Time
			panel = new JPanel(new GridLayout(1, 2));
			panel.add(new JLabel("Time:"));
			slider = new JSlider(0, nFrames-1, 0);
			slider.addChangeListener(new ChangeListener()
			{
				public void stateChanged(ChangeEvent e)
				{
					JSlider source = (JSlider) e.getSource();
					time = source.getValue();
				}
			});
			slider.setPreferredSize(new Dimension(0, 0));
			panel.add(slider);

			controlPanel.add(panel);
			JPanel buttons = new JPanel(new GridLayout(1,3));


			JButton animateButton = new JButton("play");
			animateButton.addActionListener(new ActionListener(){

				public void actionPerformed(ActionEvent arg0) {
					if (playTimer == null){
						playTimer = new Timer();
						playTimer.scheduleAtFixedRate(new TimerTask(){
							@Override
							public void run() {
								time = (time +1) % nFrames;
							}}, 0, frameDuration);
					} else {
						playTimer.cancel();
						playTimer = null;
					}

				}
			}
					);
			buttons.add(animateButton);
			JButton button = new JButton("-");
			button.addActionListener(new ActionListener(){
				public void actionPerformed(ActionEvent arg0) {
					frameDuration +=10;
					if (frameDuration > 1000) frameDuration = 1000;
					resetTimer();
				}
			}
					);
			buttons.add(button);
			button = new JButton("+");
			button.addActionListener(new ActionListener(){
				public void actionPerformed(ActionEvent arg0) {
					frameDuration -=10;
					if (frameDuration < 10) frameDuration = 10;
					resetTimer();
				}
			}
					);
			buttons.add(button);
			controlPanel.add(buttons);
		}
		return controlPanel;
	}

	public void resetTimer(){
		if (playTimer!=null){
			playTimer.cancel();
			playTimer = new Timer();
			playTimer.scheduleAtFixedRate(new TimerTask(){
				@Override
				public void run() {
					time = (time +1) % nFrames;
				}}, 0, frameDuration);
		}
	}


	@Override
	protected void initVolumeBuffer(){

		CLImageFormat format = new CLImageFormat(ChannelOrder.RGBA, ChannelType.UNORM_INT8);
		for (int i = 0; i < nFrames; i++) {
			hvolumeBuffer = glCtx.createIntBuffer(volumes[i].length, Mem.READ_ONLY);
			for (int j = 0; j < h_volume.length; j++) {
				hvolumeBuffer.getBuffer().put(j, (int)volumes[i][j]);
			}
			hvolumeBuffer.getBuffer().rewind();


			clImages.add(glCtx.createImage3d(hvolumeBuffer.getBuffer(),
					volumeSize[0], volumeSize[1], volumeSize[2],
					format, Mem.READ_ONLY, Mem.ALLOCATE_BUFFER));
			hvolumeBuffer.release();

			try {
				commandQueue.putWriteImage(clImages.get(i), true)
				.finish();
			} catch (Exception e) {
				e.printStackTrace();
				unload();
			}
		}
		hvolumeBuffer = null;
	}


	@Override
	protected void setKernelParameters(){
		// Set up the execution parameters for the kernel:
		// - One pointer for the output that is mapped to the PBO
		// - Two ints for the width and height of the image to render
		// - Four floats for the visualization parameters of the renderer
		kernelFunction.rewind()
		.putArg(d_output)
		.putArg(width)
		.putArg(height)
		.putArg(density)
		.putArg(brightness)
		.putArg(transferOffset)
		.putArg(transferScale)
		.putArg(invViewMatrix)
		.putArg(clImages.get(time))
		.putArg(transferTex)
		.rewind();
	}
	
	
	/**
	 * release all CL related objects and free memory
	 */
	@Override
	protected void unload(){
		if (commandQueue != null)
			commandQueue.release();
		//release all buffers
		if (clImages != null){
			Iterator<CLImage3d<IntBuffer>> it = clImages.iterator();
			while (it.hasNext()) {
				it.next().release();
			}
		}
		if (tex != null)
			tex.release();
		if (transferTex != null)
			transferTex.release();
		if (invViewMatrix != null)
			invViewMatrix.release();
		if (d_output != null)
			d_output.release();
		if (hvolumeBuffer != null)
			hvolumeBuffer.release();
		if (kernelFunction != null)
			kernelFunction.release();
		if (program != null)
			program.release();
		if (glCtx != null)
			glCtx.release();
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
