package edu.stanford.rsl.conrad.cuda;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.Timer;
import java.util.TimerTask;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.jogamp.opengl.util.awt.TextRenderer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUaddress_mode;
import jcuda.driver.CUarray;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfilter_mode;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;

import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImagePlus;
import ij.process.ImageProcessor;

public class ImagePlusVolumeRenderer extends JCudaDriverTextureSample {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2120647421143268776L;
	private boolean multiFrameMode = false;
	private CUtexref [] textures;
	private byte [] [] volumes;
	private int nFrames=1;
	private int time = 0;

	public ImagePlusVolumeRenderer(byte[] volumeData, int sizeX, int sizeY,
			int sizeZ, boolean stereoMode) {
		super(volumeData, sizeX, sizeY, sizeZ, stereoMode);
		transferFunc = new float[]
		                         { 
				0.0f, 0.0f, 0.0f, 0.0f, 
				1.0f, 1.0f, 1.0f, 1.0f, 
				0.0f, 0.0f, 0.0f, 0.0f,
		                         };
		start();

	}

	private double [] minmax = null;

	public ImagePlusVolumeRenderer(ImagePlus image){
		super (new byte[image.getWidth() * image.getHeight() * image.getStackSize() / image.getNFrames()], image.getWidth(), image.getHeight(), image.getStackSize() / image.getNFrames(), false);

		transferFunc = new float[]
		                         { 
				0.0f, 0.0f, 0.0f, 0.0f, 
				1.0f, 1.0f, 1.0f, 1.0f, 
				0.0f, 0.0f, 0.0f, 0.0f,
		                         };
		minmax = ImageUtil.minAndMaxOfImageProcessor(image);
		minmax[1] *= 1.05;
		if (image.getNFrames() == 1){
			for (int k = 0; k < image.getStackSize(); k++) {
				ImageProcessor img = image.getStack().getProcessor(k + 1);
				for (int i = 0; i < image.getWidth(); i++){
					for (int j = 0; j < image.getHeight(); j++){
						setVoxel(h_volume, i,j,k, img.getPixelValue(i, j));
					}	
				}
			}
		} else {
			multiFrameMode = true;
			nFrames = image.getNFrames();
			// Allocate the texture references
			textures = new CUtexref[image.getNFrames()];
			for (int i = 0; i < image.getNFrames(); i++){
				textures[i] = new CUtexref();
			}
			// rewrite imagePlus container into nFrames single byte volumes
			volumes = new byte [image.getNFrames()][volumeSize.z*volumeSize.x*volumeSize.y];
			for (int n = 0; n < image.getNFrames(); n++) {
				for (int k = 0; k < volumeSize.z; k++) {
					ImageProcessor img = image.getStack().getProcessor((n*volumeSize.z)+k + 1);
					for (int i = 0; i < image.getWidth(); i++){
						for (int j = 0; j < image.getHeight(); j++){
							setVoxel(volumes[n], i,j,k, img.getPixelValue(i, j));
						}	
					}
				}
			}
		}
		initialized = false;
		//System.out.println("init");
		frame.setTitle("Volume: " + image.getTitle());	
	}

	public void setVoxel(byte [] h_volume, int i, int j, int k, double value){
		h_volume[(((volumeSize.y * k) + j) * volumeSize.x) + i] = (byte) (((value - minmax[0]) / (minmax[1] - minmax[0])) * 256);
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
				new ImagePlusVolumeRenderer(
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
			//JCudaDriver.cuGLUnregisterBufferObject(pbo);
			JCudaDriver.cuCtxDestroy(glCtx);
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
	public void init(GLAutoDrawable drawable)
	{
		if (multiFrameMode) {
			// Perform the default GL initialization
			GL gl = drawable.getGL();
			gl.setSwapInterval(0);
			gl.glEnable(GL.GL_DEPTH_TEST);
			gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			setupView(drawable);

			// Initialize the GL_ARB_pixel_buffer_object extension
			if (!gl.isExtensionAvailable("GL_ARB_pixel_buffer_object"))
			{
				new Thread(new Runnable()
				{
					public void run()
					{
						JOptionPane.showMessageDialog(null,
								"GL_ARB_pixel_buffer_object extension not available",
								"Unavailable extension", JOptionPane.ERROR_MESSAGE);
						runExit();
					}
				}).start();
			}

			// Create a TextRenderer for the status messages
			renderer = new TextRenderer(new Font("SansSerif", Font.PLAIN, 12));

			if (initialized)
			{
				return;
			}

			// Initialize the JCudaDriver. Note that this has to be done from 
			// the same thread that will later use the JCudaDriver API.
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			CUdevice dev = new CUdevice();
			JCudaDriver.cuDeviceGet(dev, 0);
			glCtx = new CUcontext();
			JCudaDriver.cuGLCtxCreate(glCtx, 0, dev);

			// Load the CUBIN file containing the kernel
			JCudaDriver.cuModuleLoad(module, "volumeRender_kernel.sm_10.cubin");

			// Obtain the global pointer to the inverted view matrix from 
			// the module
			JCudaDriver.cuModuleGetGlobal(c_invViewMatrix, new int[1], module,
			"c_invViewMatrix");

			// Obtain a function pointer to the kernel function. This function
			// will later be called in the display method of this 
			// GLEventListener.
			function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module,
			"_Z16d_render_texturePjjjffffi");

			// Initialize CUDA with the current volume data
			initCuda();

			// Initialize the OpenGL pixel buffer object
			initPBO(gl);

			initialized = true;
		} else {
			super.init(drawable);
		}
	}

	@Override
	void initCuda()
	{
		if (multiFrameMode) {
			for (int n = 0; n < nFrames; n++){
				NumberFormat nf = NumberFormat.getInstance();
				nf.setMaximumIntegerDigits(2);
				nf.setMinimumIntegerDigits(2);
				nf.setMinimumFractionDigits(0);
				nf.setMaximumFractionDigits(0);
				//System.out.println("Filling texture: tex" + nf.format(n));
				JCudaDriver.cuModuleGetTexRef(textures[n], module, "tex" + nf.format(n));
				fillTextureReference(textures[n], volumes[n]);
			}

			CUarray d_transferFuncArray = new CUarray();
			// The RGBA components of the transfer function texture 


			// Create the 2D (float4) array that will contain the
			// transfer function data. 
			CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
			ad.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
			ad.Width = transferFunc.length / 4;
			ad.Height = 1;
			ad.NumChannels = 4;
			JCudaDriver.cuArrayCreate(d_transferFuncArray, ad);

			// Copy the transfer function data to the array  
			CUDA_MEMCPY2D copy2 = new CUDA_MEMCPY2D();
			copy2.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
			copy2.srcHost = Pointer.to(transferFunc);
			copy2.srcPitch = transferFunc.length * Sizeof.FLOAT;
			copy2.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
			copy2.dstArray = d_transferFuncArray;
			copy2.WidthInBytes = transferFunc.length * Sizeof.FLOAT;
			copy2.Height = 1;
			JCudaDriver.cuMemcpy2D(copy2);

			// Obtain the transfer texture reference from the module, 
			// set its parameters and assign the transfer function  
			// array as its reference.
			JCudaDriver.cuModuleGetTexRef(transferTex, module, "transferTex");
			JCudaDriver.cuTexRefSetFilterMode(transferTex,
					CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
			JCudaDriver.cuTexRefSetAddressMode(transferTex, 0,
					CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
			JCudaDriver.cuTexRefSetFlags(transferTex,
					JCudaDriver.CU_TRSF_NORMALIZED_COORDINATES);
			JCudaDriver.cuTexRefSetFormat(transferTex,
					CUarray_format.CU_AD_FORMAT_FLOAT, 4);
			JCudaDriver.cuTexRefSetArray(transferTex, d_transferFuncArray,
					JCudaDriver.CU_TRSA_OVERRIDE_FORMAT);

			JCudaDriver.cuParamSetTexRef(function, JCudaDriver.CU_PARAM_TR_DEFAULT,
					transferTex);
		} else {
			super.initCuda();
		}
	}

	@SuppressWarnings("deprecation")
	@Override
	protected void render()
	{
		if (multiFrameMode) {
			// Map the PBO to get a CUDA device pointer
			CUdeviceptr d_output = new CUdeviceptr();
			JCudaDriver.cuGLMapBufferObject(d_output, new int[1], pbo);
			JCudaDriver.cuMemsetD32(d_output, 0, width * height);

			// Set up the execution parameters for the kernel:
			// - One pointer for the output that is mapped to the PBO
			// - Two ints for the width and height of the image to render
			// - Four floats for the visualization parameters of the renderer
			Pointer dOut = Pointer.to(d_output);
			Pointer pWidth = Pointer.to(new int[]{width});
			Pointer pHeight = Pointer.to(new int[]{height});
			Pointer pDensity = Pointer.to(new float[]{density});
			Pointer pBrightness = Pointer.to(new float[]{brightness});
			Pointer pTransferOffset = Pointer.to(new float[]{transferOffset});
			Pointer pTransferScale = Pointer.to(new float[]{transferScale});
			Pointer pTime = Pointer.to(new int[]{time});

			int offset = 0;

			offset = align(offset, Sizeof.POINTER);
			JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
			offset += Sizeof.POINTER;

			offset = align(offset, Sizeof.INT);
			JCudaDriver.cuParamSetv(function, offset, pWidth, Sizeof.INT);
			offset += Sizeof.INT;

			offset = align(offset, Sizeof.INT);
			JCudaDriver.cuParamSetv(function, offset, pHeight, Sizeof.INT);
			offset += Sizeof.INT;

			offset = align(offset, Sizeof.FLOAT);
			JCudaDriver.cuParamSetv(function, offset, pDensity, Sizeof.FLOAT);
			offset += Sizeof.FLOAT;

			offset = align(offset, Sizeof.FLOAT);
			JCudaDriver.cuParamSetv(function, offset, pBrightness, Sizeof.FLOAT);
			offset += Sizeof.FLOAT;

			offset = align(offset, Sizeof.FLOAT);
			JCudaDriver.cuParamSetv(function, offset, pTransferOffset, Sizeof.FLOAT);
			offset += Sizeof.FLOAT;

			offset = align(offset, Sizeof.FLOAT);
			JCudaDriver.cuParamSetv(function, offset, pTransferScale, Sizeof.FLOAT);
			offset += Sizeof.FLOAT;

			offset = align(offset, Sizeof.INT);
			JCudaDriver.cuParamSetv(function, offset, pTime, Sizeof.INT);
			offset += Sizeof.INT;

			JCudaDriver.cuParamSetSize(function, offset);

			// Call the CUDA kernel, writing the results into the PBO
			JCudaDriver.cuFuncSetBlockShape(function, blockSize.x, blockSize.y, 1);
			JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
			JCudaDriver.cuCtxSynchronize();
			JCudaDriver.cuGLUnmapBufferObject(pbo);
		} else {
			super.render();
		}
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
