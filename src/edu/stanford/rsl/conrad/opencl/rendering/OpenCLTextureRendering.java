package edu.stanford.rsl.conrad.opencl.rendering;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * DISCLAIMER: THIS SOFTWARE IS PROVIDED WITHOUT WARRANTY OF ANY KIND
 * If you find any bugs or errors, contact me at http://www.jcuda.org
 *
 * LICENSE: THIS SOFTWARE IS FREE FOR NON-COMMERCIAL USE ONLY
 * For non-commercial applications, you may use this software without
 * any restrictions. If you wish to use it for commercial purposes,
 * contact me at http://www.jcuda.org
 */

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import javax.media.opengl.*;
import javax.media.opengl.awt.GLJPanel;
import javax.swing.*;
import javax.swing.event.*;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;
import com.jogamp.opengl.util.Animator;
import com.jogamp.opengl.util.awt.TextRenderer;

import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * A sample illustrating how to use textures with JCuda. This program uses
 * the CUBIN file that is created by the "volumeRender" program from the
 * NVIDIA CUDA samples web site. <br />
 * <br />
 * The program loads an 8 bit RAW volume data set and copies it into a 
 * 3D texture. The texture is accessed by the kernel to render an image
 * of the volume data. The resulting image is written into a pixel 
 * buffer object (PBO) which is then displayed using JOGL.
 */
public class OpenCLTextureRendering extends JFrame implements GLEventListener, MouseControlable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6181935230597653012L;



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
		//SwingUtilities.invokeLater(new Runnable()
		//{
			//public void run()
			//{
				OpenCLTextureRendering sample = new OpenCLTextureRendering(
						volumeData, sizeX, sizeY, sizeZ, stereoMode);
				sample.start();
			//}
		//});
	}

	protected float transferFunc[] = new float[]
	                                           { 
			0.0f, 0.0f, 0.0f, 0.0f, 
			1.0f, 0.0f, 0.0f, 1.0f, 
			1.0f, 0.5f, 0.0f, 1.0f, 
			1.0f, 1.0f, 0.0f, 1.0f, 
			0.0f, 1.0f, 0.0f, 1.0f,
			0.0f, 1.0f, 1.0f, 1.0f, 
			0.0f, 0.0f, 1.0f, 1.0f, 
			1.0f, 0.0f, 1.0f, 1.0f, 
			0.0f, 0.0f, 0.0f, 0.0f
	                                           };

	/**
	 * Whether the initialization method of this GLEventListener has already
	 * been called
	 */
	protected boolean initialized = false;

	/**
	 * Whether the stereo mode is enabled
	 */
	private boolean stereoMode = false;

	/**
	 * The GL component which is used for rendering
	 */
	private GLJPanel glComponentL;

	/**
	 * The GL component which is used for the other eye
	 */
	private GLJPanel glComponentR;

	/**
	 * Text renderer for status messages
	 */
	protected TextRenderer renderer;

	/**
	 * The animator used for rendering
	 */
	Animator animatorL;

	/**
	 * The animator for the other eye
	 */
	Animator animatorR;

	/**
	 * The OpenCL context
	 */
	protected CLGLContext glCtx;

	/**
	 * The OpenCL program
	 */
	protected CLProgram program;

	/**
	 * The OpenCL device
	 */
	private CLDevice device;
	
	/**
	 * The OpenCL kernel function binding
	 */
	protected CLKernel kernelFunction;

	/**
	 * The OpenCL command queue
	 */
	protected CLCommandQueue commandQueue;	

	/**
	 * The 3D volume texture reference
	 */
	protected CLImage3d<IntBuffer> tex = null;
	
	/**
	 * The 1D transfer texture reference
	 */
	protected CLImage2d<FloatBuffer> transferTex = null;

	
	protected CLBuffer<IntBuffer> hvolumeBuffer = null;
	
	protected CLBuffer<FloatBuffer> transferFctBuffer =null;
	
	/**
	 * The volume data that is to be rendered
	 */
	protected byte[] h_volume;
	
	/**
	 * The inverted view matrix, which will be copied to the global
	 * variable of the kernel.
	 */
	protected CLBuffer<FloatBuffer> invViewMatrix;
	
	/**
	 * The output of CL
	 */
	protected CLGLBuffer<?> d_output;
	
	/**
	 * The width of the rendered area and the PBO
	 */
	protected int width = 0;

	/**
	 * The height of the rendered area and the PBO
	 */
	protected int height = 0;

	/**
	 * The size of the volume data that is to be rendered
	 */
	protected int[] volumeSize = null;

	/**
	 * The block size for the kernel execution
	 */
	protected static int blockSize[] = {16, 16};
	
	/**
	 * The block size for the kernel execution
	 */
	protected float[] gridSize;

	/**
	 * The density of the rendered volume data
	 */
	protected float density = 0.05f;

	/**
	 * The brightness of the rendered volume data
	 */
	protected float brightness = 1.0f;

	/**
	 * The transferOffset of the rendered volume data
	 */
	protected float transferOffset = 0.0f;

	/**
	 * The transferScale of the rendered volume data
	 */
	protected float transferScale = 1.0f;

	/**
	 * The OpenGL pixel buffer object
	 */
	int pbo = 0;

	/**
	 * The translation in X-direction
	 */
	private float translationX = 0;

	/**
	 * The translation in Y-direction
	 */
	private float translationY = 0;

	/**
	 * The translation in Z-direction
	 */
	private float translationZ = -4;

	/**
	 * The rotation about the X-axis, in degrees
	 */
	private float rotationX = 0;

	protected JFrame frame = null;

	/**
	 * The rotation about the Y-axis, in degrees
	 */
	private float rotationY = 0;

	/**
	 * The System.nanoTime() of the previous rendered frame.
	 */
	private long prevFrameNanoTime = 0;



	/**
	 * Creates a new JCudaTextureSample that displays the given volume
	 * data, which has the specified size.
	 * 
	 * @param volumeData The volume data
	 * @param sizeX The size of the data set in X direction
	 * @param sizeY The size of the data set in Y direction
	 * @param sizeZ The size of the data set in Z direction
	 * @param stereoMode Whether stereo mode should be used
	 */
	public OpenCLTextureRendering(
			byte volumeData[], int sizeX, int sizeY, int sizeZ, boolean stereoMode)
	{
		volumeSize = new int[3];
		h_volume = volumeData;
		volumeSize[0] = sizeX;
		volumeSize[1] = sizeY;
		volumeSize[2] = sizeZ;
		

		this.stereoMode = stereoMode;
		if (stereoMode)
		{
			width = 280;
			height = 280;
		}
		else
		{
			width = 800;
			height = 800;
		}



		// Create the main frame
		frame = new JFrame("OpenCL 3D texture volume rendering");

		//start();

	}

	public void start(){

		// Initialize the GL component 
		glComponentL = new GLJPanel();
		glComponentL.addGLEventListener(this);
		if (stereoMode)
		{
			glComponentR = new GLJPanel();
			glComponentR.addGLEventListener(this);
		}

		// Initialize the mouse controls
		MouseControl mouseControl = new MouseControl(this);
		glComponentL.addMouseMotionListener(mouseControl);
		glComponentL.addMouseWheelListener(mouseControl);


		frame.addWindowListener(new WindowAdapter()
		{
			public void windowClosing(WindowEvent e)
			{
				runExit();
			}
		});
		frame.setLayout(new BorderLayout());
		glComponentL.setPreferredSize(new Dimension(width, height));
		JPanel p = new JPanel(new GridLayout(1,1));
		p.add(glComponentL);
		if (stereoMode)
		{
			p.setLayout(new GridLayout(1,2));
			p.add(glComponentR);
		}
		frame.add(p, BorderLayout.CENTER);
		frame.add(this.createControlPanel(), BorderLayout.SOUTH);
		frame.pack();
		frame.setVisible(true);

		// Create and start the animator
		boolean animate = true;
		if (animate) {
			animatorL = new Animator(glComponentL);
			animatorL.setRunAsFastAsPossible(true);
			animatorL.start();
			if (stereoMode)
			{
				animatorR = new Animator(glComponentR);
				animatorR.setRunAsFastAsPossible(true);
				animatorR.start();
			}
		}
	}

	/**
	 * Create the control panel containing the sliders for setting
	 * the visualization parameters.
	 * 
	 * @return The control panel
	 */
	protected JPanel createControlPanel()
	{
		JPanel controlPanel = new JPanel(new GridLayout(3, 2));
		JPanel panel = null;
		JSlider slider = null;

		// Density
		panel = new JPanel(new GridLayout(1, 2));
		panel.add(new JLabel("Density:"));
		slider = new JSlider(0, 100, 5);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 100.0f;
				density = a;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		// Brightness
		panel = new JPanel(new GridLayout(1, 2));
		panel.add(new JLabel("Brightness:"));
		slider = new JSlider(0, 100, 10);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 100.0f;
				brightness = a * 10;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		// Transfer offset
		panel = new JPanel(new GridLayout(1, 2));
		panel.add(new JLabel("Transfer Offset:"));
		slider = new JSlider(0, 1000, 550);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 1000.0f;
				transferOffset = (-0.5f + a) * 2;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		// Transfer scale
		panel = new JPanel(new GridLayout(1, 2));
		panel.add(new JLabel("Transfer Scale:"));
		slider = new JSlider(0, 1000, 100);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 1000.0f;
				transferScale = a * 5;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		return controlPanel;
	}


	
	
	/**
	 * Implementation of GLEventListener: Called to initialize the
	 * GLAutoDrawable. This method will initialize the JCudaDriver
	 * and cause the initialization of CUDA and the OpenGL PBO.
	 */
	public void init(GLAutoDrawable drawable)
	{
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

		try {
			// find gl compatible device
	        device = OpenCLUtil.getStaticContext().getMaxFlopsDevice();
	        // create OpenCL context before creating any OpenGL objects
	        // you want to share with OpenCL (AMD driver requirement)
	        glCtx = CLGLContext.create(drawable.getContext(), device);

	        // create the program
	        program = glCtx.createProgram(getClass().getResourceAsStream("volumeRender.cl")).build();
	        System.out.println(program.getBuildStatus());
	        System.out.println(program.isExecutable());
	        System.out.println(program.getBuildLog());
	        
	        commandQueue = device.createCommandQueue();
	        
	        kernelFunction = program.createCLKernel("d_render");
	        
		} catch (Exception e) {
			e.printStackTrace();
			unload();
		}

		// Initialize CUDA with the current volume data
		initCL();

		// Initialize the OpenGL pixel buffer object
		initPBO(gl);
		
		initialized = true;
	}

	
	void initVolumeBuffer(){
		hvolumeBuffer = glCtx.createIntBuffer(h_volume.length, Mem.READ_ONLY);
		for (int i = 0; i < h_volume.length; i++) {
			hvolumeBuffer.getBuffer().put(i, (int)h_volume[i]);
		}
		hvolumeBuffer.getBuffer().rewind();
		
		CLImageFormat format = new CLImageFormat(ChannelOrder.RGBA, ChannelType.UNORM_INT8);
		tex = glCtx.createImage3d(hvolumeBuffer.getBuffer(), volumeSize[0], volumeSize[1], volumeSize[2], format, Mem.READ_ONLY, Mem.ALLOCATE_BUFFER); 		
		hvolumeBuffer.release();
		
		try {
			commandQueue.putWriteImage(tex, true)
			.finish();
		} catch (Exception e) {
			e.printStackTrace();
			unload();
		}
		
		hvolumeBuffer = null;
	}
	
	/**
	 * Initialize CUDA and the 3D texture with the current volume data.
	 */
	void initCL()
	{
		
		initVolumeBuffer();
		
		transferFctBuffer = glCtx.createFloatBuffer(transferFunc.length, Mem.READ_ONLY);
		transferFctBuffer.getBuffer().put(transferFunc);
		transferFctBuffer.getBuffer().rewind();
		
		CLImageFormat formatTransFct = new CLImageFormat(ChannelOrder.RGBA, ChannelType.FLOAT);
		transferTex = glCtx.createImage2d(transferFctBuffer.getBuffer(), transferFunc.length/4, 1, formatTransFct, Mem.READ_ONLY, Mem.ALLOCATE_BUFFER);
		transferFctBuffer.release();
		
		try {
			commandQueue.putWriteImage(transferTex, true)
			.finish();
		} catch (Exception e) {
			e.printStackTrace();
			unload();
		}
		
		invViewMatrix = glCtx.createFloatBuffer(12, Mem.READ_ONLY);

	}

	/**
	 * Creates a pixel buffer object (PBO) which stores the image that
	 * is created by the kernel, and which will later be rendered 
	 * by JOGL.
	 * 
	 * @param gl The GL context
	 */
	void initPBO(GL gl)
	{
		if (pbo != 0)
		{
			gl.glDeleteBuffers(1, new int[]{ pbo }, 0);
			pbo = 0;
		}

		// Create and bind a pixel buffer object with the current 
		// width and height of the rendering component.
		int pboArray[] = new int[1];
		gl.glGenBuffers(1, pboArray, 0);
		pbo = pboArray[0];
		gl.glBindBuffer(GL4bc.GL_PIXEL_UNPACK_BUFFER, pbo);
		gl.glBufferData(GL4bc.GL_PIXEL_UNPACK_BUFFER, 
				width * height * 4, null, GL.GL_DYNAMIC_DRAW);
		gl.glBindBuffer(GL4bc.GL_PIXEL_UNPACK_BUFFER, 0);
		
		
		d_output = glCtx.createFromGLBuffer(pbo, 
				width * height * 4, 	
				CLGLBuffer.Mem.WRITE_ONLY);
	}

	/**
	 * Integral division, rounding the result to the next highest integer.
	 * 
	 * @param a Dividend
	 * @param b Divisor
	 * @return a/b rounded to the next highest integer.
	 */
	int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	/**
	 * Set up a default view for the given GLAutoDrawable
	 * 
	 * @param drawable The GLAutoDrawable to set the view for
	 */
	protected void setupView(GLAutoDrawable drawable)
	{
		GL4bc gl = (GL4bc) drawable.getGL();

		gl.glViewport(0, 0, drawable.getSurfaceWidth(), drawable.getSurfaceHeight());

		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glMatrixMode(GL4bc.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	}

	/**
	 * Returns the given (address) value, adjusted to have
	 * the given alignment. In newer versions of JCuda, this
	 * function is also available as JCudaDriver#align
	 * 
	 * @param value The address value
	 * @param alignment The desired alignment
	 * @return The aligned address value
	 */
	protected static int align(int value, int alignment)
	{
		return (((value) + (alignment) - 1) & ~((alignment) - 1));		
	}

	
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
					  .putArg(tex)
					  .putArg(transferTex)
					  .rewind();
	}
	
	
	/**
	 * Call the kernel function, rendering the 3D volume data image
	 * into the PBO
	 */
	protected void render()
	{
		
		// Set up the execution parameters for the kernel:
		setKernelParameters();
		
		
		int[] realLocalSize = {Math.min(device.getMaxWorkGroupSize(),blockSize[0]), Math.min(device.getMaxWorkGroupSize(),blockSize[1])};
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {OpenCLUtil.roundUp(realLocalSize[0], width), OpenCLUtil.roundUp(realLocalSize[1], height)}; 


		try {
	        commandQueue.putAcquireGLObject(d_output)
	        .put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], 0, 0)
	        .putReleaseGLObject(d_output)
	        .finish();
		} catch (Exception e) {
			e.printStackTrace();
			unload();
		}
	}





	/**
	 * Implementation of GLEventListener: Called when the given GLAutoDrawable
	 * is to be displayed.
	 */
	public void display(GLAutoDrawable drawable)
	{
		if (!initialized)
		{
			return;
		}
		if (pbo == 0)
		{
			return;
		}

		GL4bc gl = (GL4bc) drawable.getGL();

		float eyeDelta = 0;
		if (stereoMode)
		{
			if (drawable == glComponentL)
			{
				eyeDelta = 4f;
			}
			else
			{
				eyeDelta = -4f;
			}
		}

	
		
		// Use OpenGL to build view matrix
		float modelView[] = new float[16];
		gl.glMatrixMode(GL4bc.GL_MODELVIEW);
		gl.glPushMatrix();
		gl.glLoadIdentity();
		gl.glRotatef(-rotationX, 1.0f, 0.0f, 0.0f);
		gl.glRotatef(-(rotationY+eyeDelta), 0.0f, 1.0f, 0.0f);
		gl.glTranslatef(-translationX, -translationY, -translationZ);
		gl.glGetFloatv(GL4bc.GL_MODELVIEW_MATRIX, modelView, 0);


		gl.glCullFace(GL.GL_FRONT_AND_BACK);



		gl.glPopMatrix();

		float[] loc_invViewMatrix = new float[12];
		// Build the inverted view matrix
		loc_invViewMatrix[0] = modelView[0];
		loc_invViewMatrix[1] = modelView[4];
		loc_invViewMatrix[2] = modelView[8];
		loc_invViewMatrix[3] = modelView[12];
		loc_invViewMatrix[4] = modelView[1];
		loc_invViewMatrix[5] = modelView[5];
		loc_invViewMatrix[6] = modelView[9];
		loc_invViewMatrix[7] = modelView[13];
		loc_invViewMatrix[8] = modelView[2];
		loc_invViewMatrix[9] = modelView[6];
		loc_invViewMatrix[10] = modelView[10];
		loc_invViewMatrix[11] = modelView[14];
		
		invViewMatrix.getBuffer().rewind();
		invViewMatrix.getBuffer().put(loc_invViewMatrix);
		invViewMatrix.getBuffer().rewind();
		
		// Copy the inverted view matrix to the global variable that
		// was obtained from the module. The inverted view matrix
		// will be used by the kernel during rendering.
		try {
			commandQueue.putWriteBuffer(invViewMatrix, true)
						.finish();
		} catch (Exception e) {
			e.printStackTrace();
			unload();
		}
		
		

		// Render and fill the PBO with pixel data
		render();

		// Draw the image from the PBO
		gl.glClear(GL.GL_COLOR_BUFFER_BIT);
		gl.glDisable(GL.GL_DEPTH_TEST);
		gl.glRasterPos2i(0, 0);
		gl.glBindBuffer(GL4bc.GL_PIXEL_UNPACK_BUFFER, pbo);
		gl.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 0);
		gl.glBindBuffer(GL4bc.GL_PIXEL_UNPACK_BUFFER, 0);

		if (!stereoMode)
		{
			// Compute FPS 
			long nanoTime = System.nanoTime();
			double frameTimeMs = (nanoTime - prevFrameNanoTime) / 1000000.0;
			prevFrameNanoTime = nanoTime;
			double fps = 1000.0 / frameTimeMs;
			String fpsString = String.format("%.2f", fps);

			// Print status message
			renderer.beginRendering(drawable.getSurfaceWidth(), drawable.getSurfaceHeight());
			renderer.setColor(1.0f, 1.0f, 1.0f, 0.5f);
			renderer.draw(fpsString + " fps", 20, 10);
			renderer.endRendering();
		}

	}
	
	/**
	 * release all CL related objects and free memory
	 */
	protected void unload(){
		if (commandQueue != null)
			commandQueue.release();
		//release all buffers
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
	

	/**
	 * Implementation of GLEventListener: Called then the GLAutoDrawable was
	 * reshaped
	 */
	public void reshape(
			GLAutoDrawable drawable, int x, int y, int width, int height)
	{
		this.width = width;
		this.height = height;

		initPBO(drawable.getGL());

		setupView(drawable);
	}

	/**
	 * Implementation of GLEventListener - not used
	 */
	public void displayChanged(
			GLAutoDrawable drawable, boolean modeChanged, boolean deviceChanged)
	{}

	/**
	 * Stops the animator and calls System.exit() in a new Thread.
	 * (System.exit() may not be called synchronously inside one 
	 * of the JOGL callbacks)
	 */
	protected void runExit()
	{
		
		new Thread(new Runnable()
		{
			public void run()
			{
				animatorL.stop();
				if (animatorR != null)
				{
					animatorR.stop();
				}
				unload();
				System.exit(0);
			}
		}).start();
	}


	public void updateRotationX(double increment) {
		this.rotationX += increment;
	}


	public void updateRotationY(double increment) {
		this.rotationY += increment;
	}


	public void updateTranslationX(double increment) {
		this.translationX += increment;
	}


	public void updateTranslationY(double increment) {
		this.translationY += increment;
	}


	public void updateTranslationZ(double increment) {
		this.translationZ += increment;
	}


	public void dispose(GLAutoDrawable arg0) {
		// TODO Auto-generated method stub
		
	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
