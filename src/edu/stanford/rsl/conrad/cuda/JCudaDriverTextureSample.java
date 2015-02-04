package edu.stanford.rsl.conrad.cuda;

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

import javax.media.opengl.*;
import javax.media.opengl.awt.GLJPanel;
import javax.swing.*;
import javax.swing.event.*;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.dim3;

import com.jogamp.opengl.util.Animator;
import com.jogamp.opengl.util.awt.TextRenderer;

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
public class JCudaDriverTextureSample extends JFrame implements GLEventListener, MouseControlable
{
	private static final long serialVersionUID = 3072546722050331601L;



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
				JCudaDriverTextureSample sample = new JCudaDriverTextureSample(
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
	 * The CUDA module containing the kernel
	 */
	protected CUmodule module = new CUmodule();

	/**
	 * The handle for the CUDA function of the kernel that is to be called
	 */
	protected CUfunction function;

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
	protected dim3 volumeSize = new dim3();

	/**
	 * The volume data that is to be rendered
	 */
	protected byte h_volume[];

	/**
	 * The block size for the kernel execution
	 */
	protected dim3 blockSize = new dim3(16, 16, 0);

	/**
	 * The grid size of the kernel execution
	 */
	protected dim3 gridSize = 
		new dim3(width / blockSize.x, height / blockSize.y, 1);

	/**
	 * The global variable of the module which stores the
	 * inverted view matrix.
	 */
	protected CUdeviceptr c_invViewMatrix = new CUdeviceptr();

	/**
	 * The inverted view matrix, which will be copied to the global
	 * variable of the kernel.
	 */
	private float invViewMatrix[] = new float[12];

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
	 * The 3D texture reference
	 */
	private CUtexref tex = new CUtexref();

	/**
	 * The 1D transfer texture reference
	 */
	protected CUtexref transferTex = new CUtexref();

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
	public JCudaDriverTextureSample(
			byte volumeData[], int sizeX, int sizeY, int sizeZ, boolean stereoMode)
	{
		h_volume = volumeData;
		volumeSize.x = sizeX;
		volumeSize.y = sizeY;
		volumeSize.z = sizeZ;

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
		frame = new JFrame("JCuda 3D texture volume rendering sample");

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
		frame.add(createControlPanel(), BorderLayout.SOUTH);
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
		slider = new JSlider(0, 100, 55);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 100.0f;
				transferOffset = (-0.5f + a) * 2;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		// Transfer scale
		panel = new JPanel(new GridLayout(1, 2));
		panel.add(new JLabel("Transfer Scale:"));
		slider = new JSlider(0, 100, 10);
		slider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e)
			{
				JSlider source = (JSlider) e.getSource();
				float a = source.getValue() / 100.0f;
				transferScale = a * 10;
			}
		});
		slider.setPreferredSize(new Dimension(0, 0));
		panel.add(slider);
		controlPanel.add(panel);

		return controlPanel;
	}

	CUcontext glCtx = null;

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
		"_Z8d_renderPjjjffff");

		// Initialize CUDA with the current volume data
		initCuda();

		// Initialize the OpenGL pixel buffer object
		initPBO(gl);

		initialized = true;
	}

	protected void fillTextureReference(CUtexref tex, byte [] volume){
		CUarray d_volumeArray = new CUarray();

		// Create the 3D array that will contain the volume data
		// and will be accessed via the 3D texture
		CUDA_ARRAY3D_DESCRIPTOR allocateArray = new CUDA_ARRAY3D_DESCRIPTOR();
		allocateArray.Width = volumeSize.x;
		allocateArray.Height = volumeSize.y;
		allocateArray.Depth = volumeSize.z;
		allocateArray.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
		allocateArray.NumChannels = 1;
		JCudaDriver.cuArray3DCreate(d_volumeArray, allocateArray);

		// Copy the volume data data to the 3D array
		CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
		copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
		copy.srcHost = Pointer.to(volume);
		copy.srcPitch = volumeSize.x;
		copy.srcHeight = volumeSize.y;
		copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
		copy.dstArray = d_volumeArray;
		copy.dstPitch = volumeSize.x;
		copy.dstHeight = volumeSize.y;
		copy.WidthInBytes = volumeSize.x;
		copy.Height = volumeSize.y;
		copy.Depth = volumeSize.z;
		JCudaDriver.cuMemcpy3D(copy);

		// Create the 3D texture reference for the volume data 
		// set its parameters 

		JCudaDriver.cuTexRefSetFilterMode(tex,
				CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
		JCudaDriver.cuTexRefSetAddressMode(tex, 0,
				CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
		JCudaDriver.cuTexRefSetAddressMode(tex, 1,
				CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
		JCudaDriver.cuTexRefSetFormat(tex,
				CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, 1);
		JCudaDriver.cuTexRefSetFlags(tex,
				JCudaDriver.CU_TRSF_NORMALIZED_COORDINATES);
		JCudaDriver.cuTexRefSetArray(tex, d_volumeArray,
				JCudaDriver.CU_TRSA_OVERRIDE_FORMAT);
	}

	/**
	 * Initialize CUDA and the 3D texture with the current volume data.
	 */
	void initCuda()
	{
		CUarray d_transferFuncArray = new CUarray();

		// Obtain the 3D texture reference for the volume data from 
		// the module, set its parameters and assign the 3D volume 
		// data array as its reference.
		JCudaDriver.cuModuleGetTexRef(tex, module, "tex");
		fillTextureReference(tex, h_volume);

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

		// Set the texture references as parameters for the function call
		JCudaDriver.cuParamSetTexRef(function, JCudaDriver.CU_PARAM_TR_DEFAULT,
				tex);
		JCudaDriver.cuParamSetTexRef(function, JCudaDriver.CU_PARAM_TR_DEFAULT,
				transferTex);
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
			JCudaDriver.cuGLUnregisterBufferObject(pbo);
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
				width * height * Sizeof.BYTE * 4, null, GL.GL_DYNAMIC_DRAW);
		gl.glBindBuffer(GL4bc.GL_PIXEL_UNPACK_BUFFER, 0);

		// Register the PBO for usage with CUDA
		JCudaDriver.cuGLRegisterBufferObject(pbo);

		// Calculate new grid size
		gridSize = new dim3(
				iDivUp(width, blockSize.x), 
				iDivUp(height, blockSize.y), 1);
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

	/**
	 * Call the kernel function, rendering the 3D volume data image
	 * into the PBO
	 */
	protected void render()
	{
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

		JCudaDriver.cuParamSetSize(function, offset);

		// Call the CUDA kernel, writing the results into the PBO
		JCudaDriver.cuFuncSetBlockShape(function, blockSize.x, blockSize.y, 1);
		JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
		JCudaDriver.cuCtxSynchronize();
		JCudaDriver.cuGLUnmapBufferObject(pbo);
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

		// Build the inverted view matrix
		invViewMatrix[0] = modelView[0];
		invViewMatrix[1] = modelView[4];
		invViewMatrix[2] = modelView[8];
		invViewMatrix[3] = modelView[12];
		invViewMatrix[4] = modelView[1];
		invViewMatrix[5] = modelView[5];
		invViewMatrix[6] = modelView[9];
		invViewMatrix[7] = modelView[13];
		invViewMatrix[8] = modelView[2];
		invViewMatrix[9] = modelView[6];
		invViewMatrix[10] = modelView[10];
		invViewMatrix[11] = modelView[14];

		// Copy the inverted view matrix to the global variable that
		// was obtained from the module. The inverted view matrix
		// will be used by the kernel during rendering.
		JCudaDriver.cuMemcpyHtoD(c_invViewMatrix, Pointer.to(invViewMatrix),
				invViewMatrix.length * Sizeof.FLOAT);

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

