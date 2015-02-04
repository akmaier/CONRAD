package edu.stanford.rsl.conrad.opencl.GLCLDemo;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opengl.util.Animator;


import java.io.IOException;
import javax.media.opengl.DebugGL2;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.media.opengl.glu.gl2.GLUgl2;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import static com.jogamp.common.nio.Buffers.*;

/**
 * JOCL - JOGL interoperability example.
 * @author Michael Bien
 */
public class GLCLInteroperabilityDemo implements GLEventListener {

    private final GLUgl2 glu = new GLUgl2();

    private final int MESH_SIZE = 512;

    private int width;
    private int height;

//    private final FloatBuffer vb;
//    private final IntBuffer ib;

    private final int[] glObjects = new int[2];
    private final int VERTICES = 0;
//    private final int INDICES  = 1;

    private final UserSceneInteraction usi;

    private CLGLContext clContext;
    private CLKernel kernel;
    private CLCommandQueue commandQueue;
    private CLGLBuffer<?> clBuffer;

    private float step = 0;

    public GLCLInteroperabilityDemo() {

        this.usi = new UserSceneInteraction();

        // create direct memory buffers
//        vb = newFloatBuffer(MESH_SIZE * MESH_SIZE * 4);
//        ib = newIntBuffer((MESH_SIZE - 1) * (MESH_SIZE - 1) * 2 * 3);
//
//        // build indices
//        //    0---3
//        //    | \ |
//        //    1---2
//        for (int h = 0; h < MESH_SIZE - 1; h++) {
//            for (int w = 0; w < MESH_SIZE - 1; w++) {
//
//                // 0 - 3 - 2
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6,      w + (h    ) * (MESH_SIZE)    );
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 1,  w + (h    ) * (MESH_SIZE) + 1);
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 2,  w + (h + 1) * (MESH_SIZE) + 1);
//
//                // 0 - 2 - 1
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 3,  w + (h    ) * (MESH_SIZE)    );
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 4,  w + (h + 1) * (MESH_SIZE) + 1);
//                ib.put(w * 6 + h * (MESH_SIZE - 1) * 6 + 5,  w + (h + 1) * (MESH_SIZE)    );
//
//            }
//        }
//        ib.rewind();

        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                initUI();
            }
        });

    }

    private void initUI() {

        this.width  = 600;
        this.height = 400;

        GLCapabilities config = new GLCapabilities(GLProfile.get(GLProfile.GL2));
        config.setSampleBuffers(true);
        config.setNumSamples(4);

        GLCanvas canvas = new GLCanvas(config);
        canvas.addGLEventListener(this);
        usi.init(canvas);

        JFrame frame = new JFrame("JOGL-JOCL Interoperability Example");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(canvas);
        frame.setSize(width, height);

        frame.setVisible(true);

    }


    @Override
    public void init(GLAutoDrawable drawable) {

        if(clContext == null) {

            // find gl compatible device
            CLDevice[] devices = CLPlatform.getDefault().listCLDevices();
            CLDevice device = null;
            for (CLDevice d : devices) {
                if(d.isGLMemorySharingSupported()) {
                    device = d;
                    break;
                }
            }
            if(null==device) {
                throw new RuntimeException("couldn't find any CL/GL memory sharing devices ..");
            }
            // create OpenCL context before creating any OpenGL objects
            // you want to share with OpenCL (AMD driver requirement)
            clContext = CLGLContext.create(drawable.getContext(), device);

            // enable GL error checking using the composable pipeline
            drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));

            // OpenGL initialization
            GL2 gl = drawable.getGL().getGL2();

            gl.setSwapInterval(1);

            gl.glPolygonMode(GL2.GL_FRONT_AND_BACK, GL2.GL_LINE);

            gl.glGenBuffers(glObjects.length, glObjects, 0);

    //        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, glObjects[INDICES]);
    //        gl.glBufferData(GL2.GL_ELEMENT_ARRAY_BUFFER, ib.capacity() * SIZEOF_INT, ib, GL2.GL_STATIC_DRAW);
    //        gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, 0);

            final int bsz = MESH_SIZE * MESH_SIZE * 4 * SIZEOF_FLOAT;
            gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
                gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjects[VERTICES]);
                gl.glBufferData(GL2.GL_ARRAY_BUFFER, bsz, null, GL2.GL_DYNAMIC_DRAW);
                gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);
            gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);                       

            pushPerspectiveView(gl);
            gl.glFinish();

            // init OpenCL
            initCL(gl, bsz);

            // start rendering thread
            Animator animator = new Animator(drawable);
            animator.start();

        }
    }

    private void initCL(GL2 gl, int bufferSize) {

        CLProgram program;
        try {
            program = clContext.createProgram(getClass().getResourceAsStream("JoglInterop.cl"));
            program.build();
            System.out.println(program.getBuildStatus());
            System.out.println(program.isExecutable());
            System.out.println(program.getBuildLog());
        } catch (IOException ex) {
            throw new RuntimeException("can not handle exception", ex);
        }

        commandQueue = clContext.getMaxFlopsDevice().createCommandQueue();

        clBuffer = clContext.createFromGLBuffer(glObjects[VERTICES], 
                                                bufferSize /* gl.glGetBufferSize(glObjects[VERTICES]*/, 
                                                CLGLBuffer.Mem.WRITE_ONLY);

        System.out.println("cl buffer type: " + clBuffer.getGLObjectType());
        System.out.println("shared with gl buffer: " + clBuffer.getGLObjectID());

        kernel = program.createCLKernel("sineWave")
                        .putArg(clBuffer)
                        .putArg(MESH_SIZE)
                        .rewind();

        System.out.println("cl initialised");
    }


    @Override
    public void display(GLAutoDrawable drawable) {

        GL2 gl = drawable.getGL().getGL2();

        // ensure pipeline is clean before doing cl work
        gl.glFinish();

        computeHeightfield();

        gl.glClear(GL2.GL_COLOR_BUFFER_BIT | GL2.GL_DEPTH_BUFFER_BIT);
        gl.glLoadIdentity();

        usi.interact(gl);

        gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, glObjects[VERTICES]);
        gl.glVertexPointer(4, GL2.GL_FLOAT, 0, 0);

//            gl.glBindBuffer(GL2.GL_ELEMENT_ARRAY_BUFFER, glObjects[INDICES]);

        gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
        gl.glDrawArrays(GL2.GL_POINTS, 0, MESH_SIZE * MESH_SIZE);
//            gl.glDrawElements(GL2.GL_TRIANGLES, ib.capacity(), GL2.GL_UNSIGNED_INT, 0);
        gl.glDisableClientState(GL2.GL_VERTEX_ARRAY);

//            gl.glBindBuffer(GL2.GL_ARRAY_BUFFER, 0);

    }

    /*
     * Computes a heightfield using a OpenCL kernel.
     */
    private void computeHeightfield() {

        kernel.setArg(2, step += 0.05f);

        commandQueue.putAcquireGLObject(clBuffer)
                    .put2DRangeKernel(kernel, 0, 0, MESH_SIZE, MESH_SIZE, 0, 0)
                    .putReleaseGLObject(clBuffer)
                    .finish();

    }

    private void pushPerspectiveView(GL2 gl) {

        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glPushMatrix();

            gl.glLoadIdentity();

            glu.gluPerspective(60, width / (float)height, 1, 1000);
            gl.glMatrixMode(GL2.GL_MODELVIEW);

            gl.glPushMatrix();
                gl.glLoadIdentity();

    }

    private void popView(GL2 gl) {

                gl.glMatrixMode(GL2.GL_PROJECTION);
            gl.glPopMatrix();

            gl.glMatrixMode(GL2.GL_MODELVIEW);
        gl.glPopMatrix();

    }


    @Override
    public void reshape(GLAutoDrawable drawable, int arg1, int arg2, int width, int height) {
        this.width = width;
        this.height = height;
        GL2 gl = drawable.getGL().getGL2();
        popView(gl);
        pushPerspectiveView(gl);
    }

    @Override
    public void dispose(GLAutoDrawable drawable) {  }

    public static void main(String[] args) {
        GLProfile.initSingleton();
        
        new GLCLInteroperabilityDemo();
    }

}
