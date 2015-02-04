package edu.stanford.rsl.conrad.opencl.GLCLDemo;

import java.awt.Component;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import javax.media.opengl.GL2;

/**
 * Utility class for interacting with a scene. Supports rotation and zoom around origin.
 * @author Michael Bien
 */
public class UserSceneInteraction {

    private float z = -20;
    private float rotx = 45;
    private float roty = 30;

    private Point dragstart;
    private enum MOUSE_MODE { DRAG_ROTATE, DRAG_ZOOM }
    private MOUSE_MODE dragmode = MOUSE_MODE.DRAG_ROTATE;


    public void init(Component component) {
        initMouseListeners(component);
    }

    private void initMouseListeners(Component component) {
        component.addMouseMotionListener(new MouseMotionAdapter() {

            @Override
            public void mouseDragged(MouseEvent e) {

                if (dragstart != null) {
                    switch (dragmode) {
                        case DRAG_ROTATE:
                            rotx += e.getY() - dragstart.getY();
                            roty += e.getX() - dragstart.getX();
                            break;
                        case DRAG_ZOOM:
                            z += (e.getY() - dragstart.getY()) / 5.0f;
                            break;
                    }
                }

                dragstart = e.getPoint();
            }
        });
        component.addMouseWheelListener(new MouseWheelListener() {

            public void mouseWheelMoved(MouseWheelEvent e) {
                z += e.getWheelRotation()*5;
            }

        });
        component.addMouseListener(new MouseAdapter() {

            @Override
            public void mousePressed(MouseEvent e) {
                switch (e.getButton()) {
                    case (MouseEvent.BUTTON1):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                    case (MouseEvent.BUTTON2):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                    case (MouseEvent.BUTTON3):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                switch (e.getButton()) {
                    case (MouseEvent.BUTTON1):
                        dragmode = MOUSE_MODE.DRAG_ZOOM;
                        break;
                    case (MouseEvent.BUTTON2):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                    case (MouseEvent.BUTTON3):
                        dragmode = MOUSE_MODE.DRAG_ROTATE;
                        break;
                }

                dragstart = null;
            }
        });
    }


    public void interact(GL2 gl) {
        gl.glTranslatef(0, 0, z);
        gl.glRotatef(rotx, 1f, 0f, 0f);
        gl.glRotatef(roty, 0f, 1.0f, 0f);
    }


}