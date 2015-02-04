package edu.stanford.rsl.conrad.opencl.rendering;

import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

/**
 * Inner class encapsulating the MouseMotionListener and 
 * MouseWheelListener for the interaction
 */
public class MouseControl implements MouseMotionListener, MouseWheelListener
{
	private Point previousMousePosition = new Point();
	private MouseControlable controled;
	
	public MouseControl(MouseControlable con){
		this.controled = con;
	}
	
	
	public void mouseDragged(MouseEvent e)
	{
		int dx = e.getX() - previousMousePosition.x;
		int dy = e.getY() - previousMousePosition.y;

		// If the left button is held down, move the object
		if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) == 
			MouseEvent.BUTTON1_DOWN_MASK)
		{
			controled.updateRotationX(dy);
			controled.updateRotationY(dx);
		}

		// If the right button is held down, rotate the object
		else if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) == 
			MouseEvent.BUTTON3_DOWN_MASK)
		{

			controled.updateTranslationX(dx / 100.0f);
			controled.updateTranslationY(-( dy / 100.0f));
		}
		previousMousePosition = e.getPoint();
	}

	public void mouseMoved(MouseEvent e)
	{
		previousMousePosition = e.getPoint();
	}

	public void mouseWheelMoved(MouseWheelEvent e)
	{
		// Translate along the Z-axis
		controled.updateTranslationZ(e.getWheelRotation() * 0.25f);
		previousMousePosition = e.getPoint();
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/