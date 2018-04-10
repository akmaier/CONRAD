/*
 * Copyright (C) 2017 Andreas Maier, Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.Color;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;


public final class PlotHelper {
	
	private static int FRAME_ID = 0;
	
	private PlotHelper() {
		
	}
	/**
	 * draws a 2D plot
	 * @param xs x-coordiantes 
	 * @param ys y-coordinates
	 */
	public static void plot2D(double [] xs, double [] ys) {
		JFrame frame = createFrame("Plot 2D");
		Plot2DPanel panel = new Plot2DPanel();
		panel.addLinePlot("plot", xs, ys);
		frame.getContentPane().add(panel);
		frame.setVisible(true);
	}
	/**
	 * draws a 3D plot
	 * @param xs x-coordinates
	 * @param ys y-coordinates
	 * @param zs z-coordinates
	 */
	public static void plot3D(double [] xs, double [] ys, double [][] zs) {
		JFrame frame = createFrame("Plot 3D");
		Plot3DPanel panel = new Plot3DPanel();
		panel.addGridPlot("plot", Color.BLUE, xs, ys, zs);
		frame.getContentPane().add(panel);
		frame.setVisible(true);
	}
	
	/**
	 * creates a new JFrame 
	 * @param title title of the frame
	 * @return the JFrame
	 */
	private static JFrame createFrame(String title) {
		JFrame frame = new JFrame(title + " (" + FRAME_ID++ + ")");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLocationRelativeTo(null);
		frame.setSize(500, 400);
		return frame;
	}
	
}
