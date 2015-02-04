package edu.stanford.rsl.apps.gui;

import java.awt.Dimension;

import javax.swing.JFrame;

import edu.stanford.rsl.conrad.utils.Configuration;

public class PhantomMaker extends JFrame {


	private static final long serialVersionUID = 4782596036459267832L;


	public PhantomMaker() {
		Configuration.loadConfiguration();
		setSize(new Dimension(1024,764));
		setTitle("CONRAD Phantom Maker");
		ConfigurationFrame configurationFrame = new ConfigurationFrame();
		configurationFrame.setVisible(true);
	}
	
	
	public static void main(String [] args){
		new PhantomMaker().setVisible(true);
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/