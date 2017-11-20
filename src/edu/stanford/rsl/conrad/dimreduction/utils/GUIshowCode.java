package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.GridLayout;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class GUIshowCode extends JFrame{

	private static final long serialVersionUID = 1L;
	private JPanel panel; 
	private JLabel code; 
	
	public GUIshowCode(String code){
		super("Sammon");

		setSize(800, 600);
		setLocation(100, 100);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);
		
		this.code = new JLabel(code + "</html>");
		
		panel = new JPanel(new GridLayout(1,1));
		panel.add(this.code);
		getContentPane().add(panel);
		
	}

}
