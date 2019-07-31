package edu.stanford.rsl.BA_Niklas;


import java.awt.*;
import java.awt.event.*;

public class UserInterface extends Frame
{ 
  Button button = new Button("Dateidialog aufrufen");
  FileDialog fd;

  public UserInterface() 
  {		  
    setTitle("FileDialog-Beispiel"); 
    addWindowListener(new TestWindowListener());
   
    fd = new FileDialog(this, "Dateidialog");
  
    add(button);
   
    button.addActionListener(new ActionListener()
    {
      public void actionPerformed(ActionEvent e) 
      {			 	
        fd.setVisible(true);
      }    		
    });
     
    setSize(300,100);
    setVisible(true);                           
  }
 
  class TestWindowListener extends WindowAdapter
  {
    public void windowClosing(WindowEvent e)
    {
      e.getWindow().dispose();                  
      System.exit(0);                            
    }           
  }
 
  public static void main (String args[]) 
  {
    new UserInterface();
  }
}