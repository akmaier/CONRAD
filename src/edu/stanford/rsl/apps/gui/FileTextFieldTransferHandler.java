package edu.stanford.rsl.apps.gui;

/*
 * Copyright (c) 1995 - 2008 Sun Microsystems, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Sun Microsystems nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Modified by Andreas Maier to handle File and Text Input
 * 
 */ 


import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.io.File;
import java.io.IOException;

import javax.swing.JComponent;
import javax.swing.JTextField;
import javax.swing.TransferHandler;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import javax.swing.text.JTextComponent;
import javax.swing.text.Position;

public class FileTextFieldTransferHandler extends TransferHandler {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7758055750140825222L;
	JTextField field;
	private Position p1;
	private Position p0;
	
	public FileTextFieldTransferHandler(JTextField field){
		super();
		this.field = field;
	}
	
	public boolean canImport(TransferHandler.TransferSupport support) {
		if (!(support.isDataFlavorSupported(DataFlavor.stringFlavor)||support.isDataFlavorSupported(DataFlavor.javaFileListFlavor))) {
			return false;
		}
		
		boolean copySupported = true;
		if (support.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
			copySupported = (COPY & support.getSourceDropActions()) == COPY;
			support.setDropAction(COPY);
		}
		if (!copySupported) {
			return false;
		}




		return true;
	}

	@SuppressWarnings("unchecked")
	public boolean importData(TransferHandler.TransferSupport support) {
		if (!canImport(support)) {
			return false;
		}

		Transferable t = support.getTransferable();

		try {
			java.util.List<File> l =
				(java.util.List<File>) t.getTransferData(DataFlavor.javaFileListFlavor);

			field.setText(l.get(0).getAbsolutePath());
			
		} catch (UnsupportedFlavorException e) {
			try {
				String l =
					(String) t.getTransferData(DataFlavor.stringFlavor);

				field.replaceSelection(l);
				
			} catch (UnsupportedFlavorException e2) {
				return false;
			} catch (IOException e3) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 
		} catch (IOException e) {
			return false;
		}
		


		return true;
	}

	   /**
     * Bundle up the data for export.
     */
    protected Transferable createTransferable(JComponent c) {
        JTextField source = (JTextField)c;
        int start = source.getSelectionStart();
        int end = source.getSelectionEnd();
        Document doc = source.getDocument();
        if (start == end) {
            return null;
        }
        try {
            p0 = doc.createPosition(start);
            p1 = doc.createPosition(end);
        } catch (BadLocationException e) {
            System.out.println(
                    "Can't create position - unable to remove text from source.");
        }
        String data = source.getSelectedText();
        return new StringSelection(data);
    }
    
    /**
     * When the export is complete, remove the old text if the action
     * was a move.
     */
    protected void exportDone(JComponent c, Transferable data, int action) {
        if (action != MOVE) {
            return;
        }
        
        if ((p0 != null) && (p1 != null) &&
            (p0.getOffset() != p1.getOffset())) {
            try {
                JTextComponent tc = (JTextComponent)c;
                tc.getDocument().remove(p0.getOffset(), 
                        p1.getOffset() - p0.getOffset());
            } catch (BadLocationException e) {
                System.out.println("Can't remove text from source.");
            }
        }
    }


    /**
     * These text fields handle both copy and move actions.
     */
    public int getSourceActions(JComponent c) {
        return COPY_OR_MOVE;
    }

	
}
