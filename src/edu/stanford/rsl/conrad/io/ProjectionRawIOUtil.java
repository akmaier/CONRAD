package edu.stanford.rsl.conrad.io;


import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.io.FileInfo;
import ij.io.FileOpener;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Class to write and read raw projection matrices from files (typically .matrices).
 * The result is read to Projection[] structures and returned.
 *
 * @author Alexander Preuhs
 *
 */
public class ProjectionRawIOUtil {

    //method reading "".matrices files which are used by the rigid motion creator GUI
    //read projection matrices with 6 byte header, decoding the widht(2 byte) height (2 byte) and length (2 byte) of
    //the projection matrix stack. The projection matirx values are decoded in 8 byte length. Encoding is little endian
    public static Projection[] readProjectionMatrices_raw(String filename){
        //notes on load raw:
        //you might also use for loops and reading the data manually, however using these
        //predefined function is way way way faster...
        int byte_header = 6;
        try {
            byte[] buffer = new byte[byte_header];
            ByteBuffer byteBuffer = ByteBuffer.allocate(byte_header);

            FileInputStream fis = new FileInputStream(filename);
            DataInputStream dis = new DataInputStream(fis);
            dis.read(buffer);
            byteBuffer = ByteBuffer.wrap(buffer);
            int width = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getChar();
            int height = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getChar();
            int num_proj = byteBuffer.order(ByteOrder.LITTLE_ENDIAN).getChar();
            dis.close();

            //read header
            FileInfo fi = new FileInfo();
            File file = new File(filename);
            fi.fileName = file.getName();
            fi.directory = file.getParent() + "/";
            fi.fileType = FileInfo.GRAY64_FLOAT;
            fi.fileFormat = FileInfo.RAW;
            fi.nImages = num_proj;
            //width and height is switched, because we
            //want to read columnwise...
            fi.width = height;
            fi.height = width;
            fi.offset = byte_header;
            //data is encode in little endian which referred to as intelByteOrder
            fi.intelByteOrder = true;

            FileOpener fO = new FileOpener(fi);
            Grid3D projTbl = ImageUtil.wrapImagePlus(fO.open(false));
            Grid2D subgrid;
            Projection[] projectionMatrices = new Projection[num_proj];
            for (int i = 0; i<num_proj;i++){
                SimpleMatrix mat = new SimpleMatrix(3,4);
                subgrid = projTbl.getSubGrid(i);

                for (int j =0; j <3;j++){
                    for(int k = 0; k < 4;k++){
                        mat.setElementValue(j, k, subgrid.getPixelValue(j, k));
                    }
                }
                projectionMatrices[i] = new Projection(mat);
            }
            return projectionMatrices;

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }

    //method writing "".matrices files which are used by the rigid motion creator GUI
    //write projection matrices with 6 byte header, decoding the widht(2 byte) height (2 byte) and length (2 byte) of
    //the projection matrix stack. The projection matirx values are decoded in 8 byte length. Encoding is little endian
    public static void writeProjectionMatrices_raw(Projection[] pMat, String filename){
        short numProj = (short)pMat.length;
        short width = 4;
        short height = 3;
        short numHeader = 3;

        int byteForuint8 = 2;
        int byteForFloat64 = 8;


        FileOutputStream out = null;
        try {
            out = new FileOutputStream(new File(filename));
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        ByteBuffer buf = ByteBuffer.allocate(numHeader*byteForuint8 +byteForFloat64*width*height*numProj);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort((width));
        buf.putShort(height);
        buf.putShort((numProj));

        for(Projection p:pMat){
            SimpleMatrix prj = p.computeP();
            for(int col = 0;col<4;col++){
                for(int row = 0; row <3;row++){
                    buf.putDouble((double)prj.getElement(row, col));
                }
            }
        }
        final byte[] contents = buf.array();
        try {
            out.write(contents);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            out.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
