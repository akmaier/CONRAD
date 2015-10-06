/*
 * Copyright (C) 2015 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLEventList;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.FloatBuffer;
import java.util.LinkedList;

/**
 * Code was taken from Michael Bien's JOCL-Demo library
 * See also: <a href="https://github.com/JogAmp/jocl-demos"> the official JOCL-Demos repository</a>
 *  
 * Here is the original copyright notice including open work:
 * 
 * sample is based on Apple's FFT example.
 * initial port to JOCL Copyright 2010 Michael Zucchi
 *
 * TODO: The execute functions may allocate/use temporary memory per call hence they are
 * neither thread safe nor multiple-queue safe.  Perhaps some per-queue allocation
 * system would suffice.
 * TODO: The dynamic device-dependent variables should be dynamic and device-dependent and not
 * hardcoded.  Where possible.
 * TODO: CPU support?
 *  
 * @author notzed
 */
public class CLFFTPlan {

    private class CLFFTDim3 {

        int x;
        int y;
        int z;

        CLFFTDim3(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        CLFFTDim3(int[] size) {
            x = size[0];
            y = size.length > 1 ? size[1] : 1;
            z = size.length > 2 ? size[2] : 1;
        }
    }

    private class WorkDimensions {

        int batchSize;
        long gWorkItems;
        long lWorkItems;

        public WorkDimensions(int batchSize, long gWorkItems, long lWorkItems) {
            this.batchSize = batchSize;
            this.gWorkItems = gWorkItems;
            this.lWorkItems = lWorkItems;
        }
    }

    private class fftPadding {

        int lMemSize;
        int offset;
        int midPad;

        public fftPadding(int lMemSize, int offset, int midPad) {
            this.lMemSize = lMemSize;
            this.offset = offset;
            this.midPad = midPad;
        }
    }

    class CLFFTKernelInfo {

        CLKernel kernel;
        String kernel_name;
        int lmem_size;
        int num_workgroups;
        int num_xforms_per_workgroup;
        int num_workitems_per_workgroup;
        CLFFTKernelDir dir;
        boolean in_place_possible;
    };

    public enum CLFFTDirection {

        Forward {

            int value() {
                return -1;
            }
        },
        Inverse {

            int value() {
                return 1;
            }
        };

        abstract int value();
    };

    enum CLFFTKernelDir {
        X,
        Y,
        Z
    };

    public enum CLFFTDataFormat {
        SplitComplexFormat,
        InterleavedComplexFormat,
    }

    // context in which fft resources are created and kernels are executed
    CLContext context;
    // size of signal
    CLFFTDim3 size;
    // dimension of transform ... must be either 1, 2 or 3
    int dim;
    // data format ... must be either interleaved or plannar
    CLFFTDataFormat format;
    // string containing kernel source. Generated at runtime based on
    // size, dim, format and other parameters
    StringBuilder kernel_string;
    // CL program containing source and kernel this particular
    // size, dim, data format
    CLProgram program;
    // linked list of kernels which needs to be executed for this fft
    LinkedList<CLFFTKernelInfo> kernel_list;
    // twist kernel for virtualizing fft of very large sizes that do not
    // fit in GPU global memory
    CLKernel twist_kernel;
    // flag indicating if temporary intermediate buffer is needed or not.
    // this depends on fft kernels being executed and if transform is
    // in-place or out-of-place. e.g. Local memory fft (say 1D 1024 ...
    // one that does not require global transpose do not need temporary buffer)
    // 2D 1024x1024 out-of-place fft however do require intermediate buffer.
    // If temp buffer is needed, its allocation is lazy i.e. its not allocated
    // until its needed
    boolean temp_buffer_needed;
    // Batch size is runtime parameter and size of temporary buffer (if needed)
    // depends on batch size. Allocation of temporary buffer is lazy i.e. its
    // only created when needed. Once its created at first call of clFFT_Executexxx
    // it is not allocated next time if next time clFFT_Executexxx is called with
    // batch size different than the first call. last_batch_size caches the last
    // batch size with which this plan is used so that we dont keep allocating/deallocating
    // temp buffer if same batch size is used again and again.
    int last_batch_size;
    // temporary buffer for interleaved plan
    CLMemory tempmemobj;
    // temporary buffer for planner plan. Only one of tempmemobj or
    // (tempmemobj_real, tempmemobj_imag) pair is valid (allocated) depending
    // data format of plan (plannar or interleaved)
    CLMemory tempmemobj_real, tempmemobj_imag;
    // Maximum size of signal for which local memory transposed based
    // fft is sufficient i.e. no global mem transpose (communication)
    // is needed
    int max_localmem_fft_size;
    // Maximum work items per work group allowed. This, along with max_radix below controls
    // maximum local memory being used by fft kernels of this plan. Set to 256 by default
    int max_work_item_per_workgroup;
    // Maximum base radix for local memory fft ... this controls the maximum register
    // space used by work items. Currently defaults to 16
    int max_radix;
    // Device depended parameter that tells how many work-items need to be read consecutive
    // values to make sure global memory access by work-items of a work-group result in
    // coalesced memory access to utilize full bandwidth e.g. on NVidia tesla, this is 16
    int min_mem_coalesce_width;
    // Number of local memory banks. This is used to geneate kernel with local memory
    // transposes with appropriate padding to avoid bank conflicts to local memory
    // e.g. on NVidia it is 16.
    int num_local_mem_banks;

    public class InvalidContextException extends Exception {
    }

    /**
     * Create a new FFT plan.
     *
     * Use the matching executeInterleaved() or executePlanar() depending on the dataFormat specified.
     * @param context
     * @param sizes Array of sizes for each dimension.  The length of array defines how many dimensions there are.
     * @param dataFormat Data format, InterleavedComplex (array of complex) or SplitComplex (separate planar arrays).
     * @throws zephyr.cl.CLFFTPlan.InvalidContextException
     */
    public CLFFTPlan(CLContext context, int[] sizes, CLFFTDataFormat dataFormat) throws InvalidContextException {
        int i;
        int err;
        boolean isPow2 = true;
        String kString;
        int num_devices;
        boolean gpu_found = false;
        CLDevice[] devices;
        int ret_size;

        if (sizes.length < 1 || sizes.length > 3) {
            throw new IllegalArgumentException("Dimensions must be between 1 and 3");
        }

        this.size = new CLFFTDim3(sizes);

        isPow2 |= (this.size.x != 0) && (((this.size.x - 1) & this.size.x) == 0);
        isPow2 |= (this.size.y != 0) && (((this.size.y - 1) & this.size.y) == 0);
        isPow2 |= (this.size.z != 0) && (((this.size.z - 1) & this.size.z) == 0);

        if (!isPow2) {
            throw new IllegalArgumentException("Sizes must be power of two");
        }

        //if( (dim == FFT_1D && (size.y != 1 || size.z != 1)) || (dim == FFT_2D && size.z != 1) )
        //	ERR_MACRO(CL_INVALID_VALUE);

        this.context = context;
        //clRetainContext(context);
        //this.size = size;
        this.dim = sizes.length;
        this.format = dataFormat;
        //this.kernel_list = 0;
        //this.twist_kernel = 0;
        //this.program = 0;
        this.temp_buffer_needed = false;
        this.last_batch_size = 0;
        //this.tempmemobj = 0;
        //this.tempmemobj_real = 0;
        //this.tempmemobj_imag = 0;
        this.max_localmem_fft_size = 2048;
        this.max_work_item_per_workgroup = 256;
        this.max_radix = 16;
        this.min_mem_coalesce_width = 16;
        this.num_local_mem_banks = 16;

        boolean done = false;

        // this seems pretty shit, can't it tell this before building it?
        while (!done) {
            kernel_list = new LinkedList<CLFFTKernelInfo>();

            this.kernel_string = new StringBuilder();
            getBlockConfigAndKernelString();

            this.program = context.createProgram(kernel_string.toString());

            devices = context.getDevices();
            for (i = 0; i < devices.length; i++) {
                CLDevice dev = devices[i];

                if (dev.getType() == CLDevice.Type.GPU) {
                    gpu_found = true;
                    program.build("-cl-mad-enable", dev);
                }
            }

            if (!gpu_found) {
                throw new InvalidContextException();
            }

            createKernelList();

            // we created program and kernels based on "some max work group size (default 256)" ... this work group size
            // may be larger than what kernel may execute with ... if thats the case we need to regenerate the kernel source
            // setting this as limit i.e max group size and rebuild.
            if (getPatchingRequired(devices)) {
                this.max_work_item_per_workgroup = (int) getMaxKernelWorkGroupSize(devices);
                release();
            } else {
                done = true;
            }
        }
    }

    /**
     * Release system resources.
     */
    public void release() {
        program.release();
    }

    void allocateTemporaryBufferInterleaved(int batchSize) {
        if (temp_buffer_needed && last_batch_size != batchSize) {
            last_batch_size = batchSize;
            int tmpLength = size.x * size.y * size.z * batchSize * 2 * 4; // sizeof(float)

            if (tempmemobj != null) {
                tempmemobj.release();
            }

            tempmemobj = context.createFloatBuffer(tmpLength, Mem.READ_WRITE);
        }
    }

    /**
     * Calculate FFT on interleaved complex data.
     * @param queue
     * @param batchSize How many instances to calculate.  Use 1 for a single FFT.
     * @param dir Direction of calculation, Forward or Inverse.
     * @param data_in Input buffer.
     * @param data_out Output buffer.  May be the same as data_in for in-place transform.
     * @param condition Condition to wait for.  NOT YET IMPLEMENTED.
     * @param event Event to wait for completion.  NOT YET IMPLEMENTED.
     */
    public void executeInterleaved(CLCommandQueue queue, int batchSize, CLFFTDirection dir,
            CLBuffer<FloatBuffer> data_in, CLBuffer<FloatBuffer> data_out,
            CLEventList condition, CLEventList event) {
        int s;
        if (format != format.InterleavedComplexFormat) {
            throw new IllegalArgumentException();
        }

        WorkDimensions wd;
        boolean inPlaceDone = false;

        boolean isInPlace = data_in == data_out;

        allocateTemporaryBufferInterleaved(batchSize);

        CLMemory[] memObj = new CLMemory[3];
        memObj[0] = data_in;
        memObj[1] = data_out;
        memObj[2] = tempmemobj;
        int numKernels = kernel_list.size();

        boolean numKernelsOdd = (numKernels & 1) != 0;
        int currRead = 0;
        int currWrite = 1;

        // at least one external dram shuffle (transpose) required
        if (temp_buffer_needed) {
            // in-place transform
            if (isInPlace) {
                inPlaceDone = false;
                currRead = 1;
                currWrite = 2;
            } else {
                currWrite = (numKernels & 1) == 1 ? 1 : 2;
            }

            for (CLFFTKernelInfo kernelInfo : kernel_list) {
                if (isInPlace && numKernelsOdd && !inPlaceDone && kernelInfo.in_place_possible) {
                    currWrite = currRead;
                    inPlaceDone = true;
                }

                s = batchSize;
                wd = getKernelWorkDimensions(kernelInfo, s);
                kernelInfo.kernel.setArg(0, memObj[currRead]);
                kernelInfo.kernel.setArg(1, memObj[currWrite]);
                kernelInfo.kernel.setArg(2, dir.value());
                kernelInfo.kernel.setArg(3, wd.batchSize);
                queue.put2DRangeKernel(kernelInfo.kernel, 0, 0, wd.gWorkItems, 1, wd.lWorkItems, 1);
                //queue.put1DRangeKernel(kernelInfo.kernel, 0, wd.gWorkItems, wd.lWorkItems);

                //System.out.printf("execute %s size %d,%d batch %d, dir %d, currread %d currwrite %d\size", kernelInfo.kernel_name, wd.gWorkItems, wd.lWorkItems, wd.batchSize, dir.value(), currRead, currWrite);

                currRead = (currWrite == 1) ? 1 : 2;
                currWrite = (currWrite == 1) ? 2 : 1;
            }
        } else {
            // no dram shuffle (transpose required) transform
            // all kernels can execute in-place.
            for (CLFFTKernelInfo kernelInfo : kernel_list) {
                {
                    s = batchSize;
                    wd = getKernelWorkDimensions(kernelInfo, s);

                    kernelInfo.kernel.setArg(0, memObj[currRead]);
                    kernelInfo.kernel.setArg(1, memObj[currWrite]);
                    kernelInfo.kernel.setArg(2, dir.value());
                    kernelInfo.kernel.setArg(3, wd.batchSize);
                    queue.put2DRangeKernel(kernelInfo.kernel, 0, 0, wd.gWorkItems, 1, wd.lWorkItems, 1);

                    //System.out.printf("execute %s size %d,%d batch %d, currread %d currwrite %d\size", kernelInfo.kernel_name, wd.gWorkItems, wd.lWorkItems, wd.batchSize, currRead, currWrite);

                    currRead = 1;
                    currWrite = 1;
                }
            }
        }
    }

    void allocateTemporaryBufferPlanar(int batchSize) {
        if (temp_buffer_needed && last_batch_size != batchSize) {
            last_batch_size = batchSize;
            int tmpLength = size.x * size.y * size.z * batchSize * 4; //sizeof(cl_float);

            if (tempmemobj_real != null) {
                tempmemobj_real.release();
            }

            if (tempmemobj_imag != null) {
                tempmemobj_imag.release();
            }

            tempmemobj_real = context.createFloatBuffer(tmpLength, Mem.READ_WRITE);
            tempmemobj_imag = context.createFloatBuffer(tmpLength, Mem.READ_WRITE);
        }
    }

    /**
     * Calculate FFT of planar data.
     * @param queue
     * @param batchSize
     * @param dir
     * @param data_in_real
     * @param data_in_imag
     * @param data_out_real
     * @param data_out_imag
     * @param contition
     * @param event
     */
    public void executePlanar(CLCommandQueue queue, int batchSize, CLFFTDirection dir,
            CLBuffer<FloatBuffer> data_in_real, CLBuffer<FloatBuffer> data_in_imag, CLBuffer<FloatBuffer> data_out_real, CLBuffer<FloatBuffer> data_out_imag,
            CLEventList contition, CLEventList event) {
        int s;

        if (format != format.SplitComplexFormat) {
            throw new IllegalArgumentException();
        }

        int err;
        WorkDimensions wd;
        boolean inPlaceDone = false;

        boolean isInPlace = ((data_in_real == data_out_real) && (data_in_imag == data_out_imag));

        allocateTemporaryBufferPlanar(batchSize);

        CLMemory[] memObj_real = new CLMemory[3];
        CLMemory[] memObj_imag = new CLMemory[3];
        memObj_real[0] = data_in_real;
        memObj_real[1] = data_out_real;
        memObj_real[2] = tempmemobj_real;
        memObj_imag[0] = data_in_imag;
        memObj_imag[1] = data_out_imag;
        memObj_imag[2] = tempmemobj_imag;

        int numKernels = kernel_list.size();

        boolean numKernelsOdd = (numKernels & 1) == 1;
        int currRead = 0;
        int currWrite = 1;

        // at least one external dram shuffle (transpose) required
        if (temp_buffer_needed) {
            // in-place transform
            if (isInPlace) {
                inPlaceDone = false;
                currRead = 1;
                currWrite = 2;
            } else {
                currWrite = (numKernels & 1) == 1 ? 1 : 2;
            }

            for (CLFFTKernelInfo kernelInfo : kernel_list) {
                if (isInPlace && numKernelsOdd && !inPlaceDone && kernelInfo.in_place_possible) {
                    currWrite = currRead;
                    inPlaceDone = true;
                }

                s = batchSize;
                wd = getKernelWorkDimensions(kernelInfo, s);

                kernelInfo.kernel.setArg(0, memObj_real[currRead]);
                kernelInfo.kernel.setArg(1, memObj_imag[currRead]);
                kernelInfo.kernel.setArg(2, memObj_real[currWrite]);
                kernelInfo.kernel.setArg(3, memObj_imag[currWrite]);
                kernelInfo.kernel.setArg(4, dir.value());
                kernelInfo.kernel.setArg(5, wd.batchSize);

                queue.put1DRangeKernel(kernelInfo.kernel, 0, wd.gWorkItems, wd.lWorkItems);


                currRead = (currWrite == 1) ? 1 : 2;
                currWrite = (currWrite == 1) ? 2 : 1;

            }
        } // no dram shuffle (transpose required) transform
        else {

            for (CLFFTKernelInfo kernelInfo : kernel_list) {
                s = batchSize;
                wd = getKernelWorkDimensions(kernelInfo, s);

                kernelInfo.kernel.setArg(0, memObj_real[currRead]);
                kernelInfo.kernel.setArg(1, memObj_imag[currRead]);
                kernelInfo.kernel.setArg(2, memObj_real[currWrite]);
                kernelInfo.kernel.setArg(3, memObj_imag[currWrite]);
                kernelInfo.kernel.setArg(4, dir.value());
                kernelInfo.kernel.setArg(5, wd.batchSize);

                queue.put1DRangeKernel(kernelInfo.kernel, 0, wd.gWorkItems, wd.lWorkItems);
                currRead = 1;
                currWrite = 1;
            }
        }
    }

    /**
     * Dump the planner result to the output stream.
     * @param os if null, System.out is used.
     */
    public void dumpPlan(OutputStream os) {
        PrintStream out = os == null ? System.out : new PrintStream(os);

        for (CLFFTKernelInfo kInfo : kernel_list) {
            int s = 1;
            WorkDimensions wd = getKernelWorkDimensions(kInfo, s);
            out.printf("Run kernel %s with global dim = {%d*BatchSize}, local dim={%d}\n", kInfo.kernel_name, wd.gWorkItems, wd.lWorkItems);
        }
        out.printf("%s\n", kernel_string.toString());
    }

    WorkDimensions getKernelWorkDimensions(CLFFTKernelInfo kernelInfo, int batchSize) {
        int lWorkItems = kernelInfo.num_workitems_per_workgroup;
        int numWorkGroups = kernelInfo.num_workgroups;
        int numXFormsPerWG = kernelInfo.num_xforms_per_workgroup;

        switch (kernelInfo.dir) {
            case X:
                batchSize *= (size.y * size.z);
                numWorkGroups = ((batchSize % numXFormsPerWG) != 0) ? (batchSize / numXFormsPerWG + 1) : (batchSize / numXFormsPerWG);
                numWorkGroups *= kernelInfo.num_workgroups;
                break;
            case Y:
                batchSize *= size.z;
                numWorkGroups *= batchSize;
                break;
            case Z:
                numWorkGroups *= batchSize;
                break;
        }

        return new WorkDimensions(batchSize, numWorkGroups * lWorkItems, lWorkItems);
    }

    /*
     *
     * Kernel building/customisation code follows
     *
     */
    private void getBlockConfigAndKernelString() {
        this.temp_buffer_needed = false;
        this.kernel_string.append(baseKernels);

        if (this.format == CLFFTDataFormat.SplitComplexFormat) {
            this.kernel_string.append(twistKernelPlannar);
        } else {
            this.kernel_string.append(twistKernelInterleaved);
        }

        switch (this.dim) {
            case 1:
                FFT1D(CLFFTKernelDir.X);
                break;

            case 2:
                FFT1D(CLFFTKernelDir.X);
                FFT1D(CLFFTKernelDir.Y);
                break;

            case 3:
                FFT1D(CLFFTKernelDir.X);
                FFT1D(CLFFTKernelDir.Y);
                FFT1D(CLFFTKernelDir.Z);
                break;

            default:
                return;
        }

        this.temp_buffer_needed = false;
        for (CLFFTKernelInfo kInfo : this.kernel_list) {
            this.temp_buffer_needed |= !kInfo.in_place_possible;
        }
    }

    private void createKernelList() {
        CLFFTKernelInfo kern;
        for (CLFFTKernelInfo kinfo : this.kernel_list) {
            kinfo.kernel = program.createCLKernel(kinfo.kernel_name);
        }

        if (format == format.SplitComplexFormat) {
            twist_kernel = program.createCLKernel("clFFT_1DTwistSplit");
        } else {
            twist_kernel = program.createCLKernel("clFFT_1DTwistInterleaved");
        }
    }

    private boolean getPatchingRequired(CLDevice[] devices) {
        int i;
        for (i = 0; i < devices.length; i++) {
            for (CLFFTKernelInfo kInfo : kernel_list) {
                if (kInfo.kernel.getWorkGroupSize(devices[i]) < kInfo.num_workitems_per_workgroup) {
                    return true;
                }
            }
        }
        return false;
    }

    long getMaxKernelWorkGroupSize(CLDevice[] devices) {
        long max_wg_size = Integer.MAX_VALUE;
        int i;

        for (i = 0; i < devices.length; i++) {
            for (CLFFTKernelInfo kInfo : kernel_list) {
                long wg_size = kInfo.kernel.getWorkGroupSize(devices[i]);

                if (max_wg_size > wg_size) {
                    max_wg_size = wg_size;
                }
            }
        }

        return max_wg_size;
    }

    int log2(int x) {
        return 32 - Integer.numberOfLeadingZeros(x - 1);
    }

// For any size, this function decomposes size into factors for loacal memory tranpose
// based fft. Factors (radices) are sorted such that the first one (radixArray[0])
// is the largest. This base radix determines the number of registers used by each
// work item and product of remaining radices determine the size of work group needed.
// To make things concrete with and example, suppose size = 1024. It is decomposed into
// 1024 = 16 x 16 x 4. Hence kernel uses float2 a[16], for local in-register fft and
// needs 16 x 4 = 64 work items per work group. So kernel first performance 64 length
// 16 ffts (64 work items working in parallel) following by transpose using local
// memory followed by again 64 length 16 ffts followed by transpose using local memory
// followed by 256 length 4 ffts. For the last step since with size of work group is
// 64 and each work item can array for 16 values, 64 work items can compute 256 length
// 4 ffts by each work item computing 4 length 4 ffts.
// Similarly for size = 2048 = 8 x 8 x 8 x 4, each work group has 8 x 8 x 4 = 256 work
// iterms which each computes 256 (in-parallel) length 8 ffts in-register, followed
// by transpose using local memory, followed by 256 length 8 in-register ffts, followed
// by transpose using local memory, followed by 256 length 8 in-register ffts, followed
// by transpose using local memory, followed by 512 length 4 in-register ffts. Again,
// for the last step, each work item computes two length 4 in-register ffts and thus
// 256 work items are needed to compute all 512 ffts.
// For size = 32 = 8 x 4, 4 work items first compute 4 in-register
// lenth 8 ffts, followed by transpose using local memory followed by 8 in-register
// length 4 ffts, where each work item computes two length 4 ffts thus 4 work items
// can compute 8 length 4 ffts. However if work group size of say 64 is choosen,
// each work group can compute 64/ 4 = 16 size 32 ffts (batched transform).
// Users can play with these parameters to figure what gives best performance on
// their particular device i.e. some device have less register space thus using
// smaller base radix can avoid spilling ... some has small local memory thus
// using smaller work group size may be required etc
    int getRadixArray(int n, int[] radixArray, int maxRadix) {
        if (maxRadix > 1) {
            maxRadix = Math.min(n, maxRadix);
            int cnt = 0;
            while (n > maxRadix) {
                radixArray[cnt++] = maxRadix;
                n /= maxRadix;
            }
            radixArray[cnt++] = n;
            return cnt;
        }

        switch (n) {
            case 2:
                radixArray[0] = 2;
                return 1;

            case 4:
                radixArray[0] = 4;
                return 1;

            case 8:
                radixArray[0] = 8;
                return 1;

            case 16:
                radixArray[0] = 8;
                radixArray[1] = 2;
                return 2;

            case 32:
                radixArray[0] = 8;
                radixArray[1] = 4;
                return 2;

            case 64:
                radixArray[0] = 8;
                radixArray[1] = 8;
                return 2;

            case 128:
                radixArray[0] = 8;
                radixArray[1] = 4;
                radixArray[2] = 4;
                return 3;

            case 256:
                radixArray[0] = 4;
                radixArray[1] = 4;
                radixArray[2] = 4;
                radixArray[3] = 4;
                return 4;

            case 512:
                radixArray[0] = 8;
                radixArray[1] = 8;
                radixArray[2] = 8;
                return 3;

            case 1024:
                radixArray[0] = 16;
                radixArray[1] = 16;
                radixArray[2] = 4;
                return 3;
            case 2048:
                radixArray[0] = 8;
                radixArray[1] = 8;
                radixArray[2] = 8;
                radixArray[3] = 4;
                return 4;
            default:
                return 0;
        }
    }

    void insertHeader(StringBuilder kernelString, String kernelName, CLFFTDataFormat dataFormat) {
        if (dataFormat == CLFFTPlan.CLFFTDataFormat.SplitComplexFormat) {
            kernelString.append("__kernel void ").append(kernelName).append("(__global float *in_real, __global float *in_imag, __global float *out_real, __global float *out_imag, int dir, int S)\n");
        } else {
            kernelString.append("__kernel void ").append(kernelName).append("(__global float2 *in, __global float2 *out, int dir, int S)\n");
        }
    }

    void insertVariables(StringBuilder kStream, int maxRadix) {
        kStream.append("    int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;\n");
        kStream.append("    int s, ii, jj, offset;\n");
        kStream.append("    float2 w;\n");
        kStream.append("    float ang, angf, ang1;\n");
        kStream.append("    __local float *lMemStore, *lMemLoad;\n");
        kStream.append("    float2 a[").append(maxRadix).append("];\n");
        kStream.append("    int lId = get_local_id( 0 );\n");
        kStream.append("    int groupId = get_group_id( 0 );\n");
    }

    void formattedLoad(StringBuilder kernelString, int aIndex, int gIndex, CLFFTDataFormat dataFormat) {
        if (dataFormat == dataFormat.InterleavedComplexFormat) {
            kernelString.append("        a[").append(aIndex).append("] = in[").append(gIndex).append("];\n");
        } else {
            kernelString.append("        a[").append(aIndex).append("].x = in_real[").append(gIndex).append("];\n");
            kernelString.append("        a[").append(aIndex).append("].y = in_imag[").append(gIndex).append("];\n");
        }
    }

    void formattedStore(StringBuilder kernelString, int aIndex, int gIndex, CLFFTDataFormat dataFormat) {
        if (dataFormat == dataFormat.InterleavedComplexFormat) {
            kernelString.append("        out[").append(gIndex).append("] = a[").append(aIndex).append("];\n");
        } else {
            kernelString.append("        out_real[").append(gIndex).append("] = a[").append(aIndex).append("].x;\n");
            kernelString.append("        out_imag[").append(gIndex).append("] = a[").append(aIndex).append("].y;\n");
        }
    }

    int insertGlobalLoadsAndTranspose(StringBuilder kernelString, int N, int numWorkItemsPerXForm, int numXFormsPerWG, int R0, int mem_coalesce_width, CLFFTDataFormat dataFormat) {
        int log2NumWorkItemsPerXForm = (int) log2(numWorkItemsPerXForm);
        int groupSize = numWorkItemsPerXForm * numXFormsPerWG;
        int i, j;
        int lMemSize = 0;

        if (numXFormsPerWG > 1) {
            kernelString.append("        s = S & ").append(numXFormsPerWG - 1).append(";\n");
        }

        if (numWorkItemsPerXForm >= mem_coalesce_width) {
            if (numXFormsPerWG > 1) {
                kernelString.append("    ii = lId & ").append(numWorkItemsPerXForm - 1).append(";\n");
                kernelString.append("    jj = lId >> ").append(log2NumWorkItemsPerXForm).append(";\n");
                kernelString.append("    if( !s || (groupId < get_num_groups(0)-1) || (jj < s) ) {\n");
                kernelString.append("        offset = mad24( mad24(groupId, ").append(numXFormsPerWG).append(", jj), ").append(N).append(", ii );\n");
                if (dataFormat == dataFormat.InterleavedComplexFormat) {
                    kernelString.append("        in += offset;\n");
                    kernelString.append("        out += offset;\n");
                } else {
                    kernelString.append("        in_real += offset;\n");
                    kernelString.append("        in_imag += offset;\n");
                    kernelString.append("        out_real += offset;\n");
                    kernelString.append("        out_imag += offset;\n");
                }
                for (i = 0; i < R0; i++) {
                    formattedLoad(kernelString, i, i * numWorkItemsPerXForm, dataFormat);
                }
                kernelString.append("    }\n");
            } else {
                kernelString.append("    ii = lId;\n");
                kernelString.append("    jj = 0;\n");
                kernelString.append("    offset =  mad24(groupId, ").append(N).append(", ii);\n");
                if (dataFormat == dataFormat.InterleavedComplexFormat) {
                    kernelString.append("        in += offset;\n");
                    kernelString.append("        out += offset;\n");
                } else {
                    kernelString.append("        in_real += offset;\n");
                    kernelString.append("        in_imag += offset;\n");
                    kernelString.append("        out_real += offset;\n");
                    kernelString.append("        out_imag += offset;\n");
                }
                for (i = 0; i < R0; i++) {
                    formattedLoad(kernelString, i, i * numWorkItemsPerXForm, dataFormat);
                }
            }
        } else if (N >= mem_coalesce_width) {
            int numInnerIter = N / mem_coalesce_width;
            int numOuterIter = numXFormsPerWG / (groupSize / mem_coalesce_width);

            kernelString.append("    ii = lId & ").append(mem_coalesce_width - 1).append(";\n");
            kernelString.append("    jj = lId >> ").append((int) log2(mem_coalesce_width)).append(";\n");
            kernelString.append("    lMemStore = sMem + mad24( jj, ").append(N + numWorkItemsPerXForm).append(", ii );\n");
            kernelString.append("    offset = mad24( groupId, ").append(numXFormsPerWG).append(", jj);\n");
            kernelString.append("    offset = mad24( offset, ").append(N).append(", ii );\n");
            if (dataFormat == dataFormat.InterleavedComplexFormat) {
                kernelString.append("        in += offset;\n");
                kernelString.append("        out += offset;\n");
            } else {
                kernelString.append("        in_real += offset;\n");
                kernelString.append("        in_imag += offset;\n");
                kernelString.append("        out_real += offset;\n");
                kernelString.append("        out_imag += offset;\n");
            }

            kernelString.append("if((groupId == get_num_groups(0)-1) && s) {\n");
            for (i = 0; i < numOuterIter; i++) {
                kernelString.append("    if( jj < s ) {\n");
                for (j = 0; j < numInnerIter; j++) {
                    formattedLoad(kernelString, i * numInnerIter + j, j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, dataFormat);
                }
                kernelString.append("    }\n");
                if (i != numOuterIter - 1) {
                    kernelString.append("    jj += ").append(groupSize / mem_coalesce_width).append(";\n");
                }
            }
            kernelString.append("}\n ");
            kernelString.append("else {\n");
            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    formattedLoad(kernelString, i * numInnerIter + j, j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, dataFormat);
                }
            }
            kernelString.append("}\n");

            kernelString.append("    ii = lId & ").append(numWorkItemsPerXForm - 1).append(";\n");
            kernelString.append("    jj = lId >> ").append(log2NumWorkItemsPerXForm).append(";\n");
            kernelString.append("    lMemLoad  = sMem + mad24( jj, ").append(N + numWorkItemsPerXForm).append(", ii);\n");

            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    kernelString.append("    lMemStore[").append(j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)).append("] = a[").append(i * numInnerIter + j).append("].x;\n");
                }
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < R0; i++) {
                kernelString.append("    a[").append(i).append("].x = lMemLoad[").append(i * numWorkItemsPerXForm).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    kernelString.append("    lMemStore[").append(j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)).append("] = a[").append(i * numInnerIter + j).append("].y;\n");
                }
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < R0; i++) {
                kernelString.append("    a[").append(i).append("].y = lMemLoad[").append(i * numWorkItemsPerXForm).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG;
        } else {
            kernelString.append("    offset = mad24( groupId,  ").append(N * numXFormsPerWG).append(", lId );\n");
            if (dataFormat == dataFormat.InterleavedComplexFormat) {
                kernelString.append("        in += offset;\n");
                kernelString.append("        out += offset;\n");
            } else {
                kernelString.append("        in_real += offset;\n");
                kernelString.append("        in_imag += offset;\n");
                kernelString.append("        out_real += offset;\n");
                kernelString.append("        out_imag += offset;\n");
            }

            kernelString.append("    ii = lId & ").append(N - 1).append(";\n");
            kernelString.append("    jj = lId >> ").append((int) log2(N)).append(";\n");
            kernelString.append("    lMemStore = sMem + mad24( jj, ").append(N + numWorkItemsPerXForm).append(", ii );\n");

            kernelString.append("if((groupId == get_num_groups(0)-1) && s) {\n");
            for (i = 0; i < R0; i++) {
                kernelString.append("    if(jj < s )\n");
                formattedLoad(kernelString, i, i * groupSize, dataFormat);
                if (i != R0 - 1) {
                    kernelString.append("    jj += ").append(groupSize / N).append(";\n");
                }
            }
            kernelString.append("}\n");
            kernelString.append("else {\n");
            for (i = 0; i < R0; i++) {
                formattedLoad(kernelString, i, i * groupSize, dataFormat);
            }
            kernelString.append("}\n");

            if (numWorkItemsPerXForm > 1) {
                kernelString.append("    ii = lId & ").append(numWorkItemsPerXForm - 1).append(";\n");
                kernelString.append("    jj = lId >> ").append(log2NumWorkItemsPerXForm).append(";\n");
                kernelString.append("    lMemLoad = sMem + mad24( jj, ").append(N + numWorkItemsPerXForm).append(", ii );\n");
            } else {
                kernelString.append("    ii = 0;\n");
                kernelString.append("    jj = lId;\n");
                kernelString.append("    lMemLoad = sMem + mul24( jj, ").append(N + numWorkItemsPerXForm).append(");\n");
            }


            for (i = 0; i < R0; i++) {
                kernelString.append("    lMemStore[").append(i * (groupSize / N) * (N + numWorkItemsPerXForm)).append("] = a[").append(i).append("].x;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < R0; i++) {
                kernelString.append("    a[").append(i).append("].x = lMemLoad[").append(i * numWorkItemsPerXForm).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < R0; i++) {
                kernelString.append("    lMemStore[").append(i * (groupSize / N) * (N + numWorkItemsPerXForm)).append("] = a[").append(i).append("].y;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < R0; i++) {
                kernelString.append("    a[").append(i).append("].y = lMemLoad[").append(i * numWorkItemsPerXForm).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG;
        }

        return lMemSize;
    }

    int insertGlobalStoresAndTranspose(StringBuilder kernelString, int N, int maxRadix, int Nr, int numWorkItemsPerXForm, int numXFormsPerWG, int mem_coalesce_width, CLFFTDataFormat dataFormat) {
        int groupSize = numWorkItemsPerXForm * numXFormsPerWG;
        int i, j, k, ind;
        int lMemSize = 0;
        int numIter = maxRadix / Nr;
        String indent = "";

        if (numWorkItemsPerXForm >= mem_coalesce_width) {
            if (numXFormsPerWG > 1) {
                kernelString.append("    if( !s || (groupId < get_num_groups(0)-1) || (jj < s) ) {\n");
                indent = ("    ");
            }
            for (i = 0; i < maxRadix; i++) {
                j = i % numIter;
                k = i / numIter;
                ind = j * Nr + k;
                formattedStore(kernelString, ind, i * numWorkItemsPerXForm, dataFormat);
            }
            if (numXFormsPerWG > 1) {
                kernelString.append("    }\n");
            }
        } else if (N >= mem_coalesce_width) {
            int numInnerIter = N / mem_coalesce_width;
            int numOuterIter = numXFormsPerWG / (groupSize / mem_coalesce_width);

            kernelString.append("    lMemLoad  = sMem + mad24( jj, ").append(N + numWorkItemsPerXForm).append(", ii );\n");
            kernelString.append("    ii = lId & ").append(mem_coalesce_width - 1).append(";\n");
            kernelString.append("    jj = lId >> ").append((int) log2(mem_coalesce_width)).append(";\n");
            kernelString.append("    lMemStore = sMem + mad24( jj,").append(N + numWorkItemsPerXForm).append(", ii );\n");

            for (i = 0; i < maxRadix; i++) {
                j = i % numIter;
                k = i / numIter;
                ind = j * Nr + k;
                kernelString.append("    lMemLoad[").append(i * numWorkItemsPerXForm).append("] = a[").append(ind).append("].x;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    kernelString.append("    a[").append(i * numInnerIter + j).append("].x = lMemStore[").append(j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)).append("];\n");
                }
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < maxRadix; i++) {
                j = i % numIter;
                k = i / numIter;
                ind = j * Nr + k;
                kernelString.append("    lMemLoad[").append(i * numWorkItemsPerXForm).append("] = a[").append(ind).append("].y;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    kernelString.append("    a[").append(i * numInnerIter + j).append("].y = lMemStore[").append(j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)).append("];\n");
                }
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            kernelString.append("if((groupId == get_num_groups(0)-1) && s) {\n");
            for (i = 0; i < numOuterIter; i++) {
                kernelString.append("    if( jj < s ) {\n");
                for (j = 0; j < numInnerIter; j++) {
                    formattedStore(kernelString, i * numInnerIter + j, j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, dataFormat);
                }
                kernelString.append("    }\n");
                if (i != numOuterIter - 1) {
                    kernelString.append("    jj += ").append(groupSize / mem_coalesce_width).append(";\n");
                }
            }
            kernelString.append("}\n");
            kernelString.append("else {\n");
            for (i = 0; i < numOuterIter; i++) {
                for (j = 0; j < numInnerIter; j++) {
                    formattedStore(kernelString, i * numInnerIter + j, j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, dataFormat);
                }
            }
            kernelString.append("}\n");

            lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG;
        } else {
            kernelString.append("    lMemLoad  = sMem + mad24( jj,").append(N + numWorkItemsPerXForm).append(", ii );\n");

            kernelString.append("    ii = lId & ").append(N - 1).append(";\n");
            kernelString.append("    jj = lId >> ").append((int) log2(N)).append(";\n");
            kernelString.append("    lMemStore = sMem + mad24( jj,").append(N + numWorkItemsPerXForm).append(", ii );\n");

            for (i = 0; i < maxRadix; i++) {
                j = i % numIter;
                k = i / numIter;
                ind = j * Nr + k;
                kernelString.append("    lMemLoad[").append(i * numWorkItemsPerXForm).append("] = a[").append(ind).append("].x;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < maxRadix; i++) {
                kernelString.append("    a[").append(i).append("].x = lMemStore[").append(i * (groupSize / N) * (N + numWorkItemsPerXForm)).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < maxRadix; i++) {
                j = i % numIter;
                k = i / numIter;
                ind = j * Nr + k;
                kernelString.append("    lMemLoad[").append(i * numWorkItemsPerXForm).append("] = a[").append(ind).append("].y;\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            for (i = 0; i < maxRadix; i++) {
                kernelString.append("    a[").append(i).append("].y = lMemStore[").append(i * (groupSize / N) * (N + numWorkItemsPerXForm)).append("];\n");
            }
            kernelString.append("    barrier( CLK_LOCAL_MEM_FENCE );\n");

            kernelString.append("if((groupId == get_num_groups(0)-1) && s) {\n");
            for (i = 0; i < maxRadix; i++) {
                kernelString.append("    if(jj < s ) {\n");
                formattedStore(kernelString, i, i * groupSize, dataFormat);
                kernelString.append("    }\n");
                if (i != maxRadix - 1) {
                    kernelString.append("    jj +=").append(groupSize / N).append(";\n");
                }
            }
            kernelString.append("}\n");
            kernelString.append("else {\n");
            for (i = 0; i < maxRadix; i++) {
                formattedStore(kernelString, i, i * groupSize, dataFormat);
            }
            kernelString.append("}\n");

            lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG;
        }

        return lMemSize;
    }

    void insertfftKernel(StringBuilder kernelString, int Nr, int numIter) {
        int i;
        for (i = 0; i < numIter; i++) {
            kernelString.append("    fftKernel").append(Nr).append("(a+").append(i * Nr).append(", dir);\n");
        }
    }

    void insertTwiddleKernel(StringBuilder kernelString, int Nr, int numIter, int Nprev, int len, int numWorkItemsPerXForm) {
        int z, k;
        int logNPrev = log2(Nprev);

        for (z = 0; z < numIter; z++) {
            if (z == 0) {
                if (Nprev > 1) {
                    kernelString.append("    angf = (float) (ii >> ").append(logNPrev).append(");\n");
                } else {
                    kernelString.append("    angf = (float) ii;\n");
                }
            } else {
                if (Nprev > 1) {
                    kernelString.append("    angf = (float) ((").append(z * numWorkItemsPerXForm).append(" + ii) >>").append(logNPrev).append(");\n");
                } else {
                    kernelString.append("    angf = (float) (").append(z * numWorkItemsPerXForm).append(" + ii);\n");
                }
            }

            for (k = 1; k < Nr; k++) {
                int ind = z * Nr + k;
                //float fac =  (float) (2.0 * M_PI * (double) k / (double) len);
                kernelString.append("    ang = dir * ( 2.0f * M_PI * ").append(k).append(".0f / ").append(len).append(".0f )").append(" * angf;\n");
                kernelString.append("    w = (float2)(native_cos(ang), native_sin(ang));\n");
                kernelString.append("    a[").append(ind).append("] = complexMul(a[").append(ind).append("], w);\n");
            }
        }
    }

    fftPadding getPadding(int numWorkItemsPerXForm, int Nprev, int numWorkItemsReq, int numXFormsPerWG, int Nr, int numBanks) {
        int offset, midPad;

        if ((numWorkItemsPerXForm <= Nprev) || (Nprev >= numBanks)) {
            offset = 0;
        } else {
            int numRowsReq = ((numWorkItemsPerXForm < numBanks) ? numWorkItemsPerXForm : numBanks) / Nprev;
            int numColsReq = 1;
            if (numRowsReq > Nr) {
                numColsReq = numRowsReq / Nr;
            }
            numColsReq = Nprev * numColsReq;
            offset = numColsReq;
        }

        if (numWorkItemsPerXForm >= numBanks || numXFormsPerWG == 1) {
            midPad = 0;
        } else {
            int bankNum = ((numWorkItemsReq + offset) * Nr) & (numBanks - 1);
            if (bankNum >= numWorkItemsPerXForm) {
                midPad = 0;
            } else {
                midPad = numWorkItemsPerXForm - bankNum;
            }
        }

        int lMemSize = (numWorkItemsReq + offset) * Nr * numXFormsPerWG + midPad * (numXFormsPerWG - 1);
        return new fftPadding(lMemSize, offset, midPad);
    }

    void insertLocalStores(StringBuilder kernelString, int numIter, int Nr, int numWorkItemsPerXForm, int numWorkItemsReq, int offset, String comp) {
        int z, k;

        for (z = 0; z < numIter; z++) {
            for (k = 0; k < Nr; k++) {
                int index = k * (numWorkItemsReq + offset) + z * numWorkItemsPerXForm;
                kernelString.append("    lMemStore[").append(index).append("] = a[").append(z * Nr + k).append("].").append(comp).append(";\n");
            }
        }
        kernelString.append("    barrier(CLK_LOCAL_MEM_FENCE);\n");
    }

    void insertLocalLoads(StringBuilder kernelString, int n, int Nr, int Nrn, int Nprev, int Ncurr, int numWorkItemsPerXForm, int numWorkItemsReq, int offset, String comp) {
        int numWorkItemsReqN = n / Nrn;
        int interBlockHNum = Math.max(Nprev / numWorkItemsPerXForm, 1);
        int interBlockHStride = numWorkItemsPerXForm;
        int vertWidth = Math.max(numWorkItemsPerXForm / Nprev, 1);
        vertWidth = Math.min(vertWidth, Nr);
        int vertNum = Nr / vertWidth;
        int vertStride = (n / Nr + offset) * vertWidth;
        int iter = Math.max(numWorkItemsReqN / numWorkItemsPerXForm, 1);
        int intraBlockHStride = (numWorkItemsPerXForm / (Nprev * Nr)) > 1 ? (numWorkItemsPerXForm / (Nprev * Nr)) : 1;
        intraBlockHStride *= Nprev;

        int stride = numWorkItemsReq / Nrn;
        int i;
        for (i = 0; i < iter; i++) {
            int ii = i / (interBlockHNum * vertNum);
            int zz = i % (interBlockHNum * vertNum);
            int jj = zz % interBlockHNum;
            int kk = zz / interBlockHNum;
            int z;
            for (z = 0; z < Nrn; z++) {
                int st = kk * vertStride + jj * interBlockHStride + ii * intraBlockHStride + z * stride;
                kernelString.append("    a[").append(i * Nrn + z).append("].").append(comp).append(" = lMemLoad[").append(st).append("];\n");
            }
        }
        kernelString.append("    barrier(CLK_LOCAL_MEM_FENCE);\n");
    }

    void insertLocalLoadIndexArithmatic(StringBuilder kernelString, int Nprev, int Nr, int numWorkItemsReq, int numWorkItemsPerXForm, int numXFormsPerWG, int offset, int midPad) {
        int Ncurr = Nprev * Nr;
        int logNcurr = log2(Ncurr);
        int logNprev = log2(Nprev);
        int incr = (numWorkItemsReq + offset) * Nr + midPad;

        if (Ncurr < numWorkItemsPerXForm) {
            if (Nprev == 1) {
                kernelString.append("    j = ii & ").append(Ncurr - 1).append(";\n");
            } else {
                kernelString.append("    j = (ii & ").append(Ncurr - 1).append(") >> ").append(logNprev).append(";\n");
            }

            if (Nprev == 1) {
                kernelString.append("    i = ii >> ").append(logNcurr).append(";\n");
            } else {
                kernelString.append("    i = mad24(ii >> ").append(logNcurr).append(", ").append(Nprev).append(", ii & ").append(Nprev - 1).append(");\n");
            }
        } else {
            if (Nprev == 1) {
                kernelString.append("    j = ii;\n");
            } else {
                kernelString.append("    j = ii >> ").append(logNprev).append(";\n");
            }
            if (Nprev == 1) {
                kernelString.append("    i = 0;\n");
            } else {
                kernelString.append("    i = ii & ").append(Nprev - 1).append(";\n");
            }
        }

        if (numXFormsPerWG > 1) {
            kernelString.append("    i = mad24(jj, ").append(incr).append(", i);\n");
        }

        kernelString.append("    lMemLoad = sMem + mad24(j, ").append(numWorkItemsReq + offset).append(", i);\n");
    }

    void insertLocalStoreIndexArithmatic(StringBuilder kernelString, int numWorkItemsReq, int numXFormsPerWG, int Nr, int offset, int midPad) {
        if (numXFormsPerWG == 1) {
            kernelString.append("    lMemStore = sMem + ii;\n");
        } else {
            kernelString.append("    lMemStore = sMem + mad24(jj, ").append((numWorkItemsReq + offset) * Nr + midPad).append(", ii);\n");
        }
    }

    void createLocalMemfftKernelString() {
        int[] radixArray = new int[10];
        int numRadix;

        int n = this.size.x;

        assert (n <= this.max_work_item_per_workgroup * this.max_radix);

        numRadix = getRadixArray(n, radixArray, 0);
        assert (numRadix > 0);

        if (n / radixArray[0] > this.max_work_item_per_workgroup) {
            numRadix = getRadixArray(n, radixArray, this.max_radix);
        }

        assert (radixArray[0] <= this.max_radix);
        assert (n / radixArray[0] <= this.max_work_item_per_workgroup);

        int tmpLen = 1;
        int i;
        for (i = 0; i < numRadix; i++) {
            assert ((radixArray[i] != 0) && !(((radixArray[i] - 1) != 0) & (radixArray[i] != 0)));
            tmpLen *= radixArray[i];
        }
        assert (tmpLen == n);

        //int offset, midPad;
        StringBuilder localString = new StringBuilder();
        String kernelName;

        CLFFTDataFormat dataFormat = this.format;
        StringBuilder kernelString = this.kernel_string;

        int kCount = kernel_list.size();

        kernelName = "fft" + (kCount);

        CLFFTKernelInfo kInfo = new CLFFTKernelInfo();
        kernel_list.add(kInfo);
        //kInfo.kernel = null;
        //kInfo.lmem_size = 0;
        //kInfo.num_workgroups = 0;
        //kInfo.num_workitems_per_workgroup = 0;
        kInfo.dir = CLFFTKernelDir.X;
        kInfo.in_place_possible = true;
        //kInfo.next = null;
        kInfo.kernel_name = kernelName;

        int numWorkItemsPerXForm = n / radixArray[0];
        int numWorkItemsPerWG = numWorkItemsPerXForm <= 64 ? 64 : numWorkItemsPerXForm;
        assert (numWorkItemsPerWG <= this.max_work_item_per_workgroup);
        int numXFormsPerWG = numWorkItemsPerWG / numWorkItemsPerXForm;
        kInfo.num_workgroups = 1;
        kInfo.num_xforms_per_workgroup = numXFormsPerWG;
        kInfo.num_workitems_per_workgroup = numWorkItemsPerWG;

        int[] N = radixArray;
        int maxRadix = N[0];
        int lMemSize = 0;

        insertVariables(localString, maxRadix);

        lMemSize = insertGlobalLoadsAndTranspose(localString, n, numWorkItemsPerXForm, numXFormsPerWG, maxRadix, this.min_mem_coalesce_width, dataFormat);
        kInfo.lmem_size = (lMemSize > kInfo.lmem_size) ? lMemSize : kInfo.lmem_size;

        String xcomp = "x";
        String ycomp = "y";

        int Nprev = 1;
        int len = n;
        int r;
        for (r = 0; r < numRadix; r++) {
            int numIter = N[0] / N[r];
            int numWorkItemsReq = n / N[r];
            int Ncurr = Nprev * N[r];
            insertfftKernel(localString, N[r], numIter);

            if (r < (numRadix - 1)) {
                fftPadding pad;

                insertTwiddleKernel(localString, N[r], numIter, Nprev, len, numWorkItemsPerXForm);
                pad = getPadding(numWorkItemsPerXForm, Nprev, numWorkItemsReq, numXFormsPerWG, N[r], this.num_local_mem_banks);
                kInfo.lmem_size = (pad.lMemSize > kInfo.lmem_size) ? pad.lMemSize : kInfo.lmem_size;
                insertLocalStoreIndexArithmatic(localString, numWorkItemsReq, numXFormsPerWG, N[r], pad.offset, pad.midPad);
                insertLocalLoadIndexArithmatic(localString, Nprev, N[r], numWorkItemsReq, numWorkItemsPerXForm, numXFormsPerWG, pad.offset, pad.midPad);
                insertLocalStores(localString, numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, pad.offset, xcomp);
                insertLocalLoads(localString, n, N[r], N[r + 1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, pad.offset, xcomp);
                insertLocalStores(localString, numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, pad.offset, ycomp);
                insertLocalLoads(localString, n, N[r], N[r + 1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, pad.offset, ycomp);
                Nprev = Ncurr;
                len = len / N[r];
            }
        }

        lMemSize = insertGlobalStoresAndTranspose(localString, n, maxRadix, N[numRadix - 1], numWorkItemsPerXForm, numXFormsPerWG, this.min_mem_coalesce_width, dataFormat);
        kInfo.lmem_size = (lMemSize > kInfo.lmem_size) ? lMemSize : kInfo.lmem_size;

        insertHeader(kernelString, kernelName, dataFormat);
        kernelString.append("{\n");
        if (kInfo.lmem_size > 0) {
            kernelString.append("    __local float sMem[").append(kInfo.lmem_size).append("];\n");
        }
        kernelString.append(localString);
        kernelString.append("}\n");
    }

// For size larger than what can be computed using local memory fft, global transposes
// multiple kernel launces is needed. For these sizes, size can be decomposed using
// much larger base radices i.e. say size = 262144 = 128 x 64 x 32. Thus three kernel
// launches will be needed, first computing 64 x 32, length 128 ffts, second computing
// 128 x 32 length 64 ffts, and finally a kernel computing 128 x 64 length 32 ffts.
// Each of these base radices can futher be divided into factors so that each of these
// base ffts can be computed within one kernel launch using in-register ffts and local
// memory transposes i.e for the first kernel above which computes 64 x 32 ffts on length
// 128, 128 can be decomposed into 128 = 16 x 8 i.e. 8 work items can compute 8 length
// 16 ffts followed by transpose using local memory followed by each of these eight
// work items computing 2 length 8 ffts thus computing 16 length 8 ffts in total. This
// means only 8 work items are needed for computing one length 128 fft. If we choose
// work group size of say 64, we can compute 64/8 = 8 length 128 ffts within one
// work group. Since we need to compute 64 x 32 length 128 ffts in first kernel, this
// means we need to launch 64 x 32 / 8 = 256 work groups with 64 work items in each
// work group where each work group is computing 8 length 128 ffts where each length
// 128 fft is computed by 8 work items. Same logic can be applied to other two kernels
// in this example. Users can play with difference base radices and difference
// decompositions of base radices to generates different kernels and see which gives
// best performance. Following function is just fixed to use 128 as base radix
    int getGlobalRadixInfo(int n, int[] radix, int[] R1, int[] R2) {
        int baseRadix = Math.min(n, 128);

        int numR = 0;
        int N = n;
        while (N > baseRadix) {
            N /= baseRadix;
            numR++;
        }

        for (int i = 0; i < numR; i++) {
            radix[i] = baseRadix;
        }

        radix[numR] = N;
        numR++;

        for (int i = 0; i < numR; i++) {
            int B = radix[i];
            if (B <= 8) {
                R1[i] = B;
                R2[i] = 1;
                continue;
            }

            int r1 = 2;
            int r2 = B / r1;
            while (r2 > r1) {
                r1 *= 2;
                r2 = B / r1;
            }
            R1[i] = r1;
            R2[i] = r2;
        }
        return numR;
    }

    void createGlobalFFTKernelString(int n, int BS, CLFFTKernelDir dir, int vertBS) {
        int i, j, k, t;
        int[] radixArr = new int[10];
        int[] R1Arr = new int[10];
        int[] R2Arr = new int[10];
        int radix, R1, R2;
        int numRadices;

        int maxThreadsPerBlock = this.max_work_item_per_workgroup;
        int maxArrayLen = this.max_radix;
        int batchSize = this.min_mem_coalesce_width;
        CLFFTDataFormat dataFormat = this.format;
        boolean vertical = (dir == dir.X) ? false : true;

        numRadices = getGlobalRadixInfo(n, radixArr, R1Arr, R2Arr);

        int numPasses = numRadices;

        StringBuilder localString = new StringBuilder();
        String kernelName;
        StringBuilder kernelString = this.kernel_string;

        int kCount = kernel_list.size();
        //cl_fft_kernel_info **kInfo = &this.kernel_list;
        //int kCount = 0;

        //while(*kInfo)
        //{
        //	kInfo = &kInfo.next;
        //	kCount++;
        //}

        int N = n;
        int m = (int) log2(n);
        int Rinit = vertical ? BS : 1;
        batchSize = vertical ? Math.min(BS, batchSize) : batchSize;
        int passNum;

        for (passNum = 0; passNum < numPasses; passNum++) {

            localString.setLength(0);
            //kernelName.clear();

            radix = radixArr[passNum];
            R1 = R1Arr[passNum];
            R2 = R2Arr[passNum];

            int strideI = Rinit;
            for (i = 0; i < numPasses; i++) {
                if (i != passNum) {
                    strideI *= radixArr[i];
                }
            }

            int strideO = Rinit;
            for (i = 0; i < passNum; i++) {
                strideO *= radixArr[i];
            }

            int threadsPerXForm = R2;
            batchSize = R2 == 1 ? this.max_work_item_per_workgroup : batchSize;
            batchSize = Math.min(batchSize, strideI);
            int threadsPerBlock = batchSize * threadsPerXForm;
            threadsPerBlock = Math.min(threadsPerBlock, maxThreadsPerBlock);
            batchSize = threadsPerBlock / threadsPerXForm;
            assert (R2 <= R1);
            assert (R1 * R2 == radix);
            assert (R1 <= maxArrayLen);
            assert (threadsPerBlock <= maxThreadsPerBlock);

            int numIter = R1 / R2;
            int gInInc = threadsPerBlock / batchSize;


            int lgStrideO = log2(strideO);
            int numBlocksPerXForm = strideI / batchSize;
            int numBlocks = numBlocksPerXForm;
            if (!vertical) {
                numBlocks *= BS;
            } else {
                numBlocks *= vertBS;
            }

            kernelName = "fft" + (kCount);
            CLFFTKernelInfo kInfo = new CLFFTKernelInfo();
            if (R2 == 1) {
                kInfo.lmem_size = 0;
            } else {
                if (strideO == 1) {
                    kInfo.lmem_size = (radix + 1) * batchSize;
                } else {
                    kInfo.lmem_size = threadsPerBlock * R1;
                }
            }
            kInfo.num_workgroups = numBlocks;
            kInfo.num_xforms_per_workgroup = 1;
            kInfo.num_workitems_per_workgroup = threadsPerBlock;
            kInfo.dir = dir;
            kInfo.in_place_possible = ((passNum == (numPasses - 1)) && ((numPasses & 1) != 0));
            //kInfo.next = NULL;
            kInfo.kernel_name = kernelName;

            insertVariables(localString, R1);

            if (vertical) {
                localString.append("xNum = groupId >> ").append((int) log2(numBlocksPerXForm)).append(";\n");
                localString.append("groupId = groupId & ").append(numBlocksPerXForm - 1).append(";\n");
                localString.append("indexIn = mad24(groupId, ").append(batchSize).append(", xNum << ").append((int) log2(n * BS)).append(");\n");
                localString.append("tid = mul24(groupId, ").append(batchSize).append(");\n");
                localString.append("i = tid >> ").append(lgStrideO).append(";\n");
                localString.append("j = tid & ").append(strideO - 1).append(";\n");
                int stride = radix * Rinit;
                for (i = 0; i < passNum; i++) {
                    stride *= radixArr[i];
                }
                localString.append("indexOut = mad24(i, ").append(stride).append(", j + ").append("(xNum << ").append((int) log2(n * BS)).append("));\n");
                localString.append("bNum = groupId;\n");
            } else {
                int lgNumBlocksPerXForm = log2(numBlocksPerXForm);
                localString.append("bNum = groupId & ").append(numBlocksPerXForm - 1).append(";\n");
                localString.append("xNum = groupId >> ").append(lgNumBlocksPerXForm).append(";\n");
                localString.append("indexIn = mul24(bNum, ").append(batchSize).append(");\n");
                localString.append("tid = indexIn;\n");
                localString.append("i = tid >> ").append(lgStrideO).append(";\n");
                localString.append("j = tid & ").append(strideO - 1).append(";\n");
                int stride = radix * Rinit;
                for (i = 0; i < passNum; i++) {
                    stride *= radixArr[i];
                }
                localString.append("indexOut = mad24(i, ").append(stride).append(", j);\n");
                localString.append("indexIn += (xNum << ").append(m).append(");\n");
                localString.append("indexOut += (xNum << ").append(m).append(");\n");
            }

            // Load Data
            int lgBatchSize = log2(batchSize);
            localString.append("tid = lId;\n");
            localString.append("i = tid & ").append(batchSize - 1).append(";\n");
            localString.append("j = tid >> ").append(lgBatchSize).append(";\n");
            localString.append("indexIn += mad24(j, ").append(strideI).append(", i);\n");

            if (dataFormat == dataFormat.SplitComplexFormat) {
                localString.append("in_real += indexIn;\n");
                localString.append("in_imag += indexIn;\n");
                for (j = 0; j < R1; j++) {
                    localString.append("a[").append(j).append("].x = in_real[").append(j * gInInc * strideI).append("];\n");
                }
                for (j = 0; j < R1; j++) {
                    localString.append("a[").append(j).append("].y = in_imag[").append(j * gInInc * strideI).append("];\n");
                }
            } else {
                localString.append("in += indexIn;\n");
                for (j = 0; j < R1; j++) {
                    localString.append("a[").append(j).append("] = in[").append(j * gInInc * strideI).append("];\n");
                }
            }

            localString.append("fftKernel").append(R1).append("(a, dir);\n");

            if (R2 > 1) {
                // twiddle
                for (k = 1; k < R1; k++) {
                    localString.append("ang = dir*(2.0f*M_PI*").append(k).append("/").append(radix).append(")*j;\n");
                    localString.append("w = (float2)(native_cos(ang), native_sin(ang));\n");
                    localString.append("a[").append(k).append("] = complexMul(a[").append(k).append("], w);\n");
                }

                // shuffle
                numIter = R1 / R2;
                localString.append("indexIn = mad24(j, ").append(threadsPerBlock * numIter).append(", i);\n");
                localString.append("lMemStore = sMem + tid;\n");
                localString.append("lMemLoad = sMem + indexIn;\n");
                for (k = 0; k < R1; k++) {
                    localString.append("lMemStore[").append(k * threadsPerBlock).append("] = a[").append(k).append("].x;\n");
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
                for (k = 0; k < numIter; k++) {
                    for (t = 0; t < R2; t++) {
                        localString.append("a[").append(k * R2 + t).append("].x = lMemLoad[").append(t * batchSize + k * threadsPerBlock).append("];\n");
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
                for (k = 0; k < R1; k++) {
                    localString.append("lMemStore[").append(k * threadsPerBlock).append("] = a[").append(k).append("].y;\n");
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
                for (k = 0; k < numIter; k++) {
                    for (t = 0; t < R2; t++) {
                        localString.append("a[").append(k * R2 + t).append("].y = lMemLoad[").append(t * batchSize + k * threadsPerBlock).append("];\n");
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");

                for (j = 0; j < numIter; j++) {
                    localString.append("fftKernel").append(R2).append("(a + ").append(j * R2).append(", dir);\n");
                }
            }

            // twiddle
            if (passNum < (numPasses - 1)) {
                localString.append("l = ((bNum << ").append(lgBatchSize).append(") + i) >> ").append(lgStrideO).append(";\n");
                localString.append("k = j << ").append((int) log2(R1 / R2)).append(";\n");
                localString.append("ang1 = dir*(2.0f*M_PI/").append(N).append(")*l;\n");
                for (t = 0; t < R1; t++) {
                    localString.append("ang = ang1*(k + ").append((t % R2) * R1 + (t / R2)).append(");\n");
                    localString.append("w = (float2)(native_cos(ang), native_sin(ang));\n");
                    localString.append("a[").append(t).append("] = complexMul(a[").append(t).append("], w);\n");
                }
            }

            // Store Data
            if (strideO == 1) {

                localString.append("lMemStore = sMem + mad24(i, ").append(radix + 1).append(", j << ").append((int) log2(R1 / R2)).append(");\n");
                localString.append("lMemLoad = sMem + mad24(tid >> ").append((int) log2(radix)).append(", ").append(radix + 1).append(", tid & ").append(radix - 1).append(");\n");

                for (i = 0; i < R1 / R2; i++) {
                    for (j = 0; j < R2; j++) {
                        localString.append("lMemStore[ ").append(i + j * R1).append("] = a[").append(i * R2 + j).append("].x;\n");
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
                if (threadsPerBlock >= radix) {
                    for (i = 0; i < R1; i++) {
                        localString.append("a[").append(i).append("].x = lMemLoad[").append(i * (radix + 1) * (threadsPerBlock / radix)).append("];\n");
                    }
                } else {
                    int innerIter = radix / threadsPerBlock;
                    int outerIter = R1 / innerIter;
                    for (i = 0; i < outerIter; i++) {
                        for (j = 0; j < innerIter; j++) {
                            localString.append("a[").append(i * innerIter + j).append("].x = lMemLoad[").append(j * threadsPerBlock + i * (radix + 1)).append("];\n");
                        }
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");

                for (i = 0; i < R1 / R2; i++) {
                    for (j = 0; j < R2; j++) {
                        localString.append("lMemStore[ ").append(i + j * R1).append("] = a[").append(i * R2 + j).append("].y;\n");
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");
                if (threadsPerBlock >= radix) {
                    for (i = 0; i < R1; i++) {
                        localString.append("a[").append(i).append("].y = lMemLoad[").append(i * (radix + 1) * (threadsPerBlock / radix)).append("];\n");
                    }
                } else {
                    int innerIter = radix / threadsPerBlock;
                    int outerIter = R1 / innerIter;
                    for (i = 0; i < outerIter; i++) {
                        for (j = 0; j < innerIter; j++) {
                            localString.append("a[").append(i * innerIter + j).append("].y = lMemLoad[").append(j * threadsPerBlock + i * (radix + 1)).append("];\n");
                        }
                    }
                }
                localString.append("barrier(CLK_LOCAL_MEM_FENCE);\n");

                localString.append("indexOut += tid;\n");
                if (dataFormat == dataFormat.SplitComplexFormat) {
                    localString.append("out_real += indexOut;\n");
                    localString.append("out_imag += indexOut;\n");
                    for (k = 0; k < R1; k++) {
                        localString.append("out_real[").append(k * threadsPerBlock).append("] = a[").append(k).append("].x;\n");
                    }
                    for (k = 0; k < R1; k++) {
                        localString.append("out_imag[").append(k * threadsPerBlock).append("] = a[").append(k).append("].y;\n");
                    }
                } else {
                    localString.append("out += indexOut;\n");
                    for (k = 0; k < R1; k++) {
                        localString.append("out[").append(k * threadsPerBlock).append("] = a[").append(k).append("];\n");
                    }
                }

            } else {
                localString.append("indexOut += mad24(j, ").append(numIter * strideO).append(", i);\n");
                if (dataFormat == dataFormat.SplitComplexFormat) {
                    localString.append("out_real += indexOut;\n");
                    localString.append("out_imag += indexOut;\n");
                    for (k = 0; k < R1; k++) {
                        localString.append("out_real[").append(((k % R2) * R1 + (k / R2)) * strideO).append("] = a[").append(k).append("].x;\n");
                    }
                    for (k = 0; k < R1; k++) {
                        localString.append("out_imag[").append(((k % R2) * R1 + (k / R2)) * strideO).append("] = a[").append(k).append("].y;\n");
                    }
                } else {
                    localString.append("out += indexOut;\n");
                    for (k = 0; k < R1; k++) {
                        localString.append("out[").append(((k % R2) * R1 + (k / R2)) * strideO).append("] = a[").append(k).append("];\n");
                    }
                }
            }

            insertHeader(kernelString, kernelName, dataFormat);
            kernelString.append("{\n");
            if (kInfo.lmem_size > 0) {
                kernelString.append("    __local float sMem[").append(kInfo.lmem_size).append("];\n");
            }
            kernelString.append(localString);
            kernelString.append("}\n");

            N /= radix;
            kernel_list.add(kInfo);
            kCount++;
        }
    }

    void FFT1D(CLFFTKernelDir dir) {
        int[] radixArray = new int[10];

        switch (dir) {
            case X:
                if (this.size.x > this.max_localmem_fft_size) {
                    createGlobalFFTKernelString(this.size.x, 1, dir, 1);
                } else if (this.size.x > 1) {
                    getRadixArray(this.size.x, radixArray, 0);
                    if (this.size.x / radixArray[0] <= this.max_work_item_per_workgroup) {
                        createLocalMemfftKernelString();
                    } else {
                        getRadixArray(this.size.x, radixArray, this.max_radix);
                        if (this.size.x / radixArray[0] <= this.max_work_item_per_workgroup) {
                            createLocalMemfftKernelString();
                        } else {
                            createGlobalFFTKernelString(this.size.x, 1, dir, 1);
                        }
                    }
                }
                break;

            case Y:
                if (this.size.y > 1) {
                    createGlobalFFTKernelString(this.size.y, this.size.x, dir, 1);
                }
                break;

            case Z:
                if (this.size.z > 1) {
                    createGlobalFFTKernelString(this.size.z, this.size.x * this.size.y, dir, 1);
                }
            default:
                return;
        }
    }

    /*
     *
     * Pre-defined kernel parts
     *
     */
    static String baseKernels =
            "#ifndef M_PI\n"
            + "#define M_PI 0x1.921fb54442d18p+1\n"
            + "#endif\n"
            + "#define complexMul(a,b) ((float2)(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))\n"
            + "#define conj(a) ((float2)((a).x, -(a).y))\n"
            + "#define conjTransp(a) ((float2)(-(a).y, (a).x))\n"
            + "\n"
            + "#define fftKernel2(a,dir) \\\n"
            + "{ \\\n"
            + "    float2 c = (a)[0];    \\\n"
            + "    (a)[0] = c + (a)[1];  \\\n"
            + "    (a)[1] = c - (a)[1];  \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel2S(d1,d2,dir) \\\n"
            + "{ \\\n"
            + "    float2 c = (d1);   \\\n"
            + "    (d1) = c + (d2);   \\\n"
            + "    (d2) = c - (d2);   \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel4(a,dir) \\\n"
            + "{ \\\n"
            + "    fftKernel2S((a)[0], (a)[2], dir); \\\n"
            + "    fftKernel2S((a)[1], (a)[3], dir); \\\n"
            + "    fftKernel2S((a)[0], (a)[1], dir); \\\n"
            + "    (a)[3] = (float2)(dir)*(conjTransp((a)[3])); \\\n"
            + "    fftKernel2S((a)[2], (a)[3], dir); \\\n"
            + "    float2 c = (a)[1]; \\\n"
            + "    (a)[1] = (a)[2]; \\\n"
            + "    (a)[2] = c; \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel4s(a0,a1,a2,a3,dir) \\\n"
            + "{ \\\n"
            + "    fftKernel2S((a0), (a2), dir); \\\n"
            + "    fftKernel2S((a1), (a3), dir); \\\n"
            + "    fftKernel2S((a0), (a1), dir); \\\n"
            + "    (a3) = (float2)(dir)*(conjTransp((a3))); \\\n"
            + "    fftKernel2S((a2), (a3), dir); \\\n"
            + "    float2 c = (a1); \\\n"
            + "    (a1) = (a2); \\\n"
            + "    (a2) = c; \\\n"
            + "}\n"
            + "\n"
            + "#define bitreverse8(a) \\\n"
            + "{ \\\n"
            + "    float2 c; \\\n"
            + "    c = (a)[1]; \\\n"
            + "    (a)[1] = (a)[4]; \\\n"
            + "    (a)[4] = c; \\\n"
            + "    c = (a)[3]; \\\n"
            + "    (a)[3] = (a)[6]; \\\n"
            + "    (a)[6] = c; \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel8(a,dir) \\\n"
            + "{ \\\n"
            + "	const float2 w1  = (float2)(0x1.6a09e6p-1f,  dir*0x1.6a09e6p-1f);  \\\n"
            + "	const float2 w3  = (float2)(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f);  \\\n"
            + "	float2 c; \\\n"
            + "	fftKernel2S((a)[0], (a)[4], dir); \\\n"
            + "	fftKernel2S((a)[1], (a)[5], dir); \\\n"
            + "	fftKernel2S((a)[2], (a)[6], dir); \\\n"
            + "	fftKernel2S((a)[3], (a)[7], dir); \\\n"
            + "	(a)[5] = complexMul(w1, (a)[5]); \\\n"
            + "	(a)[6] = (float2)(dir)*(conjTransp((a)[6])); \\\n"
            + "	(a)[7] = complexMul(w3, (a)[7]); \\\n"
            + "	fftKernel2S((a)[0], (a)[2], dir); \\\n"
            + "	fftKernel2S((a)[1], (a)[3], dir); \\\n"
            + "	fftKernel2S((a)[4], (a)[6], dir); \\\n"
            + "	fftKernel2S((a)[5], (a)[7], dir); \\\n"
            + "	(a)[3] = (float2)(dir)*(conjTransp((a)[3])); \\\n"
            + "	(a)[7] = (float2)(dir)*(conjTransp((a)[7])); \\\n"
            + "	fftKernel2S((a)[0], (a)[1], dir); \\\n"
            + "	fftKernel2S((a)[2], (a)[3], dir); \\\n"
            + "	fftKernel2S((a)[4], (a)[5], dir); \\\n"
            + "	fftKernel2S((a)[6], (a)[7], dir); \\\n"
            + "	bitreverse8((a)); \\\n"
            + "}\n"
            + "\n"
            + "#define bitreverse4x4(a) \\\n"
            + "{ \\\n"
            + "	float2 c; \\\n"
            + "	c = (a)[1];  (a)[1]  = (a)[4];  (a)[4]  = c; \\\n"
            + "	c = (a)[2];  (a)[2]  = (a)[8];  (a)[8]  = c; \\\n"
            + "	c = (a)[3];  (a)[3]  = (a)[12]; (a)[12] = c; \\\n"
            + "	c = (a)[6];  (a)[6]  = (a)[9];  (a)[9]  = c; \\\n"
            + "	c = (a)[7];  (a)[7]  = (a)[13]; (a)[13] = c; \\\n"
            + "	c = (a)[11]; (a)[11] = (a)[14]; (a)[14] = c; \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel16(a,dir) \\\n"
            + "{ \\\n"
            + "    const float w0 = 0x1.d906bcp-1f; \\\n"
            + "    const float w1 = 0x1.87de2ap-2f; \\\n"
            + "    const float w2 = 0x1.6a09e6p-1f; \\\n"
            + "    fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \\\n"
            + "    fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \\\n"
            + "    fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \\\n"
            + "    fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \\\n"
            + "    (a)[5]  = complexMul((a)[5], (float2)(w0, dir*w1)); \\\n"
            + "    (a)[6]  = complexMul((a)[6], (float2)(w2, dir*w2)); \\\n"
            + "    (a)[7]  = complexMul((a)[7], (float2)(w1, dir*w0)); \\\n"
            + "    (a)[9]  = complexMul((a)[9], (float2)(w2, dir*w2)); \\\n"
            + "    (a)[10] = (float2)(dir)*(conjTransp((a)[10])); \\\n"
            + "    (a)[11] = complexMul((a)[11], (float2)(-w2, dir*w2)); \\\n"
            + "    (a)[13] = complexMul((a)[13], (float2)(w1, dir*w0)); \\\n"
            + "    (a)[14] = complexMul((a)[14], (float2)(-w2, dir*w2)); \\\n"
            + "    (a)[15] = complexMul((a)[15], (float2)(-w0, dir*-w1)); \\\n"
            + "    fftKernel4((a), dir); \\\n"
            + "    fftKernel4((a) + 4, dir); \\\n"
            + "    fftKernel4((a) + 8, dir); \\\n"
            + "    fftKernel4((a) + 12, dir); \\\n"
            + "    bitreverse4x4((a)); \\\n"
            + "}\n"
            + "\n"
            + "#define bitreverse32(a) \\\n"
            + "{ \\\n"
            + "    float2 c1, c2; \\\n"
            + "    c1 = (a)[2];   (a)[2] = (a)[1];   c2 = (a)[4];   (a)[4] = c1;   c1 = (a)[8];   (a)[8] = c2;    c2 = (a)[16];  (a)[16] = c1;   (a)[1] = c2; \\\n"
            + "    c1 = (a)[6];   (a)[6] = (a)[3];   c2 = (a)[12];  (a)[12] = c1;  c1 = (a)[24];  (a)[24] = c2;   c2 = (a)[17];  (a)[17] = c1;   (a)[3] = c2; \\\n"
            + "    c1 = (a)[10];  (a)[10] = (a)[5];  c2 = (a)[20];  (a)[20] = c1;  c1 = (a)[9];   (a)[9] = c2;    c2 = (a)[18];  (a)[18] = c1;   (a)[5] = c2; \\\n"
            + "    c1 = (a)[14];  (a)[14] = (a)[7];  c2 = (a)[28];  (a)[28] = c1;  c1 = (a)[25];  (a)[25] = c2;   c2 = (a)[19];  (a)[19] = c1;   (a)[7] = c2; \\\n"
            + "    c1 = (a)[22];  (a)[22] = (a)[11]; c2 = (a)[13];  (a)[13] = c1;  c1 = (a)[26];  (a)[26] = c2;   c2 = (a)[21];  (a)[21] = c1;   (a)[11] = c2; \\\n"
            + "    c1 = (a)[30];  (a)[30] = (a)[15]; c2 = (a)[29];  (a)[29] = c1;  c1 = (a)[27];  (a)[27] = c2;   c2 = (a)[23];  (a)[23] = c1;   (a)[15] = c2; \\\n"
            + "}\n"
            + "\n"
            + "#define fftKernel32(a,dir) \\\n"
            + "{ \\\n"
            + "    fftKernel2S((a)[0],  (a)[16], dir); \\\n"
            + "    fftKernel2S((a)[1],  (a)[17], dir); \\\n"
            + "    fftKernel2S((a)[2],  (a)[18], dir); \\\n"
            + "    fftKernel2S((a)[3],  (a)[19], dir); \\\n"
            + "    fftKernel2S((a)[4],  (a)[20], dir); \\\n"
            + "    fftKernel2S((a)[5],  (a)[21], dir); \\\n"
            + "    fftKernel2S((a)[6],  (a)[22], dir); \\\n"
            + "    fftKernel2S((a)[7],  (a)[23], dir); \\\n"
            + "    fftKernel2S((a)[8],  (a)[24], dir); \\\n"
            + "    fftKernel2S((a)[9],  (a)[25], dir); \\\n"
            + "    fftKernel2S((a)[10], (a)[26], dir); \\\n"
            + "    fftKernel2S((a)[11], (a)[27], dir); \\\n"
            + "    fftKernel2S((a)[12], (a)[28], dir); \\\n"
            + "    fftKernel2S((a)[13], (a)[29], dir); \\\n"
            + "    fftKernel2S((a)[14], (a)[30], dir); \\\n"
            + "    fftKernel2S((a)[15], (a)[31], dir); \\\n"
            + "    (a)[17] = complexMul((a)[17], (float2)(0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \\\n"
            + "    (a)[18] = complexMul((a)[18], (float2)(0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \\\n"
            + "    (a)[19] = complexMul((a)[19], (float2)(0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \\\n"
            + "    (a)[20] = complexMul((a)[20], (float2)(0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \\\n"
            + "    (a)[21] = complexMul((a)[21], (float2)(0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \\\n"
            + "    (a)[22] = complexMul((a)[22], (float2)(0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \\\n"
            + "    (a)[23] = complexMul((a)[23], (float2)(0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \\\n"
            + "    (a)[24] = complexMul((a)[24], (float2)(0x0p+0f, dir*0x1p+0f)); \\\n"
            + "    (a)[25] = complexMul((a)[25], (float2)(-0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \\\n"
            + "    (a)[26] = complexMul((a)[26], (float2)(-0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \\\n"
            + "    (a)[27] = complexMul((a)[27], (float2)(-0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \\\n"
            + "    (a)[28] = complexMul((a)[28], (float2)(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \\\n"
            + "    (a)[29] = complexMul((a)[29], (float2)(-0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \\\n"
            + "    (a)[30] = complexMul((a)[30], (float2)(-0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \\\n"
            + "    (a)[31] = complexMul((a)[31], (float2)(-0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \\\n"
            + "    fftKernel16((a), dir); \\\n"
            + "    fftKernel16((a) + 16, dir); \\\n"
            + "    bitreverse32((a)); \\\n"
            + "}\n\n";
    static String twistKernelInterleaved =
            "__kernel void \\\n"
            + "clFFT_1DTwistInterleaved(__global float2 *in, unsigned int startRow, unsigned int numCols, unsigned int N, unsigned int numRowsToProcess, int dir) \\\n"
            + "{ \\\n"
            + "   float2 a, w; \\\n"
            + "   float ang; \\\n"
            + "   unsigned int j; \\\n"
            + "	unsigned int i = get_global_id(0); \\\n"
            + "	unsigned int startIndex = i; \\\n"
            + "	 \\\n"
            + "	if(i < numCols) \\\n"
            + "	{ \\\n"
            + "	    for(j = 0; j < numRowsToProcess; j++) \\\n"
            + "	    { \\\n"
            + "	        a = in[startIndex]; \\\n"
            + "	        ang = 2.0f * M_PI * dir * i * (startRow + j) / N; \\\n"
            + "	        w = (float2)(native_cos(ang), native_sin(ang)); \\\n"
            + "	        a = complexMul(a, w); \\\n"
            + "	        in[startIndex] = a; \\\n"
            + "	        startIndex += numCols; \\\n"
            + "	    } \\\n"
            + "	}	 \\\n"
            + "} \\\n";
    static String twistKernelPlannar =
            "__kernel void \\\n"
            + "clFFT_1DTwistSplit(__global float *in_real, __global float *in_imag , unsigned int startRow, unsigned int numCols, unsigned int N, unsigned int numRowsToProcess, int dir) \\\n"
            + "{ \\\n"
            + "    float2 a, w; \\\n"
            + "    float ang; \\\n"
            + "    unsigned int j; \\\n"
            + "	unsigned int i = get_global_id(0); \\\n"
            + "	unsigned int startIndex = i; \\\n"
            + "	 \\\n"
            + "	if(i < numCols) \\\n"
            + "	{ \\\n"
            + "	    for(j = 0; j < numRowsToProcess; j++) \\\n"
            + "	    { \\\n"
            + "	        a = (float2)(in_real[startIndex], in_imag[startIndex]); \\\n"
            + "	        ang = 2.0f * M_PI * dir * i * (startRow + j) / N; \\\n"
            + "	        w = (float2)(native_cos(ang), native_sin(ang)); \\\n"
            + "	        a = complexMul(a, w); \\\n"
            + "	        in_real[startIndex] = a.x; \\\n"
            + "	        in_imag[startIndex] = a.y; \\\n"
            + "	        startIndex += numCols; \\\n"
            + "	    } \\\n"
            + "	}	 \\\n"
            + "} \\\n";
}
