
 --------------------------------------
 BASIC VIDEO TOOLS  (version 2)
 Jesus Malo            jesus.malo@uv.es 
 Juan Gutierrez
 Valero Laparra
 (c) Universitat de Valencia. 1996-2015
 --------------------------------------
 
 (1) What is in this toolbox?
 (2) What is not in here?
 (3) Installation and requirements
 (4) How to get started?
 (5) License & Acknowledgements

 ..................................
 (1) WHAT IS IN BASIC VIDEO TOOLS?
 ..................................
 
 BasicVideoTools is a Matlab Toolbox to deal with video data, apply spatio-temporal filters, 
 and work with models of motion-sensitive neurons.
 In particular, it includes convenient *.m files to:
 
 - Read standard (VQEG and LIVE) video data 
 - Rearrange video data (as for instance to perform statistical analysis)
 - Generate controlled sequences (controlled contrast, texture, and 2d and 3d speed)
 - Compute and visualize 3D Fourier transforms
 - Play with perceptually meaningful spatio-temporal filters: 
   (receptive fileds of LGN, V1 and MT cells, and spatio-temporal Contrast Sensitivity Functions)
 - Visualize movies (achromatic only)

 .....................................
 (2) What IS NOT in BasicVideoTools ?
 .....................................

 BasicVideoTools does not include: 
 - Motion estimation/compensation algorithms  
 - Video Coding algorithms
 - Video Quality Mesures 
 
 If you are looking for the above, please consider downloading other Toolboxes:

 - Motion estimation: http://isp.uv.es/Video_coding.html  (Hierarchical Block Matching)
                      http://www.scholarpedia.org/article/Optic_flow

 - Video Coding (improved MPEG): http://isp.uv.es/Video_coding.html  

 - Video Quality:     http://isp.uv.es/Video_quality.html

 .....................................
 (3) INSTALLATION AND REQUIREMENTS
 .....................................

 - Download BasicVideoTools_code.zip from http://isp.uv.es/Basic_Video.html
 - Decompress at your machine in the folder BasicVideoTools (no location restrictions for this folder)
 - Update the matlab/octave path including all subfolders

 - Tested on Matlab 2006b

 * Video and image data are given in separate (and optional) files to 
   simplify the download process (big files!)
   These databases are useful to gather video statistics
         image_data.zip
         video_data.zip 
          
   If you use these data please cite the VQEG and LIVE databases (for video), 
   and the CVC Barcelona Database (for images) 

 .....................................
 (4) HOW TO GET STARTED?
 .....................................

 For a general overview please take a look at the contents.m file: just type, help BasicVideoTools
 For additional details on how to use the functions in practice, see the demos:

  demo_motion_programs         - Demo on how to use most functions 
                                 (except random dots and newtonian sequences)
  example_random_dots_sequence - Demo on random dots sequences with controlled flow
  example_newtonian_sequence   - Demo on physics-controlled sequences

 .....................................
 (5) LICENSE & ACKNOWLEDGEMENTS
 .....................................

Copyright (c) 2015, Jesus Malo. All rights reserved.

We thank Avan Suinesiaputra and Fadillah Z Tala by making public the functions ndgauss.m and hermite.m 
which we used in the model of LGN cells to compute derivatives of n-dimensional Gaussian kernels.

Copyright (c) 2010, Avan Suinesiaputra. All rights of ndgauss.m and hermite.m reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notices, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notices, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
