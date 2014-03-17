Original code by Eero Simoncelli <http://www.cns.nyu.edu/~eero/>, 2/97.
Ported by Dzung Nguyen <dzungng89@gmail.com>, 04/2012

A simplified version of steerable pyramid, where the steerable pyramid 
is organized into cell structure for easy access. 
For example, we can access different subbands
    
    highpass:   coeff{1} 
    subbands:   coeff{2}{1}
    lowpass:    coeff{ht}
 
Include three versions as in original code

1) Steerable pyramid in time domain 
    coeff=buildSpyr(im,5,'sp3.mat');
    out=reconSpyr(coeff,'sp3.mat');

2) Steerable pyramid in frequency domain
    coeff=buildSFpyr(im,5);
    out=reconSFpyr(coeff);

3) Complex steerable pyramid in frequency domain
    coeff=buildSCFpyr(im,5);
    out=reconSCFpyr(coeff);

Show the pyramid using the showsteerable function