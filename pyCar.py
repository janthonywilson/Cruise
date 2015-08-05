'''
Created on Feb 13, 2015

@author: Tony
'''

import os
import cv2
import numpy
from skimage import filter, morphology

#===================================================================================
# pyLaneTracker CLASS
#===================================================================================
class pyLaneTracker(object):
    """Main Lane Tracking Interfacing Class"""
    def __init__( self, OutputDir ):
        self.Img = []
        self.FileName = []
        self.Gamma = 1.0
        self.Gabor = []
        self.GaborOrentatios = []
        self.Sobel = []
        self.Blur = []
        self.Denoise = []
        self.SobelTheta = []
        self.Threshold = []
        self.Method = []
        self.ImageStats = []
        self.Slopes = []
        self.Coordinates = []
        self.OutputDir = OutputDir
        
        # Across images data...
        self.FileNames = []
        self.Intercepts = []
    #===============================================================================
    def normalizeImage(self):
        """Normalize the image"""
        ImgPow = numpy.power( self.Img, self.Gamma )
        ImgMin = min(ImgPow[ImgPow > 0].ravel())
        ImgStretch = (ImgPow - ImgMin)
        ImgNorm = ImgStretch / numpy.max(ImgStretch.ravel())
        ImgNorm[numpy.nonzero( ImgNorm > 1 )] = 1
        ImgNorm[numpy.nonzero( ImgNorm < 0 )] = 0
        ImgNorm *= 255
        self.Img = ImgNorm
    #===============================================================================   
    def genSobel(self):
        """Generate Sobel from x and y gradients"""
        if (len(self.Img.shape) > 2):
            Dim = 1
            ImgGradient = numpy.gradient(self.Img[:,:,Dim])
        else:
            ImgGradient = numpy.gradient(self.Img)
            
        self.Sobel = numpy.sqrt( (ImgGradient[0]**2) + ImgGradient[1]**2 )
        self.SobelTheta = numpy.rad2deg(numpy.arctan2(ImgGradient[0], ImgGradient[1]))
    #===============================================================================
    def thresholdImage(self):
        """Threshold Image"""
        ### THRESHOLDING ###
        #=======================================================================
        # self.ThresholdMethod:
        # thresholdGlobalOtsu = filter.threshold_otsu(pyLaneTracker.Img, 64)
        # thresholdGlobalYen = filter.threshold_yen(pyLaneTracker.Img, 64)
        #=======================================================================
        
        thresholdAdaptive = filter.threshold_adaptive(self.Img, 96, method='median', offset=0, mode='reflect', param=None)
        self.Threshold = thresholdAdaptive
    #===============================================================================
    def GaborKernel(self, ImageShape, Lambda, Theta, Psi, Sigma, Gamma):
        """Gabor kernel for filtering (not currently used)"""
        #=======================================================================
        # LAMBDA represents the wavelength of the sinusoidal factor, 
        # THETA represents the orientation of the normal to the parallel stripes of a Gabor function, 
        # PSI is the phase offset, 
        # SIGMA is the sigma/standard deviation of the Gaussian envelope 
        # GAMMA is the spatial aspect ratio
        #=======================================================================
        sigmaX = Sigma;
        sigmaY = Sigma/Gamma;
    
        CenterPoint = ImageShape / 2
        ndMesh =  numpy.asarray(numpy.mgrid[0:ImageShape[0],0:ImageShape[1]], dtype='float')
        xGrid = ( (ndMesh[0,:] - CenterPoint[0]) )
        yGrid = ( (ndMesh[1,:] - CenterPoint[1]) ) 
        
        thetaX = xGrid * numpy.cos(Theta) + yGrid * numpy.sin(Theta);
        thetaY = -xGrid * numpy.sin(Theta) + yGrid * numpy.cos(Theta);
        
        kernel = numpy.exp(-0.5*(thetaX ** 2 / sigmaX ** 2 + thetaY ** 2 / sigmaY ** 2)) * numpy.cos(2 * numpy.pi / Lambda * thetaX + Psi);
        return kernel
    #===============================================================================
    def gaborFilter(self):
        """Filter image with a series of Gabors"""
        Gabor = numpy.zeros(self.Img.shape)
        fftImg = numpy.fft.fft2(self.Img)
        kernelSize = (self.Img.shape[1], self.Img.shape[0])
        gaborSigma = 16
        gaborWavelength = 34
        gaborAspect = 0.15
        gaborPhase = 0
        gaborOrientStep = 15
        
        for i in xrange(gaborOrientStep, 180 - gaborOrientStep, gaborOrientStep):
            if (i != 90):
                gaborOrient = i * numpy.pi / 180
                Kernel = cv2.getGaborKernel(kernelSize, gaborSigma, gaborOrient, gaborWavelength, gaborAspect, gaborPhase)
                KernelImage = numpy.fft.ifft2( fftImg * numpy.fft.fft2(Kernel) )
                KernelImagePow = numpy.real( numpy.fft.fftshift( KernelImage * numpy.conj(KernelImage) ) )
                Gabor += (KernelImagePow / numpy.max(KernelImagePow.ravel()))

        self.Gabor = Gabor
    #=============================================================================== 
    def GaussianKernel(self, ImageShape, Lambda, Theta, Psi, Sigma, Gamma):
        """Gaussian kernel for blurring"""
        CenterPoint = ImageShape / 2
        ndMesh =  numpy.asarray(numpy.mgrid[0:ImageShape[0],0:ImageShape[1]], dtype='float')
        xGrid = numpy.sqrt( (ndMesh[0,:] - CenterPoint[0]) ** 2 )
        yGrid = numpy.sqrt( (ndMesh[1,:] - CenterPoint[1]) ** 2 ) 
    
        kernel = (numpy.exp(-(((xGrid ** 2)+(yGrid ** 2)))/(Sigma ** 2)));
        return kernel
    #===============================================================================
    def blurImage(self):
        """Blur image using a Gaussian"""
        kernelSize = numpy.array((self.Img.shape[0], self.Img.shape[1]))
        gaussSigma = 2

        Kernel = self.GaussianKernel(kernelSize, gaussSigma)
        KernelImage = numpy.fft.fftshift( numpy.fft.ifft2( numpy.fft.fft2(self.Img) * numpy.fft.fft2(Kernel) ) )

        self.Blur = KernelImage
    #===============================================================================   
    def denoiseImage(self):
        """Denoise image using a method (TODO: add more options)"""
        
        self.Denoise = filter.rank.median( self.Img.astype(numpy.uint8), morphology.disk(5) )
    #===============================================================================    
    def writeIntersections(self):
        """Write intesections and bunch of other stuff"""
        self.OutputDir = os.path.abspath(self.OutputDir)
        if not os.path.exists(self.OutputDir):
            os.makedirs(self.OutputDir)
        with open(self.OutputDir +os.sep+ 'Intercepts.txt', 'wb') as outputFileObj:
            outputFileObj.write('File Name\tLeft Intercept\tRight Intercept\n')
            
            if (len(self.Intercepts) != len(self.FileNames)):
                print 'ERROR: Can not print.  Intercepts does not match File Names...'
                
            for i in xrange(0,len(self.Intercepts)):
                outputFileObj.write(self.FileNames[i] +'\t'+ self.Intercepts[i,0].astype('str') +'\t'+ self.Intercepts[i,1].astype('str') + '\n')
#===================================================================================       



