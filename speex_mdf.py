# The echo canceller is based on the MDF algorithm described in:
# 
# J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter, 
# IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2, 
# February 1990.
# 
# We use the Alternatively Updated MDF (AUMDF) variant. Robustness to 
# double-talk is achieved using a variable learning rate as described in:
# 
# Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo 
# Cancellation With Double-Talk. IEEE Transactions on Audio,
# Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
# http://people.xiph.org/~jm/papers/valin_taslp2006.pdf
# 
# There is no explicit double-talk detection, but a continuous variation
# in the learning rate based on residual echo, double-talk and background
# noise.
# 
# Another kludge that seems to work good: when performing the weight
# update, we only move half the way toward the "goal" this seems to
# reduce the effect of quantization noise in the update phase. This
# can be seen as applying a gradient descent on a "soft constraint"
# instead of having a hard constraint.
# 
# Notes for this file:
#
# Usage: 
#
#    processor = MDF(Fs, frame_size, filter_length)
#    processor.main_loop(u, d)
#    
#    Fs                  sample rate
#    u                   speaker signal, vector in range [-1; 1]
#    d                   microphone signal, vector in range [-1; 1]
#    filter_length       typically 250ms, i.e. 4096 @ 16k FS
#                        must be a power of 2
#    frame_size          typically 8ms, i.e. 128 @ 16k Fs
#                        must be a power of 2
#
# Shimin Zhang <shmzhang@npu-aslp.org>
# 

import numpy as np
import time

def float_to_short(x):
    x = x*32768.0
    x[x < -32767.5] = -32768
    x[x > 32766.5] = 32767
    x = np.floor(0.5+x)
    return x

class MDF:
    def __init__(self, fs: int, frame_size: int, filter_length: int ) -> None:
        nb_mic = 1
        nb_speakers = 1
        self.K = nb_speakers
        K = self.K
        self.C = nb_mic
        C = self.C

        self.frame_size = frame_size
        self.filter_length = filter_length
        self.window_size = frame_size*2
        N = self.window_size
        self.M = int(np.fix((filter_length+frame_size-1)/frame_size))
        M = self.M
        self.cancel_count = 0
        self.sum_adapt = 0
        self.saturated = 0
        self.screwed_up = 0

        self.sampling_rate = fs
        self.spec_average = (self.frame_size)/(self.sampling_rate)
        self.beta0 = (2.0*self.frame_size)/self.sampling_rate
        self.beta_max = (.5*self.frame_size)/self.sampling_rate
        self.leak_estimate = 0

        self.e = np.zeros((N, C),)
        self.x = np.zeros((N, K),)
        self.input = np.zeros((self.frame_size, C),)
        self.y = np.zeros((N, C),)
        self.last_y = np.zeros((N, C),)
        self.Yf = np.zeros((self.frame_size+1, 1),)
        self.Rf = np.zeros((self.frame_size+1, 1),)
        self.Xf = np.zeros((self.frame_size+1, 1),)
        self.Yh = np.zeros((self.frame_size+1, 1),)
        self.Eh = np.zeros((self.frame_size+1, 1),)

        self.X = np.zeros((N, K, M+1), dtype=np.complex)
        self.Y = np.zeros((N, C), dtype=np.complex)
        self.E = np.zeros((N, C), dtype=np.complex)
        self.W = np.zeros((N, K, M, C), dtype=np.complex)
        self.foreground = np.zeros((N, K, M, C), dtype=np.complex)
        self.PHI = np.zeros((frame_size+1, 1),)
        self.power = np.zeros((frame_size+1, 1),)
        self.power_1 = np.ones((frame_size+1, 1),)
        self.window = np.zeros((N, 1),)
        self.prop = np.zeros((M, 1),)
        self.wtmp = np.zeros((N, 1),)
        self.window = .5-.5 * \
            np.cos(2*np.pi*(np.arange(1, N+1).reshape(-1, 1)-1)/N)
        decay = np.exp(-2.4/M)
        self.prop[0, 0] = .7
        for i in range(1, M):
            self.prop[i, 0] = self.prop[i-1, 0]*decay
        self.prop = (.8 * self.prop)/np.sum(self.prop)

        self.memX = np.zeros((K, 1),)
        self.memD = np.zeros((C, 1),)
        self.memE = np.zeros((C, 1),)
        self.preemph = .9
        if self.sampling_rate < 12000:
            self.notch_radius = .9
        elif self.sampling_rate < 24000:
            self.notch_radius = .982
        else:
            self.notch_radius = .992
        self.notch_mem = np.zeros((2*C, 1),)
        self.adapted = 0
        self.Pey = 1
        self.Pyy = 1
        self.Davg1 = 0
        self.Davg2 = 0
        self.Dvar1 = 0
        self.Dvar2 = 0
        
        self.Sx=np.zeros((frame_size*2,1),)
        self.Se=np.zeros((frame_size*2,1),)
        self.Sd=np.zeros((frame_size*2,1),)
        self.Sxd=np.zeros((frame_size*2,1),)
        self.Sed=np.zeros((frame_size*2,1),)
        
        
        self.suppState = 1
        self.cohxdLocalMin = 1
        self.hnlLocalMin = 1
        self.mult=fs/8000
        self.ovrdSm=2
        self.divergeState = 0
        self.hnlNewMin = 1
        self.hnlMinCtr = 0
        self.hnlMin=1
        self.ovrd=2
        
        

    def main_loop(self, u, d):
        """MDF core function

        Args:
            u (array): reference signal
            d (array): microphone signal
        """
        assert u.shape == d.shape
        u = float_to_short(u)
        d = float_to_short(d)

        e = np.zeros_like(u)
        y = np.zeros_like(u)
        end_point = len(u)
        #print(self.frame_size)

        for n in range(0, end_point, self.frame_size):
            nStep = np.floor(n/self.frame_size) + 1
            self.nStep = nStep
            # the break operation not understand.
            # only for signal channel AEC
            if n+self.frame_size > end_point:
                break
            u_frame = u[n:n+self.frame_size] #first round:0:128 , second:128:256
            d_frame = d[n:n+self.frame_size]
            out = self.speex_echo_cancellation_mdf(d_frame[:, None], u_frame[:, None])[:,0]
            y[n:n+self.frame_size] = out #y out
            e[n:n+self.frame_size] = d_frame - out #far end
        e = e/32768.0
        y = y/32768.0
        return e, y

    def speex_echo_cancellation_mdf(self, mic, far_end):
        N = self.window_size  #256
        M = self.M  #32
        C = self.C  #1
        K = self.K  #1
        #print(N,M,C,K)

        Pey_cur = 1
        Pyy_cur = 1

        out = np.zeros((self.frame_size, C),)  #128*1
        self.cancel_count += 1

        ss = .35/M
        ss_1 = 1 - ss
        #print(np.shape(mic))
        for chan in range(C):
            # Apply a notch filter to make sure DC doesn't end up causing problems
            self.input[:, chan], self.notch_mem[:, chan] = self.filter_dc_notch16(
                mic[:, chan], self.notch_mem[:, chan])
            #print(np.shape(self.input))  128*1
            #Copy input data to buffer and apply pre-emphasis
            for i in range(self.frame_size): #做128次
                tmp32 = self.input[i, chan] - (np.dot(self.preemph, self.memD[chan]))
                self.memD[chan] = self.input[i, chan]
                self.input[i, chan] = tmp32

        for speak in range(K):
            for i in range(self.frame_size):
                self.x[i, speak] = self.x[i+self.frame_size, speak]
                tmp32 = far_end[i, speak] - np.dot(self.preemph, self.memX[speak])
                self.x[i+self.frame_size, speak] = tmp32
                self.memX[speak] = far_end[i, speak]
        
        # self.X = np.roll(self.X, [0, 0, 1])
        #朝第三方向移動一個位置，為了餵下一個資料進來
        self.X = np.roll(self.X, 1, axis=2)
        
        #print(np.shape(self.x))
        #Convert x (echo input) to frequency domain
        #we divide by N to get values as in speex
        for speak in range(K):
            self.X[:, speak, 0] = np.fft.fft(self.x[:, speak])/N     
        
        Sxx = 0
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak]**2)
            self.Xf = np.abs(self.X[:self.frame_size+1, speak, 0])**2
        #print(np.shape(self.foreground))
        #print(np.shape(self.X))
        Sff = 0
        for chan in range(C):
            #Compute foreground filter
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + self.X[:,speak, j]*self.foreground[:, speak, j, chan]    #全部時間項加到y
            self.e[:, chan] = np.fft.ifft(self.Y[:, chan]).real * N  #we multiply by N to get values as in speex
            self.e[:self.frame_size, chan] = self.input[:, chan] - self.e[self.frame_size:, chan]      #計算e paper(4)
            #st.e : [out foreground | leak foreground ]
            Sff = Sff + np.sum(np.abs(self.e[:self.frame_size, chan])**2)
        #Adjust proportional adaption rate
        if self.adapted:
            self.prop = self.mdf_adjust_prop()
        #Compute weight gradient
        if self.saturated == 0:
            for chan in range(C):
                for speak in range(K):
                    for j in list(range(M)[::-1]):
                        #print(np.shape(self.power_1))
                        #print(np.shape(self.power_1[-2:0:-1]))
                        #print(np.shape(np.concatenate([self.power_1, self.power_1[-2:0:-1]], axis=0)))
                        #print(np.shape(np.conj(self.X[:, speak, j+1])[:,None]))
                        #print(np.shape(self.prop[j]))
                        self.PHI = np.concatenate([self.power_1, self.power_1[-2:0:-1]], axis=0) * self.prop[j] * np.conj(self.X[:, speak, j+1])[:,None] * self.E[:, chan][:,None]   #phi paper(5)  前面沒E 
                        self.W[:,speak,j,chan] = self.W[:,speak,j,chan]+self.PHI[:,0]  #paper(7)
                
        else:
            self.saturated -= 1
        #print(np.shape(self.PHI))
        #Update weight to prevent circular convolution(?) (MDF / AUMDF)
        for chan in range(C):
            for speak in range(K):
                for j in range(M):
                    if j == 0 or (2+self.cancel_count) % (M-1) == j:
                        self.wtmp = np.fft.ifft(self.W[:, speak, j, chan]).real
                        self.wtmp[self.frame_size:N] = 0
                        self.W[:, speak, j, chan] = np.fft.fft(self.wtmp)
        #So we can use power_spectrum_accum
        self.Yf = np.zeros((self.frame_size+1, 1),)  #129*1
        self.Rf = np.zeros((self.frame_size+1, 1),)
        self.Xf = np.zeros((self.frame_size+1, 1),)

        Dbf = 0
        for chan in range(C):
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + self.X[:, speak, j] * self.W[:, speak, j, chan]  #32個時間帶的x加成y
            #we multiply by N to get values as in speex
            self.y[:, chan] = np.fft.ifft(self.Y[:, chan]).real*N
            #st.y : [ ~ | leak background ] (?)
#w end?        
        #print(np.shape(self.y))
        
        See = 0
        #Difference in response, this is used to estimate the variance of our residual power estimate  e分前後半
        for chan in range(C):
            self.e[:self.frame_size, chan] = self.e[self.frame_size:N,chan]-self.y[self.frame_size:N, chan]

                
            Dbf = Dbf + 10 + np.sum(np.abs(self.e[:self.frame_size, chan])**2)
            self.e[:self.frame_size, chan] = self.input[:, chan] - self.y[self.frame_size:N, chan]
            See = See + np.sum(np.abs(self.e[:self.frame_size, chan])**2)
            
        # Logic for updating the foreground filter    
        # For two time windows, compute the mean of the energy difference, as well as the variance
        VAR1_UPDATE = .5
        VAR2_UPDATE = .25
        VAR_BACKTRACK = 4
        MIN_LEAK = .005

        self.Davg1 = .6*self.Davg1 + .4*(Sff-See)
        self.Dvar1 = .36*self.Dvar1 + .16*Sff*Dbf
        self.Davg2 = .85*self.Davg2 + .15*(Sff-See)
        self.Dvar2 = .7225*self.Dvar2 + .0225*Sff*Dbf

        update_foreground = 0
        #Check if we have a statistically significant reduction in the residual echo
        #Note that this is *not* Gaussian, so we need to be careful about the longer tail
        if (Sff-See)*abs(Sff-See) > (Sff*Dbf):
            update_foreground = 1
        elif (self.Davg1 * abs(self.Davg1) > (VAR1_UPDATE*self.Dvar1)):
            update_foreground = 1
        elif (self.Davg2 * abs(self.Davg2) > (VAR2_UPDATE*(self.Dvar2))):
            update_foreground = 1

        if update_foreground:
            self.Davg1 = 0
            self.Davg2 = 0
            self.Dvar1 = 0
            self.Dvar2 = 0
            self.foreground = self.W  #update(?)
            #Apply a smooth transition so as to not introduce blocking artifacts
            for chan in range(C):
                self.e[self.frame_size:N, chan] = (self.window[self.frame_size:N][:,0] * self.e[self.frame_size:N, chan]) + (self.window[:self.frame_size][:,0] * self.y[self.frame_size:N, chan])   #e=e+y
        else:
            reset_background = 0
            #Otherwise, check if the background filter is significantly worse
            if (-(Sff-See)*np.abs(Sff-See) > VAR_BACKTRACK*(Sff*Dbf)):
                reset_background = 1
            if ((-self.Davg1 * np.abs(self.Davg1)) > (VAR_BACKTRACK*self.Dvar1)):
                reset_background = 1
            if ((-self.Davg2 * np.abs(self.Davg2)) > (VAR_BACKTRACK*self.Dvar2)):
                reset_background = 1

            if reset_background:
                #Copy foreground filter to background filter
                self.W = self.foreground
                #We also need to copy the output so as to get correct adaptation
                for chan in range(C):

                    self.y[self.frame_size:N,
                           chan] = self.e[self.frame_size:N, chan]
                    self.e[:self.frame_size, chan] = self.input[:,
                                                                chan] - self.y[self.frame_size:N, chan]
                See = Sff
                self.Davg1 = 0
                self.Davg2 = 0
                self.Dvar1 = 0
                self.Dvar2 = 0

        Sey = 0
        Syy = 0
        Sdd = 0

        for chan in range(C):
            #Compute error signal (for the output with de-emphasis)
            for i in range(self.frame_size):
                tmp_out = self.input[i, chan] - self.e[i+self.frame_size, chan]
                tmp_out = tmp_out + self.preemph * self.memE[chan]
                # This is an arbitrary test for saturation in the microphone signal
                if mic[i, chan] <= -32000 or mic[i, chan] >= 32000:
                    if self.saturated == 0:
                        self.saturated = 1
                out[i, chan] = tmp_out[0]   #output result!?
                self.memE[chan] = tmp_out
#MDF end?
            #Compute error signal (filter update version)
            self.e[self.frame_size:N, chan] = self.e[:self.frame_size, chan]
            self.e[:self.frame_size, chan] = 0
            #st.e : [ zeros | out background ]
            
            #Compute a bunch of correlations
            #FIXME: bad merge
            Sey = Sey + np.sum(self.e[self.frame_size:N, chan]
                               * self.y[self.frame_size:N, chan])
            Syy = Syy + np.sum(self.y[self.frame_size:N, chan]**2)
            Sdd = Sdd + np.sum(self.input**2)

            #Convert error to frequency domain
            #MATLAB_MATCH: we divide by N to get values as in speex
            self.E = np.fft.fft(self.e,axis=0) / N

            self.y[:self.frame_size, chan] = 0
            self.Y = np.fft.fft(self.y,axis=0) / N
            
            #Compute power spectrum of echo (X), error (E) and filter response (Y)
            self.Rf = np.abs(self.E[:self.frame_size+1, chan])**2
            self.Yf = np.abs(self.Y[:self.frame_size+1, chan])**2
        
        #try nlp
        if self.sampling_rate==8000:
            gamma=0.9
        else:
            gamma=0.93
        
        
        tmpd=np.zeros((self.frame_size*2,1),)
        tmpd[self.frame_size:]=self.input
        tmpdf= np.fft.fft(tmpd,axis=0) / N
        
        tmpx=np.zeros((self.frame_size*2,1),)
        tmpx[self.frame_size:]=self.x[self.frame_size:,]
        tmpxf= np.fft.fft(tmpx,axis=0) / N
        
        tmpef=self.E
        #print(np.shape(self.E))
        #print(np.shape(self.input))
        #print(np.shape(self.x[self.frame_size:,]))
        #print(np.shape(self.e[self.frame_size:,]))
        
        self.Se = gamma*self.Se + (1-gamma)*np.real(tmpef*np.conj(tmpef))
        self.Sd = gamma*self.Sd + (1-gamma)*np.real(tmpdf*np.conj(tmpdf))
        self.Sx = gamma*self.Sx + (1-gamma)*np.real(tmpxf*np.conj(tmpxf))
        

        self.Sxd = gamma*self.Sxd + (1 - gamma)*tmpxf*np.conj(tmpdf)
        self.Sed = gamma*self.Sed + (1-gamma)*tmpef*np.conj(tmpdf)

        cohed = np.real(self.Sed*np.conj(self.Sed))/(self.Se*self.Sd + 1e-10)
        cohxd = np.real(self.Sxd*np.conj(self.Sxd))/(self.Sx*self.Sd + 1e-10)
        
        hnled = np.minimum(1 - cohxd, cohed)
        
        #self.E=self.E*hnled
        #echobandrange=np.arange(self.frame_size)
        
        cohedMean=np.mean(cohed[:self.frame_size])
        hnlsort=np.sort(1-cohxd[:self.frame_size])
        xsort=np.sort(self.Sx)
        
        hnlSortQ = np.mean(1 - cohxd[:self.frame_size])
        
        hnlSort2 = np.sort(hnled[:self.frame_size])
        
        hnlQuant = 0.75
        hnlQuantLow = 0.5
        
        qIdx = int(np.floor(hnlQuant*self.frame_size))
        qIdxLow = int(np.floor(hnlQuantLow*self.frame_size))
        
        hnlPrefAvg = hnlSort2[qIdx]
        hnlPrefAvgLow = hnlSort2[qIdxLow]
        
        if cohedMean > 0.98 and hnlSortQ > 0.9:
            self.suppState = 0
        elif cohedMean < 0.95 or hnlSortQ < 0.8:
            self.suppState = 1
        
        if hnlSortQ < self.cohxdLocalMin and hnlSortQ < 0.75:
            self.cohxdLocalMin = hnlSortQ
        
        
        if self.cohxdLocalMin == 1: #small echo
            ovrd = 3
            hnled = 1 - cohxd
            hnlPrefAvg = hnlSortQ
            hnlPrefAvgLow = hnlSortQ
        

        if self.suppState == 0: #no echo
            hnled = cohed
            hnlPrefAvg = cohedMean
            hnlPrefAvgLow = cohedMean
        
        
        #if hnlPrefAvg < hnlLocalMin & hnlPrefAvg < 0.6
        if hnlPrefAvgLow < self.hnlLocalMin and hnlPrefAvgLow < 0.6:
            #hnlLocalMin = hnlPrefAvg;
            #hnlMin = hnlPrefAvg;
            self.hnlLocalMin = hnlPrefAvgLow
            self.hnlMin = hnlPrefAvgLow
            self.hnlNewMin = 1
            self.hnlMinCtr = 0
        
        if self.hnlNewMin == 1:
            self.hnlMinCtr = self.hnlMinCtr + 1
        
        if self.hnlMinCtr == 2:
            self.hnlNewMin = 0
            self.hnlMinCtr = 0
            self.ovrd = max(np.log(0.001)/np.log(self.hnlMin), 8)
        
        self.hnlLocalMin = min(self.hnlLocalMin + 0.0008/self.mult, 1)
        self.cohxdLocalMin = min(self.cohxdLocalMin + 0.0004/self.mult, 1)

        if self.ovrd < self.ovrdSm:
            self.ovrdSm = 0.99*self.ovrdSm + 0.01*self.ovrd
        else:
            self.ovrdSm = 0.9*self.ovrdSm + 0.1*self.ovrd
        
        ekEn = sum(self.Se)
        dkEn = sum(self.Sd)

        #process diverge
        if self.divergeState == 0:
            if ekEn > dkEn:
                tmpef = tmpdf
                self.divergeState = 1
        else:
            if ekEn*1.05 < dkEn:
                self.divergeState = 0
            else:
                tmpef = tmpdf
            
        if ekEn > dkEn*19.95: #filter diverge and set 0 to start again
            self.W = np.zeros((N, K, M, C), dtype=np.complex) #Block-based FD NLMS
        

        
        
        aggrFact = 0.3
        weight = aggrFact*np.sqrt(np.linspace(0, 1,self.frame_size))+0.1
        #solve it by ones
        hnlPrefAvga=np.ones((self.frame_size*2,1),)*hnlPrefAvg
        hnled = weight*np.minimum(hnlPrefAvga, hnled) + (1 - weight)*hnled
        
        
        #         od = ovrdSm*fliplr(sqrt(linspace(0,1,N+1))' + 1)
        od = self.ovrdSm*(np.sqrt(np.linspace(0, 1,self.frame_size)) + 1)
#         sshift = 1.5*ones(N+1,1)
        sshift = np.ones((self.frame_size,1),)
        
        for i in range(self.frame_size):
            hnled[i] = hnled[i]**(od[i]*sshift[i])


        self.E=self.E*hnled
        #tmpef = tmpef*hnled
#         ef = ef.*(min(1 - cohxd, cohed).^2)

        #if (CNon):
        #    snn = sqrt(Sym)
        #    snn(1) = 0      #% Reject LF noise
        #    Un = 10.0*snn.*exp(1j*2*pi.*[0;rand(N-1,1);0])     #increase amplitude
        #    # Weight comfort noise by suppression
        #    Un = sqrt(1 - hnled.^2).*Un
        #    Fmix = ef + Un
        #else
        #    Fmix = ef        
        
        #Do some sanity check
        if not (Syy >= 0 and Sxx >= 0 and See >= 0):
            #Things have gone really bad
            self.screwed_up = self.screwed_up + 50
            out = np.zeros_like(out)
        elif Sff > Sdd + N * 10000:
            #AEC seems to add lots of echo instead of removing it, let's see if it will improve
            self.screwed_up = self.screwed_up + 1
        else:
            #Everything's fine
            self.screwed_up = 0
        if self.screwed_up >= 50:
            print("Screwed up, full reset")
            self.__init__(self.sampling_rate,
                          self.frame_size, self.filter_length)

        #Add a small noise floor to make sure not to have problems when dividing
        See = max(See, N * 100)
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak]**2)
            self.Xf = np.abs(self.X[:self.frame_size+1, speak, 0])**2
        
        #Smooth far end energy estimate over time
        self.power = ss_1*self.power + 1 + ss*self.Xf[:,None]
        
        #Compute filtered spectra and (cross-)correlations
        Eh_cur = self.Rf - self.Eh
        Yh_cur = self.Yf - self.Yh
        Pey_cur = Pey_cur + np.sum(Eh_cur*Yh_cur)
        Pyy_cur = Pyy_cur + np.sum(Yh_cur**2)
        self.Eh = (1-self.spec_average)*self.Eh + self.spec_average*self.Rf
        self.Yh = (1-self.spec_average)*self.Yh + self.spec_average*self.Yf
        Pyy = np.sqrt(Pyy_cur)
        Pey = Pey_cur/Pyy
        
        
        #Compute correlation updatete rate
        tmp32 = self.beta0*Syy
        if tmp32 > self.beta_max*See:
            tmp32 = self.beta_max*See
        alpha = tmp32 / See
        alpha_1 = 1 - alpha
        
        #Update correlations (recursive average)
        self.Pey = alpha_1*self.Pey + alpha*Pey
        self.Pyy = alpha_1*self.Pyy + alpha*Pyy
        if self.Pyy < 1:
            self.Pyy = 1
            
        #We don't really hope to get better than 33 dB (MIN_LEAK-3dB) attenuation anyway
        if self.Pey < MIN_LEAK * self.Pyy:
            self.Pey = MIN_LEAK * self.Pyy
        if self.Pey > self.Pyy:
            self.Pey = self.Pyy
        
        #leak_estimate is the linear regression result
        self.leak_estimate = self.Pey/self.Pyy
        
        #This looks like a stupid bug, but it's right (because we convert from Q14 to Q15)
        if self.leak_estimate > 16383:
            self.leak_estimate = 32767
        
        #Compute Residual to Error Ratio
        RER = (.0001*Sxx + 3.*self.leak_estimate*Syy) / See
        
        #Check for y in e (lower bound on RER)
        if RER < Sey*Sey/(1+See*Syy):
            RER = Sey*Sey/(1+See*Syy)
        if RER > .5:
            RER = .5
        if (not self.adapted and self.sum_adapt > M and self.leak_estimate*Syy > .03*Syy):
            self.adapted = 1

        if self.adapted:
            for i in range(self.frame_size+1):
                r = self.leak_estimate*self.Yf[i]
                e = self.Rf[i]+1
                if r > .5*e:
                    r = .5*e
                r = 0.7*r + 0.3*(RER*e)
                self.power_1[i] = (r/(e*self.power[i]+10))
        else:
            adapt_rate = 0
            if Sxx > N * 1000:
                tmp32 = 0.25 * Sxx
                if tmp32 > .25*See:
                    tmp32 = .25*See
                adapt_rate = tmp32 / See
            self.power_1 = adapt_rate/(self.power+10)
            self.sum_adapt = self.sum_adapt+adapt_rate

        self.last_y[:self.frame_size] = self.last_y[self.frame_size:N]
        if self.adapted:
            self.last_y[self.frame_size:N] = mic-out
        return out

    def filter_dc_notch16(self, mic, mem):
        out = np.zeros_like(mic)
        den2 = self.notch_radius**2 + 0.7 * \
            (1-self.notch_radius)*(1 - self.notch_radius)
        for i in range(self.frame_size):
            vin = mic[i]
            vout = mem[0] + vin
            mem[0] = mem[1] + 2*(-vin + self.notch_radius*vout)
            mem[1] = vin - (den2*vout)
            out[i] = self.notch_radius * vout
        return out, mem

    def mdf_adjust_prop(self,):
        N = self.window_size
        M = self.M
        C = self.C
        K = self.K
        prop = np.zeros((M, 1),)
        for i in range(M):
            tmp = 1
            for chan in range(C):
                for speak in range(K):
                    tmp = tmp + np.sum(np.abs(self.W[:N//2+1, speak, i, chan])**2)
            prop[i] = np.sqrt(tmp)
        max_sum = np.maximum(prop, 1)
        prop = prop + .1 * max_sum
        prop_sum = 1 + np.sum(prop)
        prop = 0.99*prop/prop_sum
        return prop


if __name__ == "__main__":
    import soundfile as sf
    import librosa
    mic, sr = sf.read("samples/mic1-16k.wav")   #mic
    ref, sr = sf.read("samples/ref-16k.wav")   #far end
    min_len = min(len(mic), len(ref))      #取最小長度
    mic = mic[:min_len]
    ref = ref[:min_len]
    # 64 2048 for 8kHz.
    processor = MDF(sr, 128, 4096)
    start_time=time.time()
    e, y = processor.main_loop(ref, mic)
    mdf_time=time.time()
    print(mdf_time-start_time)
    sf.write('e.wav', e, sr)
    sf.write('y.wav', y, sr)
