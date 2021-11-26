# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:05:06 2020

@author: Wilm Decre
"""

#from scipy import signal
#from scipy.fft import fft
#from scipy import interpolate
import wavio
#import matplotlib.pyplot as plt
import detection_lib as dl
import os
#from os import listdir
#from os.path import isfile, join
from pathlib import Path
import xlsxwriter
import string
import csv
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


## Config

recordingsDir = 'C:/Users/wardv/Documents/Ward/KUL/MIIµ/Thesis/opnames/' 
filteredDir = '../RecordingsFiltered/' 
filteredNormalizedDir = '../RecordingsFilteredNormalized/' 
resultsDir = '../Results/'
resultsFile = 'Results_rows.xlsx'
SampledRecordingsDir = 'C:/Users/wardv/Documents/Ward/KUL/MIIµ/Thesis/Sampled/Sampled_recordings.xlsx'
resultsPath = resultsDir+resultsFile
drawplots = False
writefilteredrecordings = True

## create result file
workbook = xlsxwriter.Workbook(resultsPath)
workbook_sampled = xlsxwriter.Workbook(SampledRecordingsDir)

sheet1 = workbook_sampled.add_worksheet('TwoFlo')
sheet2 = workbook_sampled.add_worksheet('TriFlo')


# create result sheet and write headers
# worksheet = workbook.add_worksheet('Results')
# col=0
# worksheet.write(0,col,'name of recording')
# col+=1
# worksheet.write(0,col,'opening clicks [s]')
# col+=1
# worksheet.write(0,col,'closing clicks [s]')
# col+=1
# worksheet.write(0,col,'time between opening and closing clicks [s]')
# col+=1
# worksheet.write(0,col,'instantaneous heart rate based on time between closing clicks [bpm]')
# col+=1
# worksheet.write(0,col,'opening clicks sound level')
# col+=1
# worksheet.write(0,col,'closing clicks sound level')
# col+=1
# worksheet.write(0,col,'background sound level')
# col+=1
# worksheet.write(0,col,'opening clicks sound level [dB]')
# col+=1
# worksheet.write(0,col,'closing clicks sound level [dB]')
# col+=1
# worksheet.write(0,col,'background sound level [dB]')
# # set column to zero and row to 1 for writing results
# col = 0
# row_start = 1

## magic numbers:

#rms_npoints = 960
rms_npoints = 360
#rms_npoints = 5000
f_low = 5000 #[Hz]
f_high = 16000 #[Hz]
performfreqanalysis = False
background_percentile = 10
peaks_percentile = 70
background_factor = 0.5
click_relativetreshold = 2.0
relative_closing_level = 0.5

# additional functions

num2alpha = dict(zip(range(0,26),string.ascii_uppercase))

## read file

recordingsFiles = [os.path.relpath(os.path.join(root,name),recordingsDir)
             for root, dirs, files in os.walk(recordingsDir)
             for name in files
             if name.endswith((".wav"))]

#recordingsFiles = [f for f in listdir(recordingsDir) if isfile(join(recordingsDir, f))]
#recordingsFiles = ['Triflo1 on 2020-03-03 @ 14u54m46s.wav']
#recordingsFiles = ['Triflo2_3 on 2020-05-19 @ 12u08m19s.wav']
#recordingsFiles = ['0880_explant on 2020-08-19 @ 09u45m47s.wav']
#recordingsFiles = ['Triflo3_3 on 2020-05-19 @ 10u02m57s.wav']

col_1 = 0
col_2 = 0



kk = 0
for recordingsFile in recordingsFiles:
    
    print('file number: '+str(kk))
    kk+=1
    print(recordingsFile)
    
    recordingsPath = recordingsDir+recordingsFile
    
    if Path(recordingsPath).suffix != '.wav':
        raise Exception('selected file is not .wav')
    
    ## read file
    sound,fs = dl.read_wav(recordingsPath)
        
    ## magic number:
    click_mindistance = 0.2*fs 
    
    if drawplots:
        plt.clf()
    
    ## band pass filter
    
    sos = signal.butter(4, [f_low,f_high], btype='bandpass', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sound)
        
    if writefilteredrecordings:
              
        dl.create_dirs(filteredDir+recordingsFile)
        dl.create_dirs(filteredNormalizedDir+recordingsFile)
                
        filtered_export = filtered - np.mean(filtered)
        filtered_export_normalized = filtered_export/np.max(filtered_export)
        #wavio.write(filteredDir+recordingsFile,filtered_export.astype(int),fs,sampwidth=3,scale='none')
        #wavio.write(filteredNormalizedDir+recordingsFile,filtered_export_normalized.astype(int),fs,sampwidth=3,scale='none')
            
    ## rms
    
    ## normalize signal
    filtered = filtered/np.max(abs(filtered))
    
    moving_rms = dl.window_rms(filtered,rms_npoints)
    moving_rms = moving_rms/np.max(np.abs(moving_rms))
    
    #background_level = (1-background_factor)*np.min(moving_rms)+background_factor*np.max(moving_rms)
    background_level = (1+background_factor)*np.percentile(moving_rms,background_percentile)
       
    
    ## peak detection
    
    # to do, write height as a function of background level
    peaks, properties = signal.find_peaks(moving_rms,distance=click_mindistance,height=click_relativetreshold*np.min(moving_rms))
    
    # remove first and last three peaks to remove boundary effects
    peaks = peaks[3:-3]
    
    ## opening versus closing
    
    # this can be improved, now not very robust...
    
    if len(peaks)>0:
        peak_level_percentile = np.percentile(moving_rms[peaks],peaks_percentile)
    else:
        peak_level_percentile = np.max(moving_rms)
        
    
    closing_level_threshold = background_level+relative_closing_level*(peak_level_percentile-background_level)
    
    peaks_opening =  [x for x in peaks if moving_rms[x]<closing_level_threshold]
    peaks_closing =  [x for x in peaks if moving_rms[x]>=closing_level_threshold]
    
    
    ## we do not necessarily want to start with an opening peak, as opening peaks are not always detected
    ## remove first closing peak if needed, in order to have opening-closing sequences
    #if peaks_closing[0]<peaks_opening[0]:
    #  peaks_closing = peaks_closing[1:]
    
    # remove last opening peak if needed, in order to end with closing
    if (len(peaks_opening)>0):
        if peaks_opening[-1]>peaks_closing[-1]:
            peaks_opening = peaks_opening[:-1]
        
    leftcrossings_opening = dl.leftcrossings(moving_rms,peaks_opening,background_level)
    leftcrossings_closing = dl.leftcrossings(moving_rms,peaks_closing,background_level)
    
    rightcrossings_opening = dl.rightcrossings(moving_rms,peaks_opening,background_level)
    rightcrossings_closing = dl.rightcrossings(moving_rms,peaks_closing,background_level)
    
    midcrossings_opening = np.mean([leftcrossings_opening,rightcrossings_opening], axis=0).astype(int)
    midcrossings_closing = np.mean([leftcrossings_closing,rightcrossings_closing], axis=0).astype(int)
    
    j=0 
    while j<len(peaks_closing):
        row = 0
        index = peaks_closing[j]
        sample = filtered_export_normalized[index-1250:index+1250]
        i = 0
        if kk < 62:
            while i < len(sample):
                sheet1.write(row, col_1, sample[i])
                i += 1
                row += 1 
            col_1 +=1
        else:
            while i < len(sample):
                sheet2.write(row, col_2, sample[i])
                i += 1
                row += 1 
            col_2 += 1
        j += 1

    
    t = 1/fs*np.arange(len(filtered))
    
    if drawplots:
        plt.figure(1)
        plt.plot(t,filtered)
        plt.plot(t,moving_rms,linewidth=5)
        plt.plot(t[peaks_opening],moving_rms[peaks_opening],"x",markersize=15,linewidth=12)
        plt.plot(t[peaks_closing],moving_rms[peaks_closing],"x",markersize=15,linewidth=12)
        plt.plot(t[leftcrossings_opening],moving_rms[leftcrossings_opening],"o",markersize=15,linewidth=12)
        plt.plot(t[leftcrossings_closing],moving_rms[leftcrossings_closing],"o",markersize=15,linewidth=12)
        plt.plot(t[rightcrossings_opening],moving_rms[rightcrossings_opening],"o",markersize=15,linewidth=12)
        plt.plot(t[rightcrossings_closing],moving_rms[rightcrossings_closing],"o",markersize=15,linewidth=12)
        plt.plot(t[midcrossings_opening],moving_rms[midcrossings_opening],"x",markersize=15,linewidth=12)
        plt.plot(t[midcrossings_closing],moving_rms[midcrossings_closing],"x",markersize=15,linewidth=12)
        plt.plot(t,np.repeat(background_level,len(t)),linewidth=5)
        plt.xlabel('time [s]')
        plt.ylabel('normalized amplitude [-]')
        plt.grid(b=True)
    
    #plt.figure(2)
    #plt.plot(sound)
        
    ## match opening clicks
        
    # assumption: here we assume that closing clicks are always found, i.e. that there are no opening clicks for which there is no corresponding closing click
    # extra behaviour: if there is more than one opening click between two closing clicks, the last detected one will be used

    # UGLY, used -999 as value if no matching opening peak is found - to be changes to a better value
    # Note: to rework, this is an inefficient way, as we will later iterate over peaks_opening matched => write more efficient
    
    
    invalid_match = -999
    peaks_opening_matched = -999*np.ones(len(peaks_closing),dtype=int)
    
    jj=0
    for ii in np.arange(len(peaks_opening)):
        while ((jj<len(peaks_closing) and (peaks_closing[jj]-peaks_opening[ii]) < 0.0)):
            jj=jj+1
        if (jj<len(peaks_closing)):
            peaks_opening_matched[jj]=peaks_opening[ii]
                  
    ## Frequency analysis
    
    if (performfreqanalysis==True):
        
        Nf_summed = 2000
        freq_summed = np.linspace(0.0, 1.0/2.0*fs, Nf_summed//2)
        spectrum_summed_closing = np.zeros([Nf_summed//2],dtype=np.complex128)
        spectrum_summed_opening = np.zeros([Nf_summed//2],dtype=np.complex128)
        
        N_closing = len(leftcrossings_closing)
        N_opening = len(leftcrossings_opening)
        
        
        for ii in np.arange(N_closing):
          
          t_click = t[leftcrossings_closing[ii]:rightcrossings_closing[ii]]
          click = filtered[leftcrossings_closing[ii]:rightcrossings_closing[ii]]
          N = len(click)
        
          freq,spectrum_windowed,spectrum_interpolated = dl.spectrum(t_click,click,fs,freq_summed)
          spectrum_windowed = np.abs(spectrum_windowed)
          spectrum_summed_closing = spectrum_summed_closing + spectrum_interpolated
        
          if drawplots:
              plt.figure(3)
              plt.semilogy(freq,spectrum_windowed)
              plt.xlabel('frequency [Hz]')
              plt.ylabel('amplitude [-]')
            
        for ii in np.arange(N_opening):
          
          t_click = t[leftcrossings_opening[ii]:rightcrossings_opening[ii]]
          click = filtered[leftcrossings_opening[ii]:rightcrossings_opening[ii]]
          N = len(click)
        
          freq,spectrum_windowed,spectrum_interpolated = dl.spectrum(t_click,click,fs,freq_summed)
          spectrum_windowed = np.abs(spectrum_windowed)
          spectrum_summed_opening = spectrum_summed_opening + spectrum_interpolated
        
          if drawplots:
              plt.figure(4)
              plt.semilogy(freq,spectrum_windowed)
              plt.xlabel('frequency [Hz]')
              plt.ylabel('amplitude [-]')
            
        spectrum_averaged_closing = np.abs(spectrum_summed_closing)/N_closing
        spectrum_averaged_opening = np.abs(spectrum_summed_opening)/N_opening
        
        if drawplots:
            plt.figure(5)
            plt.semilogy(freq_summed,spectrum_averaged_closing)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('amplitude [-]')
            plt.title('closing clicks')
            
            plt.figure(6)
            plt.semilogy(freq_summed,spectrum_averaged_opening)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('amplitude [-]')
            plt.title('opening clicks')

    ## Writing results to file
    
    # if separate worksheet per file:
    #worksheet = workbook.add_worksheet(recordingsFile.replace(" ", "").replace("_","").replace("-","")[1:31])
    
#     col = 0
#     #worksheet.write(0,col,'name of recording')
#     worksheet.write(row_start,col,recordingsFile)
    
#     col+=1
#     #worksheet.write(0,col,'opening clicks [s]')
#     for row_number in np.arange(0,len(peaks_opening_matched)):
#         if (peaks_opening_matched[row_number] != invalid_match):
#             worksheet.write(row_number+row_start,col,t[peaks_opening_matched[row_number]])
    
#     col+=1
#     #worksheet.write(0,col,'closing clicks [s]')
#     for row_number in np.arange(0,len(peaks_closing)):
#         worksheet.write(row_start+row_number,col,t[peaks_closing[row_number]])
    
#     col+=1
#     #worksheet.write(0,col,'time between opening and closing clicks [s]')
#     for row_number in np.arange(0,len(peaks_closing)):
#         if (peaks_opening_matched[row_number] != invalid_match):
#             worksheet.write_formula(row_start+row_number,col,('='+xlsxwriter.utility.xl_col_to_name(col-1)+'%d-'+xlsxwriter.utility.xl_col_to_name(col-2)+'%d')% (row_start+row_number+1,row_start+row_number+1))
    
#     col+=1
#     #worksheet.write(0,col,'instantaneous heart rate based on time between closing clicks [bpm]')
#     for row_number in np.arange(1,len(peaks_closing)):
#         worksheet.write_formula(row_start+row_number,col,('=1/('+xlsxwriter.utility.xl_col_to_name(col-2)+'%d-'+xlsxwriter.utility.xl_col_to_name(col-2)+'%d)*60')% (row_start+row_number+1,row_start+row_number))
    
#     col+=1
#     #worksheet.write(0,col,'opening clicks sound level')
#     for row_number in np.arange(0,len(peaks_opening_matched)):
#         if (peaks_opening_matched[row_number] != invalid_match):
#             worksheet.write(row_start+row_number,col,moving_rms[peaks_opening_matched[row_number]])
    
#     col+=1
#     #worksheet.write(0,col,'closing clicks sound level')
#     for row_number in np.arange(0,len(peaks_closing)):
#         worksheet.write(row_start+row_number,col,moving_rms[peaks_closing[row_number]])
      
#     col+=1
#     #worksheet.write(0,col,'background sound level')
#     worksheet.write(row_start,col,background_level)
    
#     col+=1
#     #worksheet.write(0,col,'opening clicks sound level') -dB
#     for row_number in np.arange(0,len(peaks_opening_matched)):
#         if (peaks_opening_matched[row_number] != invalid_match):
#             worksheet.write_formula(row_start+row_number,col,('=20*LOG10('+xlsxwriter.utility.xl_col_to_name(col-3)+'%d)')% (row_start+row_number+1))
    
#     col+=1
#     #worksheet.write(0,col,'closing clicks sound level') -dB
#     for row_number in np.arange(0,len(peaks_closing)):
#             worksheet.write_formula(row_start+row_number,col,('=20*LOG10('+xlsxwriter.utility.xl_col_to_name(col-3)+'%d)')% (row_start+row_number+1))
#     col+=1
#     #worksheet.write(0,col,'background sound level') -dB
#     worksheet.write_formula(row_start,col,('=20*LOG10('+xlsxwriter.utility.xl_col_to_name(col-3)+'%d)')% (row_start+1))

#     row_start += len(peaks_closing)
# ## write marker file for Adobe Audition
    
#     dl.create_dirs(resultsDir+recordingsFile)
#     with open(resultsDir+recordingsFile[:-3]+'csv',mode='w') as markerFile:
#         fieldnames = ['Name','Start','Duration','Time Format','Type','Description']
#         markerWriter = csv.DictWriter(markerFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL,fieldnames=fieldnames)
#         markerWriter.writeheader()
        
#         for ii in range(len(peaks_closing)):       
#             markerWriter.writerow({'Name': 'Closing '+str(ii+1),'Start':peaks_closing[ii],'Duration': '0','Time Format': str(fs)+' Hz','Type': 'Cue', 'Description': ''})

#         for ii in range(len(peaks_opening_matched)):
#             if (peaks_opening_matched[ii] != invalid_match):
#                 markerWriter.writerow({'Name': 'Opening '+str(ii+1),'Start':peaks_opening_matched[ii],'Duration': '0','Time Format': str(fs)+' Hz','Type': 'Cue', 'Description': ''})
     
    ## histograms
    
    #nbins = 5
    #times1 = np.diff(t[midcrossings_closing])
    #times2 = np.diff(t[peaks_closing])
    #
    #times3 = np.diff(t[midcrossings_opening])
    #times4 = np.diff(t[peaks_opening])
    #
    #times5 = t[midcrossings_closing] - t[midcrossings_opening]
    #times6 = t[peaks_closing] - t[peaks_opening]
    #
    #
    #plt.figure(2)
    #plt.plot(times1,'x-')
    #plt.plot(times2,'o-')
    #plt.xlabel('number of closing click')
    #plt.ylabel('time between consecutive closing clicks [s]')
    #
    #plt.figure(3)
    #plt.plot(times3,'x-')
    #plt.plot(times4,'o-')
    #plt.xlabel('number of opening click')
    #plt.ylabel('time between consecutive opening clicks [s]')
    #
    #plt.figure(4)
    #plt.plot(times5,'x-')
    #plt.plot(times6,'o-')
    #plt.xlabel('number of beat')
    #plt.ylabel('time between opening and closing click of beat [s]')
    #
    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    #axs[0].hist(times1, bins=nbins)
    #axs[1].hist(times2, bins=nbins)
    #
    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    #axs[0].hist(times3, bins=nbins)
    #axs[1].hist(times4, bins=nbins)
    #
    #print(peaks_opening)
    #print(peaks_closing)
    #
    #
    
    plt.show()

workbook.close()
workbook_sampled.close()

