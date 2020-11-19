# -*- coding: utf-8 -*-

#### GENERAL PARAMETER ######################################################################
setpoint = 0.8
#             [x0,    y0, width,     height, x_res,     y_res,     rotation,    t_int]
param_scan = [50, 50, 5, 5, 200, 200, 0, 3.3]
# possible channels: 'Z','M0-5A','M1-5P','O0-5A','O1-5A'
channels_list = ['Z', 'M1A', 'M1P', 'O0A', 'O2A', 'O2P']
path = 'data/'

#### IMPORT OF MODULES ######################################################################
import sys
import time
import clr
import re
import numpy as np
import os
import matplotlib.pyplot as plt


#### DEFINITION OF FUNCTIONS ######################################################################
def connectMic():
    # connects to the microscope of neaSNOM
    # returns an instance of a microscope neaMIC and the connection to the SDK neaClient
    assembly_folder = '//nea-server/updates/SDK'  # Import all DLLs in the folder or 'N:/updates/SDK' to mapped drive

    # Basic steps to connect to Microscope
    sys.path.append(assembly_folder)  # Import all DLLs in the folder
    clr.AddReference('Nea.Client.Hardware')  # Load the main DLL
    import Nea.Client.Hardware.SDK as neaSDK  # Import the DLL as element neaSDK
    neaClient = neaSDK.Connection(
        'nea-server')  # Open up connection to microscope called neaClient
    neaMic = neaClient.Connect()  # Define the Microscope neaMIC
    time.sleep(0.1)  # Short delay makes things working fine

    # Basic infos of Client and Server and print them to the console of python
    ClVersion = neaMic.ClientVersion  # get Client Version
    print("Client Version: " + ClVersion)
    SeVersion = neaMic.ServerVersion  # get Server Version
    print("Server Version: " + SeVersion)

    return neaMic, neaClient


def ConvToNumArray(D2):
    # For 2D AFM data conversion
    # takes 2D System Single D2
    # returns 1D python numpy array data
    limit1 = D2.GetUpperBound(0) + 1
    limit2 = D2.GetUpperBound(1) + 1
    data = np.zeros([limit1, limit2])
    for a in range(limit1):
        for b in range(limit2):
            data[a, b] = D2[a, b]
    return data


def SetAFMparam(scan, x0, y0, dx, dy, resx, resy, angle, t_s):
    # Sets AFM-scan parameter
    # scan AFM-Scan object
    # x0, y0 Centerposition of scan [mum]
    # dx, dy width and height of measurement [mum]
    # resx, resy Amount of pixels
    # angle rotation of scan area
    # t_s sampling time [ms]
    scan.set_CenterX(x0)
    scan.set_CenterY(y0)
    scan.set_ScanAreaWidth(dx)
    scan.set_ScanAreaHeight(dy)
    scan.set_ResolutionColumns(resx)
    scan.set_ResolutionRows(resy)
    scan.set_ScanAngle(angle)
    scan.set_SamplingTime(t_s)


def plot_scan(scan, channels, channels_list):
    # plots live the current measurement
    # scan             scan object, e.g. neaMic.PrepareAfmScan.Start()
    # channels        list of channels to plot
    # returns        NOTHING
    if len(channels_list) > 5:
        numb_graphs_row = 5
        if len(channels_list) % 5 != 0:
            rows = len(channels_list) // 5 + 1
        else:
            rows = len(channels_list) // 5
    else:
        numb_graphs_row = len(channels_list)
        rows = 1
    fig = plt.figure(figsize=(numb_graphs_row * 3, 3 * rows))
    fig_axes = {}
    i = 0
    for channel in channels_list:
        fig_axes[channels[channel]] = plt.subplot(rows, numb_graphs_row, i + 1)
        fig_axes[channels[channel]].title.set_text(channel)
        i += 1
    while not scan.IsCompleted:
        for channel in channels_list:
            data = ConvToNumArray(channels[channel].GetData())
            fig_axes[channels[channel]].imshow(data)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)


def save_data(channels, folder):
    # save all selected channels to an Excel file
    # channels   dictionary of channels name:SDK Image space
    # folder     location where to save data
    if not os.path.exists(folder):
        os.makedirs(folder)
    for channel in channels.keys():
        file_name = folder + channel
        ExpGwy.gsf_write(ConvToNumArray(channels[channel].GetData()), file_name)
        np.savetxt(file_name + '.txt',
                   ConvToNumArray(channels[channel].GetData()), delimiter=',  ')


class ExpGwy:
    # class to import and export numpy dataset to gsf file for Gwyddion
    @staticmethod
    def gsf_write(data, file_name, metadata={}):
        '''Write a 2D array to a Gwyddion Simple Field 1.0 file format
        http://gwyddion.net/documentation/user-guide-en/gsf.html

        Args:
            file_name (string): the name of the output (any extension will be replaced)
            data (2darray): an arbitrary sized 2D array of arbitrary numeric type
            metadata (dict): additional metadata to be included in the file
        Returns:
            nothing
        '''

        XRes = data.shape[1]
        YRes = data.shape[0]

        data = data.astype('float32')

        if file_name.rpartition('.')[1] == '.':
            file_name = file_name[0:file_name.rfind('.')]

        gsfFile = open(file_name + '.gsf', 'wb')

        s = ''
        s += 'Gwyddion Simple Field 1.0' + '\n'
        s += 'XRes = {0:d}'.format(XRes) + '\n'
        s += 'YRes = {0:d}'.format(YRes) + '\n'

        for i in metadata.keys():
            try:
                s += i + ' = ' + '{0:G}'.format(metadata[i]) + '\n'
            except BaseException:
                s += i + ' = ' + str(metadata[i]) + '\n'

        # print s

        gsfFile.write(bytes(s, 'utf-8'))

        gsfFile.write(b'\x00' * (4 - len(s) % 4))

        gsfFile.write(data.tobytes(None))

        gsfFile.close()

    @staticmethod
    def gsf_read(file_name):
        '''Read a Gwyddion Simple Field 1.0 file format
        http://gwyddion.net/documentation/user-guide-en/gsf.html

        Args:
            file_name (string): the name of the output (any extension will be replaced)
        Returns:
            metadata (dict): additional metadata to be included in the file
            data (2darray): an arbitrary sized 2D array of arbitrary numeric type
        '''
        if file_name.rpartition('.')[2] != 'gsf':
            messagebox.showerror('Error', 'Needs a .gsf file')
            return 0, 0
        if file_name.rpartition('.')[1] == '.':
            file_name = file_name[0:file_name.rfind('.')]

        gsfFile = open(file_name + '.gsf', 'rb')

        metadata = {}

        # check if header is OK
        if not (gsfFile.readline().decode(
                'UTF-8') == 'Gwyddion Simple Field 1.0\n'):
            gsfFile.close()
            raise ValueError('File has wrong header')

        term = b'00'
        # read metadata header
        while term != b'\x00':
            line_string = gsfFile.readline().decode('UTF-8')
            if re.compile(' = ').search(line_string):
                metadata[line_string.rpartition(' = ')[0]] = \
                line_string.rpartition(' = ')[2]
            else:
                metadata[line_string.rpartition('=')[0]] = \
                line_string.rpartition('=')[2]
            term = gsfFile.read(1)
            gsfFile.seek(-1, 1)
        gsfFile.read(4 - gsfFile.tell() % 4)

        # fix known metadata types from .gsf file specs
        # first the mandatory ones...
        metadata['XRes'] = np.int(metadata['XRes'])
        metadata['YRes'] = np.int(metadata['YRes'])

        # now check for the optional ones
        if 'XReal' in metadata:
            metadata['XReal'] = np.float(metadata['XReal'])

        if 'YReal' in metadata:
            metadata['YReal'] = np.float(metadata['YReal'])

        if 'XOffset' in metadata:
            metadata['XOffset'] = np.float(metadata['XOffset'])

        if 'YOffset' in metadata:
            metadata['YOffset'] = np.float(metadata['YOffset'])

        if 'Neaspec_ZRes' in metadata:
            metadata['Neaspec_ZRes'] = np.float(metadata['Neaspec_ZRes'])

        if 'Neaspec_Runs' in metadata:
            metadata['Neaspec_Runs'] = np.float(metadata['Neaspec_Runs'])

        if 'Neaspec_MOffset' in metadata:
            metadata['Neaspec_MOffset'] = np.float(metadata['Neaspec_MOffset'])

        if 'Neaspec_MReal' in metadata:
            metadata['Neaspec_MReal'] = np.float(metadata['Neaspec_MReal'])

        if 'Neaspec_WavenumberScaling' in metadata:
            metadata['Neaspec_WavenumberScaling'] = np.float(
                metadata['Neaspec_WavenumberScaling'])

        data = np.frombuffer(gsfFile.read(), dtype='float32').reshape(
            metadata['YRes'], metadata['XRes'])

        gsfFile.close()

        return metadata, data


#### ACTUAL PROGRAM ######################################################################
neaMic, neaClient = connectMic()  # Connect to Microscope

# example to start auto approach after being connected
neaMic.CancelCurrentProcedure()  # Cancel any running procedure (not obligatory)
neaMic.RegulatorOff()  # Retract sample (not obligatory)
if not neaMic.IsInContact:  # Check if system in contact (not obligatory)
    neaMic.AutoApproach(setpoint)  # Start auto approach with chosen setpoint

time.sleep(5)  # delay to let the system settle before the measurement

# example to perform, plot and save AFM scan after being connected and being in contact
scan = neaMic.PrepareAfmScan()  # Prepare scan
SetAFMparam(scan, *param_scan)
image = scan.Start()  # Start scan
channels_dict = {}  # Creates a dictionarry of data channels name:SDK image
for channel in channels_list:
    channels_dict[channel] = image.GetChannel(channel)
plot_scan(scan, channels_dict, channels_list)  #
save_data(channels_dict, path)  # save all data as channel.txt and channel.gsf

# go out of contact and disconnect from the microscope
neaMic.RegulatorOff()  # Retract sample
neaClient.Disconnect()  # Disconnect from SDK
plt.show()  # Only to keep the matplolib image open at the end