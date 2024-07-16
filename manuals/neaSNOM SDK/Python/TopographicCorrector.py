# -*- coding: utf-8 -*-

import os, clr

path_to_dll = r'C:\Users\Patrick\Documents\Nea\ClientApplication\Windows\SDK\bin\Debug'

clr.AddReference(os.path.join(path_to_dll, "Nea.Client.Processing.dll"))

import Nea.Client.Processing as neap
import Nea.Client.SharedDefinitions as neas

averages = 1  # runs 1 time, so no averaging at all
resolution = neas.IntSize3D(5, 4, 1)  # width, height, depth

# data from the channels Z and R-Z would normally come from
# the microscope or database as ObservablePixels, but here
# we create it manually...

import random
pixels = [random.randrange(-20, 120) for _ in range(resolution.PixelsCount)]

z = neas.ObservablePixels(resolution, averages, pixels, len(pixels), 0)
rz = neas.ObservablePixels(resolution, averages, pixels, len(pixels), 0)

print('    INPUT    \n-------------')
print(z.Items)
print('\n')

import System

# can be used for a cancel button
cancellation = getattr(System.Threading.CancellationToken, 'None')  # because None is a keyword

# can be used for a progress bar
progress = System.Progress[float]() #  reports percenage 0..1

neap.TopographyCorrector.TryCorrect(z, rz, cancellation, progress)

print('    OUTPUT    \n-------------')
print(z.Items)
print('\n')
