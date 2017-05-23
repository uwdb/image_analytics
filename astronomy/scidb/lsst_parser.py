data_path = '/home/ubuntu/input_coadds/'
visits = [
"0288935",
"0288976",
"0289016",
"0289056",
"0289161",
"0289202",
"0289243",
"0289284",
"0289368",
"0289409",
"0289450",
"0289493",
"0289614",
"0289697",
"0289783",
"0289818",
"0289820",
"0289850",
"0289851",
"0289871",
"0289892",
"0289893",
"0289913",
"0289931",
]

from astropy.io import fits
import numpy as np
import math
import os 

#cnt = 0
#for i in range(2000):
#    for j in range(2000):
#        if not math.isnan(a[i,j]) and not math.isinf(a[i,j]):
#            cnt += 1
#            print("b[%d,%d] = %.3f"%(i,j,a[i,j]))
#            if (cnt > 10): 
#                exit(0)

#dump to csv file
csv_file = open('/home/ubuntu/lsst_1-24.csv', 'w')
no_visit = 0
for visit in visits:
    no_visit += 1
    rootdir = data_path + visit
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            #for full size (inclusive, before 2000x): x[13:24], y[10:21], t[1:24]
            print "processing "+subdir+"/"+f
            idx_x = int(f[15:17])
            idx_y = int(f[18:20])
            hdulist = fits.open(rootdir + '/instcal' + visit + '.' +
                        str(idx_x) + '_' + str(idx_y) + '.fits')
            a = np.array(hdulist[1].data[100:2100, 100:2100])
            for local_x in range(2000):
                for local_y in range(2000):
                    global_x = local_x + 1 + (idx_x-13) * 2000
                    global_y = local_y + 1 + (idx_y-10) * 2000
                    flux = a[local_x, local_y]
                    csv_file.write('%d,%d,%d,%.3f\n' %
                            (global_x, global_y, no_visit, flux))
csv_file.close()

print "Done!"
