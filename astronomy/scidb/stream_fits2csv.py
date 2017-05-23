data_path = '/home/ubuntu/lsst_data/raw/crblasted'
visits = [
"0289697",
#"0289783",
#"0288935",
#"0288976",
#"0289016",
#"0289056",
#"0289161",
#"0289202",
#"0289243",
#"0289284",
#"0289368",
#"0289409",
#"0289450",
#"0289493",
#"0289614",
#"0289818",
#"0289820",
#"0289850",
#"0289851",
#"0289871",
#"0289892",
#"0289893",
#"0289913",
#"0289931",
]

from astropy.io import fits
import numpy as np
import math
import os 
import time
#cnt = 0
#for i in range(2000):
#    for j in range(2000):
#        if not math.isnan(a[i,j]) and not math.isinf(a[i,j]):
#            cnt += 1
#            print("b[%d,%d] = %.3f"%(i,j,a[i,j]))
#            if (cnt > 10): 
#                exit(0)

#dump to csv file
print time.ctime()
csv_file = open('/home/ubuntu/lsst_data/raw/lsst_1_visit.csv', 'w+')
no_visit = 0
for visit in visits:
    no_visit += 1
    rootdir = data_path + visit
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            ccd = int(float(f[15:17]))
            print "processing "+subdir+"/"+f, "ccd =", ccd
            hdulist = fits.open(rootdir + '/instcal' + visit + '.' + str(ccd) + '.fits')
            a = np.array(hdulist[1].data)
            for x in range(len(a)):
                for y in range(len(a[0])):
                    csv_file.write('%d,%d,%d,%d,%.3f\n' %
                            (no_visit, ccd, x, y, a[x,y]))
csv_file.close()
print time.ctime()

print "Done!"
