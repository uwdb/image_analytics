import dipy.core.gradients as dpg
import nibabel as nib

DATA_LOC = "/home/dongfang/download/mri_data/100307/"

img = nib.load(DATA_LOC + './data.nii.gz')
data = img.get_data()

print "Done with reading, start converting..."

nib.save(nib.Nifti1Image(data[:,:,:,0:36], img.affine), DATA_LOC + 'small_data.nii.gz')
