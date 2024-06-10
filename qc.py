import matplotlib 
matplotlib.rcParams['figure.facecolor'] = '1.'
matplotlib.use('Agg')
import ants
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.filters import threshold_otsu
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

def load_3d(fn, t=0, verbose=False):
    if verbose : print('Reading Frame %d'%t,'from', fn)
    img = nib.load(fn)
    vol = img.get_fdata() 
    if len(vol.shape) == 4 :
        vol = vol[:,:,:,t]
    vol = vol.reshape(vol.shape[0:3] )
    img = nib.Nifti1Image(vol, img.affine)
    return img, vol

def get_spacing(aff, i) : 
    return aff[i, np.argmax(np.abs(aff[i,0:3]))] 


def get_slices(vol,  dim, i) :

    if dim == 0:
        r = vol[i, :, : ]
    elif dim == 1 :
        r = vol[ :, i, : ]
    else :
        r = vol[ :, :, i ]
    return r

class ImageParam():
    def __init__(self, in_fn, out_fn, overlay_fn=None, alpha=[1.], dpi=100, duration=100, cmap1=plt.cm.nipy_spectral, cmap2=plt.cm.gray, colorbar=False, edge_1=-1, edge_2=-1,nframes=15, time_frames=1, ndim=3):
        self.in_fn = in_fn
        self.out_fn = out_fn
        self.alpha = alpha
        self.dpi = dpi
        self.overlay_fn = overlay_fn
        self.duration = duration
        self.cmap1 = cmap1
        self.cmap2 = cmap2
        self.colorbar=colorbar
        self.edge_1 = edge_1
        self.edge_2 = edge_2
        self.nframes = nframes
        self.ndim = ndim
        self.time_frames=time_frames

    def load_isotropic(self,in_fn,t=0):
        aff = nib.load(in_fn).affine
        vol_img, vol = load_3d(in_fn,t)
        sep =[ get_spacing(vol_img.affine, i) for i in range(3) ]
        min_unit=np.min(np.abs(sep))
        #new_units=[min_unit*np.sign(sep[0]), min_unit*np.sign(sep[1]), min_unit*np.sign(sep[2]) ] 
        vol_img = resample_to_output(vol_img, [min_unit]*3,order=1 )
        vol=vol_img.get_fdata()
        return vol_img, vol

    def volume2gif(self):
        in_fn = self.in_fn
        out_fn = self.out_fn
        overlay_fn = self.overlay_fn
        alpha  = self.alpha
        dpi = self.dpi
        duration  = self.duration
        cmap1 = self.cmap1
        cmap2 = self.cmap2

        def apply_tfm(img, sigma):
            if sigma >= 0  : 
                img = gaussian_filter(img, sigma)
                img = np.sqrt(np.sum(np.abs(np.gradient(img)),axis=0)) 
                img[ img < threshold_otsu(img) ] =0 
            return img
        img = nib.load(in_fn)
        ndim=len(img.shape)
        full_vol = img.get_fdata()
        vmin, vmax  = (np.min(full_vol)*.02, np.max(full_vol)*0.98 )

        tmax=1
        if ndim == 4 :
            tmax = nib.load(in_fn).shape[3]
        for t in range(tmax) :
            vol_img, vol = self.load_isotropic(in_fn,t)
            vol = apply_tfm(vol,self.edge_1)

            if overlay_fn != None :
                overlay_img, overlay_vol = self.load_isotropic(overlay_fn)
                overlay_vol = resample_from_to(overlay_img, vol_img).get_fdata()
                overlay_vol = apply_tfm(overlay_vol,self.edge_2)
                omin, omax  = (np.min(overlay_vol), np.max(overlay_vol) )#np.percentile(vol, [1,99])

            frames=[]
            plt.clf()
            fig = plt.figure()

            axes=[fig.add_subplot(1, 3, ii) for ii in [1,2,3]]
            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")

            frame=[ axes[ii].imshow(get_slices(vol,ii,0), cmap=cmap1, animated=True,origin='lower', vmin=vmin, vmax=vmax, interpolation='gaussian' ) for ii in [0,1,2]]
            nframes_per_alpha= self.nframes
            total_frames = nframes_per_alpha * len(alpha) 
            def animate(i):
                alpha_level = int(i / nframes_per_alpha)
                ii = i % nframes_per_alpha
                for dim in [0,1,2] :
                    idx = np.round(vol.shape[dim] * ii / (self.nframes+0.0)).astype(int)
                    r = get_slices(vol, dim, idx)

                    frame[dim] = axes[dim].imshow(r.T, cmap=cmap1, animated=True,origin='lower', vmin=vmin, vmax=vmax, interpolation='gaussian' )

                    if overlay_fn != None :
                        m = get_slices(overlay_vol, dim, idx)
                        frame[dim] = axes[dim].imshow(m.T,alpha=alpha[alpha_level], cmap=cmap2, vmin=omin, vmax=omax, interpolation='gaussian', origin='lower', animated=True)
                return frame

            if self.colorbar :
                fig.colorbar(frame[2], shrink=0.35 )
            plt.tight_layout()
            stime=time.time()
            ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=duration, blit=True, repeat_delay=1000)
            if ndim == 4 :
                out_fn = self.out_fn[t]
            ani.save(out_fn, dpi=self.dpi) #, writer='imagemagick')
            plt.close(fig)
            #print(time.time()-stime)  
            print('Writing', out_fn)

'''
        visual_qc_images=[  
                ImageParam(self.inputs.pet_3d , self.inputs.pet_3d_gif, self.inputs.pet_brain_mask, cmap1=plt.cm.Greys, cmap2=plt.cm.Reds, alpha=[0.3], duration=300),
                ImageParam(self.inputs.pet_space_mri , self.inputs.pet_coreg_gif, self.inputs.mri_space_nat, alpha=[0.55,0.70,0.85], duration=400,  nframes=15 ),
                ImageParam(self.inputs.pet_space_mri , self.inputs.pet_coreg_edge_2_gif, self.inputs.mri_space_nat, alpha=[0.4], duration=300, edge_2=1, cmap1=plt.cm.Greys, cmap2=plt.cm.Reds ),
                # Results Labels
                ImageParam(self.inputs.t1_analysis_space, self.inputs.results_labels_gif, self.inputs.results_labels, alpha=[0.4], duration=300, cmap1=plt.cm.Greys, cmap2=plt.cm.nipy_spectral ),
                ImageParam(self.inputs.mri_space_nat, self.inputs.template_alignment_gif, self.inputs.template_space_mri, alpha=[0.4], duration=300, cmap1=plt.cm.Greys, cmap2=plt.cm.Reds )
                ]
'''
