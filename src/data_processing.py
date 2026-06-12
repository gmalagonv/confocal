import tifffile as tiff
import numpy as np
from aicsimageio.readers import LifReader
# import napari
# from aicsimageio import AICSImage
# import numpy as np
from tifffile import imwrite
import os
import matplotlib.pyplot as plt
import platform

from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import richardson_lucy

def loader2(date,user,split_frames=False, server=False):
    
    """_summary_
    
    """
    system = platform.system()
    if system == 'Linux':
        if server:
            base_dir = '/home/gerard/ITB/home/data/confocal'
        else:
            base_dir = '/home/gerard/data/confocal'
        
    elif system == 'Darwin':
        if server:
            base_dir = '/Users/gerard/ITB/home/data/confocal'
        else:
            base_dir = '/Users/gerard/data/confocal'
        
    elif system == "Windows":
        base_dir = 'C:/Users/cviko/data/confocal/'
        

    
    all_path = os.path.join(base_dir, f"{date}_{user}")
    
    print(all_path)

    project_path = os.path.join(all_path, 'Project.lif')
    reader = LifReader(project_path)
    print(f"Number of series: {len(reader.scenes)}")
    
    for i, series in enumerate(reader.scenes):
        series_path = os.path.join(all_path, f"series_{i}")
        os.makedirs(series_path, exist_ok=True)
        reader.set_scene(series)
        print(f"series {i}: shape = {reader.dims.shape}")
        num_channels = reader.dims.C 
        
        pxsz = reader.physical_pixel_sizes
        vxy = abs(float(pxsz.X))
        vz  = abs(float(pxsz.Z))

        for channel in range(0,num_channels):

            if split_frames == False:
                zyx = reader.get_image_data("ZYX", T=0, C=channel)
                name_c = f"{date}_s{i}_ch{channel}.tif"
                channel_path = os.path.join(series_path,f"channel_{channel}")
                
                os.makedirs(channel_path, exist_ok=True)
                
                imwrite(os.path.join(channel_path, name_c), zyx,
                        imagej=True, resolution=(1/vxy, 1/vxy),
                        metadata={'spacing': vz, 'unit': 'um', 'axes': 'ZYX'})
            else:
                channel_path = os.path.join(series_path,f"channel_{channel}" )
                os.makedirs(channel_path, exist_ok=True)
                num_frames = reader.dims.Z
                for frame in range(0,num_frames):
                    yx = reader.get_image_data("YX", T=0, C=channel, Z=frame)
                    name_f = f"{date}_s{i}_ch{channel}_f{frame}.tif"
                    imwrite(os.path.join(channel_path, name_f), yx,
                            imagej=True, resolution=(1/vxy, 1/vxy),
                            metadata={'unit': 'um', 'axes': 'YX'})
                    
def signal_extractor(mask_file_path, file_path, force_one_roi= False, print_area = False, Messages=False):
    """
    

    Args:
        mask_source (str): string with the name of the directory where masks are stored(ex: '06.19.25_f1_r_00001_ch_2_msk')
        filename (str): ex: '06.19.25_f1_r_00003_ch_1.tif'
    """
    
    ## idea in the future: 
    # date_str = filename.split('_')[0]
    # input_base_path = '/home/gerard/data/confocal'
    # file_path =  os.path.join(input_base_path, date_str, filename)
    # mask_dir_path =  os.path.join(input_base_path, date_str, mask_source)
    results= {}
    with tiff.TiffFile(file_path) as tif:
        # Read each frame manually and stack into a NumPy array
        stack = np.stack([page.asarray() for page in tif.pages])
    
    for frame in range(stack.shape[0]):
        # mask_file_name = os.path.join(mask_dir_path, mask_source.replace("msk", f"{str(frame)}.tif"))
        mask = tiff.imread(mask_file_path)
        if Messages:
            print(f"frame: {frame}")
        frame_signal = stack[frame]
        empty_roi = False
        factor = 0
        
        if force_one_roi:
            mask_mask = (mask != 1) & (mask != 0)            
            mask[mask_mask] = 1
        
        if print_area:
            print(f"area roi:{mask.sum()}")
            
        for roi_num in range(1, mask.max()+1):
            if Messages:
                print('roi number:' , roi_num)
            mask_per_roi = mask == roi_num
            
            # if (mask_per_roi == False).all():
            #     print(f"empty roi =  {roi_num}")
            #     empty_roi = True
            #     factor += 1
            #     continue
                 
            roi_intensities = frame_signal[mask_per_roi]
            mean_intensity = np.mean(roi_intensities)
            # print('mean_intensity', mean_intensity)
            if empty_roi:
                roi_num = roi_num - factor
            if str(frame) not in results:
                results[str(frame)] = {}
                    
            results[str(frame)][str(roi_num)] = mean_intensity
     
    return results   
                    
def pooler(signals,mask, channel, mask_number):
    pooled = []
    for sig in signals:
        val = sig[mask][channel][str(mask_number)]
        pooled.append(val)
    return pooled
    
    

def plot_bars_with_sem3(groups, labels=None, ylabel="Value", figsize=(3,6),
                       bar_color="lightgray", pattern = '', dot_color="black", spine_width=2):
    """
    Plot multiple groups as bars with SEM and overlay individual data points.

    Parameters
    ----------
    groups : list of array-like
        List of numeric arrays/lists, one per group.
    labels : list of str, optional
        Labels for each group (x-axis).
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        (width, height) of the figure in inches.
    bar_color : str, optional
        Color of the bars.
    dot_color : str, optional
        Color of the overlaid dots.
    spine_width : float, optional
        Thickness of the axis spines.
    """

    groups = [np.asarray(g) for g in groups]
    n_groups = len(groups)

    
    means = [np.nanmean(g) for g in groups]
    sems  = [np.nanstd(g, ddof=1) / np.sqrt(np.sum(~np.isnan(g))) for g in groups]
    
    for i,lbl in enumerate(labels):
        print(f"group: {lbl}, mean: {means[i]}, sem: {sems[i]}")
    if labels is None:
        labels = [f"Group {i+1}" for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    if len(bar_color) > 1:
        if len(bar_color) != n_groups:
           num_groups = n_groups / len(bar_color)
           bar_color = bar_color * int(num_groups)
           
    if len(pattern) > 1:
        if len(pattern) != n_groups:
           num_groups = n_groups / len(pattern)
           pattern = pattern * int(num_groups)
            # raise ValueError("Length of bar_color list must match number of groups.")
        # bar_colors = bar_color
    # Bars + SEM
    ax.bar(x, means, yerr=sems, capsize=8,
           color=bar_color, edgecolor="black",hatch = pattern, linewidth=1.5)

    # Overlay dots
    for i, vals in enumerate(groups):
        # jitter = (np.random.rand(len(vals)) - 0.5) * 0.2
        jitter = 0
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
                   color=dot_color, s=40, alpha=0.8, zorder=3)

    # Axis labels & limits
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)

    # --- Style adjustments ---
    # Thicker left & bottom spines, remove top/right
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(spine_width)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Ticks only on left/bottom
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # Add baseline at y = 0
    ax.axhline(0, color="black", linewidth=1.2)

    plt.tight_layout()
    return ax





def signals(date, user, series_n, masks_suffixes, channels_suffixes =[], server=False):
    """_summary_

    Args:
        date (_type_): _description_
        user (_type_): _description_
        series_n (_type_): _description_
        suffix_masks (_type_): _description_
        server (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    system = platform.system()
    if system == 'Linux':
        home = '/home'
    elif system == 'Darwin':
        home = '/Users'
    
    if server:
        base_dir = home + '/gerard/ITB/home/data/confocal'
    else:
        base_dir = home + '/gerard/data/confocal'
    
    all_path = os.path.join(base_dir, f"{date}_{user}")
    
    series = 'series_' + str(series_n)
    s = 's' + str(series_n)
    
    
    
    signals = {}
    for masks_suffix in masks_suffixes:
        mask = f'{all_path}/{series}/masks/msk_{masks_suffix}.tif'
        
        
        path = Path(f'{all_path}/{series}/projections/')
        # print(path)

        substring = masks_suffix

        files = [
            p for p in path.iterdir()
            if (
                p.is_file()
                and not p.name.startswith(".")   # exclude ._*, .DS_Store, etc.
                and substring in p.name          # keep only names containing substring

            )
        ]
        signals[masks_suffix] = {}
        for chn, f in enumerate(files):
            print(f)
            # signals[masks_suffix][str(chn)] = signal_extractor(mask, f)
            s = signal_extractor(mask, f)
            if channels_suffixes:
                key_channel = channels_suffixes[chn]
            else:
                key_channel = 'ch' + str(chn)

            signals[masks_suffix][key_channel] = s['0']
            # signals[masks_suffix][str(chn)] = signal_extractor(mask, f)

    
    return signals      
    
    # def ratios(signals, left_numerator, right_numerator, left_denominator, right_denominator):
        
    #     pass



    # al = signal_extractor(msk_AL, f'{all_path}/{series}/projections/{date}_{s}_ch0_AL.tif')
    
    # brp = signal_extractor(msk, f'{all_path}/{series}/projections/{date}_{s}_ch0_MBa.tif')
    # mito = signal_extractor(msk, f'{all_path}/{series}/projections/{date}_{s}_ch1_MBa.tif')
    # hsp = signal_extractor(msk, f'{all_path}/{series}/projections/{date}_{s}_ch2_MBa.tif')

    # denominatorL = al['0']['1']
    # denominatorR = al['0']['2']
    
    # results = {}
    # for channel in channels:
    #     results[channel] = {}
    # # results['brp'] = {}
    # # results['mito'] = {}
    # # results['hsp'] = {}
    
    # results['brp']['a1'] = brp['0']['1'] / denominatorL, brp['0']['4'] / denominatorR
    # results['brp']['a2'] = brp['0']['2'] / denominatorL, brp['0']['5'] / denominatorR
    # results['brp']['a3'] = brp['0']['3'] / denominatorL, brp['0']['6'] / denominatorR

    # results['mito']['a1'] = mito['0']['1'] / denominatorL, mito['0']['4'] / denominatorR
    # results['mito']['a2'] = mito['0']['2'] / denominatorL, mito['0']['5'] / denominatorR
    # results['mito']['a3'] = mito['0']['3'] / denominatorL, mito['0']['6'] / denominatorR
 
    
    # results['hsp']['a1'] = hsp['0']['1'] / denominatorL, hsp['0']['4'] / denominatorR
    # results['hsp']['a2'] = hsp['0']['2'] / denominatorL, hsp['0']['5'] / denominatorR
    # results['hsp']['a3'] = hsp['0']['3'] / denominatorL, hsp['0']['6'] / denominatorR
    

    
    
    # return results
    
    
# THRESHOLDING #####################################################


def fast_threshold_li(stack):
    s_min, s_max = float(stack.min()), float(stack.max())
    if s_max == s_min:
        return s_min
    u16 = ((stack - s_min) / (s_max - s_min) * 65535).astype(np.uint16)
    t_u16 = threshold_li(u16)
    return t_u16 / 65535 * (s_max - s_min) + s_min


def thresholding(date, user, series_list, deconv_iter_list, do_plot = True ):
    
    server = False
    system = platform.system()
    if system == 'Linux':
        if server:
            base_dir = '/home/gerard/ITB/home/data/confocal/'
        else:
            base_dir = '/home/gerard/data/confocal/'
        
    elif system == 'Darwin':
        if server:
            base_dir = '/Users/gerard/ITB/home/data/confocal/'
        else:
            base_dir = '/Users/gerard/data/confocal/'
        
    elif system == "Windows":
        base_dir = 'C:/Users/cviko/data/confocal/'
    print(base_dir + date + '_' + user + '/Project.lif')
    info = describe_acquisition( base_dir + date + '_' + user + '/Project.lif', False)
    
    
    results = {}
    
    treshold_algorithm_list =[
    'triangle',
    'otsu',
    'yen',
    'li',
    ]
    
    
    for series in series_list:
        print(f'Processing series {series}---------------')
        results['s_'+str(series)] = {}
        
        channel_list = list(range(len(info[list(info.keys())[series]]['image_channels']))) 
        
        
        for channel in channel_list:
            results['s_'+str(series)]['c_'+str(channel)] = {}
            print(f'Channel {channel}************************')
            
            series_str = f'series_{series}'
            channel_str = f'channel_{channel}'
            
            base = os.path.join(base_dir,f'{date}_{user}',f'series_{series}', f'channel_{channel}')
            #path_series_channel = f'gerard/data/confocal/2026_05_26_Gerardo/series_{series}/channel_{channel}'#gerard/data/confocal/2026_05_26_Gerardo/Project.lif'
            #base = home + path_series_channel #os.path.join(home, )
            
            path_raw        = os.path.join(base, f'{date}_s{series}_ch{channel}.tif')
            path_raw_masks  = os.path.join(base, f'{date}_s{series}_ch{channel}_masks')
            
            
            for deconv_iter in deconv_iter_list:
            
                
                path_deconv      = os.path.join(base, f'{date}_s{series}_ch{channel}_deconv_iter_{deconv_iter}.tif')
                path_deconv_masks = os.path.join(base, f'{date}_s{series}_ch{channel}_deconv_iter_{deconv_iter}_masks')

                for path_in, path_out in [(path_raw, path_raw_masks), (path_deconv, path_deconv_masks)]:
                    
                    if 'deconv' in path_in:
                        deconv = True
                        key_dict = 'deconv_iter_'+str(deconv_iter)
                    else:
                            
                        deconv = False
                        key_dict = 'raw'
                    
                    print(key_dict, '************************')    
                    if not key_dict in results['s_'+str(series)]['c_'+str(channel)]:
                        results['s_'+str(series)]['c_'+str(channel)][key_dict] = {}

                    
                        with tiff.TiffFile(path_in) as tf:
                            stack = tf.asarray()
                            vxy = 1.0 / (tf.pages[0].tags['XResolution'].value[0] / tf.pages[0].tags['XResolution'].value[1])
                            vz  = tf.imagej_metadata['spacing']
                        
                        #stack_proj = stack.sum(axis=0)
                        stack_proj = stack.sum(axis=0).astype(np.float32)
    
                        results['s_'+str(series)]['c_'+str(channel)][key_dict]['stack'] = stack
                        results['s_'+str(series)]['c_'+str(channel)][key_dict]['stack_proj'] = stack_proj

                        for treshold_algorithm in treshold_algorithm_list:
                            results['s_'+str(series)]['c_'+str(channel)][key_dict][treshold_algorithm] = {}
                            
                        
                            if treshold_algorithm == 'triangle':
                                thresh = threshold_triangle(stack)
                                thresh_proj = threshold_triangle(stack_proj)
                            elif treshold_algorithm == 'otsu':
                                thresh = threshold_otsu(stack)
                                thresh_proj = threshold_otsu(stack_proj)
                            elif treshold_algorithm == 'yen':
                                thresh = threshold_yen(stack)
                                thresh_proj = threshold_yen(stack_proj) 
                            elif treshold_algorithm == 'li':
                                thresh = fast_threshold_li(stack)
                                thresh_proj = fast_threshold_li(stack_proj)
                            
                        

                            
                            #stack_flat = stack[stack > 0]
                            # pct_90 = np.percentile(stack_flat, 90)
                            # pct_95 = np.percentile(stack_flat, 95)
                            
                            print(f'treshold using {treshold_algorithm} = {thresh}')
                        
                            masks  = (stack > thresh).astype(np.uint8)
                            mask_proj = stack_proj > thresh_proj
                            
                            results['s_'+str(series)]['c_'+str(channel)][key_dict][treshold_algorithm]['threshold'] = thresh
                            results['s_'+str(series)]['c_'+str(channel)][key_dict][treshold_algorithm]['masks'] = masks
                            results['s_'+str(series)]['c_'+str(channel)][key_dict][treshold_algorithm]['mask_proj'] = mask_proj
                            


                            path_out_al = path_out + '_' + treshold_algorithm + '.tif'
                            imwrite(path_out_al, masks, imagej=True, resolution=(1/vxy, 1/vxy),
                                    metadata={'spacing': vz, 'unit': 'um', 'axes': 'ZYX'})
                            
                            print(f'saved {path_out_al}')
                            
                        if do_plot:
                            plt.figure()
                            plt.hist(stack.ravel(), bins=256)
                            colors = [cm.tab10(i / len(treshold_algorithm_list)) for i in range(len(treshold_algorithm_list))]
                            
                            for i, thrsh_al in enumerate(treshold_algorithm_list):

                                # print(thrsh_al, '<------------')
                                thresh = results['s_'+str(series)]['c_'+str(channel)][key_dict][thrsh_al]['threshold']
                                plt.axvline(thresh, color=colors[i], linestyle='--', label=thrsh_al)
                            
                
                            
                            plt.legend(loc='upper right', title='Thresholds')
                            plt.title(f'series = {series}, channel = {channel}, {key_dict}')
                            # plt.axvline(pct_90, color='c', linestyle='--')
                            # plt.axvline(pct_95, color='m', linestyle='--')
                            
                            vmin, vmax = np.percentile(stack.ravel(), [1, 100])
                            plt.xlim(0,vmax)# stack.max())
                            plt.show()
        return results
# COLOCALIZATION #####################################################


def compute_colocalization(dict_data, series, chA_n, chA_thrsh_al, chB_n, chB_thrsh_al, deconv_iter, min_val=500, do_plot=True):
   
    if deconv_iter != 0:
        deconv_str = 'deconv_iter_'+str(deconv_iter)
    else:
        deconv_str = 'raw'

    chA = dict_data['s_'+str(series)]['c_'+str(chA_n)][deconv_str]['stack']
    chB = dict_data['s_'+str(series)]['c_'+str(chB_n)][deconv_str]['stack']
    chA_proj = dict_data['s_'+str(series)]['c_'+str(chA_n)][deconv_str]['stack_proj']
    chB_proj = dict_data['s_'+str(series)]['c_'+str(chB_n)][deconv_str]['stack_proj']

    mask_chA = dict_data['s_'+str(series)]['c_'+str(chA_n)][deconv_str][chA_thrsh_al]['masks'].astype(bool)
    mask_chB = dict_data['s_'+str(series)]['c_'+str(chB_n)][deconv_str][chB_thrsh_al]['masks'].astype(bool)
    mask_chA_proj = dict_data['s_'+str(series)]['c_'+str(chA_n)][deconv_str][chA_thrsh_al]['mask_proj'].astype(bool)
    mask_chB_proj = dict_data['s_'+str(series)]['c_'+str(chB_n)][deconv_str][chB_thrsh_al]['mask_proj'].astype(bool)

    
    
    mask_union = mask_chA | mask_chB

    # per-frame Pearson R and Manders
    r_per_frame  = []
    M1_per_frame = []  # fraction of chA signal in chB-positive pixels
    M2_per_frame = []  # fraction of chB signal in chA-positive pixels

    for z in range(chA.shape[0]):
        chA_f, chB_f     = chA[z], chB[z]
        union_f          = mask_union[z]
        maskA_f, maskB_f = mask_chA[z], mask_chB[z]

        if union_f.sum() < min_val:
            r_per_frame.append(np.nan)
            M1_per_frame.append(np.nan)
            M2_per_frame.append(np.nan)
            continue

        r = np.corrcoef(chA_f[union_f], chB_f[union_f])[0, 1]
        r_per_frame.append(r)

        # M1: fraction of chA signal that falls in chB-positive pixels
        chA_total = chA_f[union_f].sum()
        M1 = chA_f[maskB_f].sum() / chA_total if chA_total > 0 else np.nan
        M1_per_frame.append(M1)

        # M2: fraction of chB signal that falls in chA-positive pixels
        chB_total = chB_f[union_f].sum()
        M2 = chB_f[maskA_f].sum() / chB_total if chB_total > 0 else np.nan
        M2_per_frame.append(M2)

    # sum-projection metrics (single value per stack)
    # chA_proj   = chA.sum(axis=0)
    # chB_proj   = chB.sum(axis=0)

    # threshold the sum projections directly — foreground = bright integrated across all frames
    # maskA_proj = chA_proj > fast_threshold_li(chA_proj)
    # maskB_proj = chB_proj > fast_threshold_li(chB_proj)
    # previous version (too permissive — foreground if above threshold in any single frame):
    # maskA_proj = mask_chA.any(axis=0)
    # maskB_proj = mask_chB.any(axis=0)

    union_proj = mask_chA_proj | mask_chB_proj

    r_sum = np.corrcoef(chA_proj[union_proj], chB_proj[union_proj])[0, 1]
    M1_sum = chA_proj[mask_chB_proj].sum() / chA_proj[union_proj].sum()
    M2_sum = chB_proj[mask_chA_proj].sum() / chB_proj[union_proj].sum()

    if do_plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(r_per_frame)
        axes[0].axhline(r_sum, color='r', linestyle='--', label=f'sum proj R={r_sum:.2f}')
        axes[0].set_title(f'Pearson R  ch{chA_n} vs ch{chB_n}')
        axes[0].set_xlabel('Frame'); axes[0].set_ylabel('R'); axes[0].legend()

        axes[1].plot(M1_per_frame)
        axes[1].axhline(M1_sum, color='r', linestyle='--', label=f'sum proj M1={M1_sum:.2f}')
        axes[1].set_title(f'M1: ch{chA_n} signal in ch{chB_n}-positive pixels')
        axes[1].set_xlabel('Frame'); axes[1].set_ylabel('M1'); axes[1].legend()

        axes[2].plot(M2_per_frame)
        axes[2].axhline(M2_sum, color='r', linestyle='--', label=f'sum proj M2={M2_sum:.2f}')
        axes[2].set_title(f'M2: ch{chB_n} signal in ch{chA_n}-positive pixels')
        axes[2].set_xlabel('Frame'); axes[2].set_ylabel('M2'); axes[2].legend()

        plt.suptitle(f'Series {series}  |  deconv iter {deconv_iter}  |  threshold algorithms: {chA_thrsh_al} (ch{chA_n}), {chB_thrsh_al} (ch{chB_n})')
        plt.tight_layout()
        plt.show()

    return {
        'r_per_frame':  r_per_frame,
        'M1_per_frame': M1_per_frame,
        'M2_per_frame': M2_per_frame,
        'r_sum':        r_sum,
        'M1_sum':       M1_sum,
        'M2_sum':       M2_sum,
    }


######################################## deconvolution





def parse_lif_psf_params(lif_path, scene=0):
    """
    Extract optical parameters needed for PSF computation from a Leica .lif file.

    Returns
    -------
    dict with keys:
        NA              : numerical aperture (float)
        n               : refractive index of immersion medium (float)
        voxel_z_um      : z-step size in µm (float)
        voxel_xy_um     : lateral pixel size in µm (float)
        emission_nm     : list of emission wavelengths per channel in nm.
                          Uses the known fluorophore emission peak when the
                          DyeName field is set in the .lif metadata; falls back
                          to the midpoint of the spectral detection band otherwise.
    """
    reader = LifReader(lif_path)
    reader.set_scene(reader.scenes[scene])

    pxsz = reader.physical_pixel_sizes
    voxel_z_um  = abs(float(pxsz.Z))
    voxel_xy_um = abs(float(pxsz.X))

    NA, n = None, 1.518
    for elem in reader.metadata.iter():
        if 'NumericalAperture' in elem.attrib:
            #print('here')
            NA = float(elem.attrib['NumericalAperture'])
            n  = float(elem.attrib.get('RefractionIndex', '1.518'))
            break


    # Known fluorophore emission peaks (nm). Used when Leica's DyeName field is set.
    # Falls back to detection-band midpoint when DyeName is absent or unrecognised.
    _dye_emission = {
        'Leica/ALEXA 405':  430,
        'Leica/ALEXA 488':  519,
        'Leica/ALEXA 546':  573,
        'Leica/ALEXA 555':  565,
        'Leica/ALEXA 568':  603,
        'Leica/ALEXA 594':  617,
        'Leica/ALEXA 633':  647,
        'Leica/ALEXA 647':  668,
        'Leica/ALEXA 680':  702,
        'Leica/ALEXA 750':  779,
        'Leica/Cy3':        570,
        'Leica/Cy5':        670,
        'Leica/Cy7':        800,
        'Leica/FITC':       519,
        'Leica/TRITC':      573,
    }

    # Build emission_nm by parsing each ATLConfocalSettingDefinition block (one per
    # acquisition sequence) rather than flattening the whole tree. This correctly
    # handles sequential acquisitions where the same detector channel number is reused
    # across sequences for different dyes.
    #
    # Channel ordering: sort sequences by minimum active laser wavelength, then within
    # each sequence sort by detector channel number. This matches how LifReader assigns
    # image channel indices for sequential acquisitions.
    #
    # Blocks with >2 simultaneous lasers are skipped (live-view / alignment scans).
    sequences = []
    seen_configs = set()

    for block in reader.metadata.iter('ATLConfocalSettingDefinition'):
        active_lasers = [
            int(e.attrib['LaserLine'])
            for e in block.iter('LaserLineSetting')
            if float(e.attrib.get('IntensityDev', '0')) > 0
        ]
        if not active_lasers or len(active_lasers) > 2:
            continue

        active_det_chs = {
            int(e.attrib['Channel'])
            for e in block.iter('Detector')
            if e.attrib.get('IsActive') == '1' and e.attrib.get('ScanType') == 'Internal'
        }
        if not active_det_chs:
            continue

        chan_emission = {}
        for mb in block.iter('MultiBand'):
            ch = int(mb.attrib['Channel'])
            if ch not in active_det_chs:
                continue
            dye = mb.attrib.get('DyeName', '')
            if dye in _dye_emission:
                chan_emission[ch] = _dye_emission[dye]
            else:
                left  = float(mb.attrib['LeftWorld'])
                right = float(mb.attrib['RightWorld'])
                chan_emission[ch] = (left + right) / 2.0

        if not chan_emission:
            continue

        # Deduplicate: same dye/band configuration appears once per series
        config_key = frozenset(chan_emission.items())
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)

        sequences.append({'min_laser': min(active_lasers), 'channels': chan_emission})

    # Sort sequences by minimum active laser wavelength, then flatten
    sequences.sort(key=lambda s: s['min_laser'])
    emission_nm = [
        em
        for seq in sequences
        for _, em in sorted(seq['channels'].items())
    ]

    return {
        'NA': NA,
        'n': n,
        'voxel_z_um': voxel_z_um,
        'voxel_xy_um': voxel_xy_um,
        'emission_nm': emission_nm,
    }


def describe_acquisition(lif_path, do_print=True):
    print("....")
    
    """
    Parse a Leica .lif file and print a per-scene summary of acquisition
    parameters: objective, sequential acquisition structure, per-channel PSF
    parameters, Nyquist status, and spectral bleed-through risks.

    Returns
    -------
    dict keyed by scene name.  Each value contains:
        objective, NA, n, voxel_xy_um, voxel_z_um, shape_zyx,
        sequences      -- list of {lasers, channels} dicts
        image_channels -- list of per-channel PSF/Nyquist dicts (0-indexed)
        bleed_through  -- list of bleed-through warning dicts
    """
    _dye_emission = {
        'Leica/ALEXA 405': 430, 'Leica/ALEXA 488': 519, 'Leica/ALEXA 546': 573,
        'Leica/ALEXA 555': 565, 'Leica/ALEXA 568': 603, 'Leica/ALEXA 594': 617,
        'Leica/ALEXA 633': 647, 'Leica/ALEXA 647': 668, 'Leica/ALEXA 680': 702,
        'Leica/ALEXA 750': 779, 'Leica/Cy3': 570, 'Leica/Cy5': 670,
        'Leica/Cy7': 800, 'Leica/FITC': 519, 'Leica/TRITC': 573,
    }
    _dye_excitation = {
        'Leica/ALEXA 405': 402, 'Leica/ALEXA 488': 495, 'Leica/ALEXA 546': 556,
        'Leica/ALEXA 555': 555, 'Leica/ALEXA 568': 578, 'Leica/ALEXA 594': 590,
        'Leica/ALEXA 633': 632, 'Leica/ALEXA 647': 650, 'Leica/ALEXA 680': 679,
        'Leica/ALEXA 750': 749, 'Leica/Cy3': 550, 'Leica/Cy5': 649,
        'Leica/Cy7': 743, 'Leica/FITC': 495, 'Leica/TRITC': 547,
    }

    scene_names = LifReader(lif_path).scenes
    all_scenes = {}

    for scene_idx, scene_name in enumerate(scene_names):
        # Fresh reader per scene — avoids physical_pixel_sizes caching across scenes
        reader = LifReader(lif_path)
        reader.set_scene(scene_name)

        pxsz = reader.physical_pixel_sizes
        vxy = abs(float(pxsz.X))
        vz  = abs(float(pxsz.Z))
        nz, ny, nx = reader.dims.Z, reader.dims.Y, reader.dims.X

        obj_name, NA, n = 'unknown', 1.4, 1.518
        for elem in reader.metadata.iter():
            if 'NumericalAperture' in elem.attrib:
                NA = float(elem.attrib['NumericalAperture'])
                n  = float(elem.attrib.get('RefractionIndex', '1.518'))
                obj_name = elem.attrib.get('ObjectiveName', 'unknown').strip()
                break

        # Parse unique acquisition sequences (same logic as parse_lif_psf_params)
        sequences = []
        seen_configs = set()
        for block in reader.metadata.iter('ATLConfocalSettingDefinition'):
            active_lasers = sorted([
                int(e.attrib['LaserLine'])
                for e in block.iter('LaserLineSetting')
                if float(e.attrib.get('IntensityDev', '0')) > 0
            ])
            if not active_lasers or len(active_lasers) > 2:
                continue
            active_det_chs = {
                int(e.attrib['Channel'])
                for e in block.iter('Detector')
                if e.attrib.get('IsActive') == '1' and e.attrib.get('ScanType') == 'Internal'
            }
            if not active_det_chs:
                continue
            seq_chs = []
            for mb in block.iter('MultiBand'):
                ch = int(mb.attrib['Channel'])
                if ch not in active_det_chs:
                    continue
                dye  = mb.attrib.get('DyeName', '')
                left  = float(mb.attrib['LeftWorld'])
                right = float(mb.attrib['RightWorld'])
                em = _dye_emission.get(dye, (left + right) / 2.0)
                seq_chs.append({'detector_ch': ch, 'band_nm': (left, right),
                                 'dye_name': dye, 'emission_nm': em})
            if not seq_chs:
                continue
            config_key = frozenset((c['detector_ch'], c['emission_nm']) for c in seq_chs)
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            pinhole_m    = float(block.attrib.get('Pinhole', 0))
            pinhole_airy = float(block.attrib.get('PinholeAiry', 0))
            sequences.append({
                'lasers':       active_lasers,
                'channels':     sorted(seq_chs, key=lambda c: c['detector_ch']),
                'pinhole_um':   round(pinhole_m * 1e6, 2),
                'pinhole_airy': round(pinhole_airy, 3),
            })
        sequences.sort(key=lambda s: min(s['lasers']))

        # Build image_channels with PSF + Nyquist info (ordered by acquisition sequence)
        image_channels = []
        for seq in sequences:
            for ch in seq['channels']:
                lam    = ch['emission_nm'] * 1e-3  # nm → µm
                sxy_um = 0.21 * lam / NA
                sz_um  = 0.66 * lam * n / NA ** 2
                sxy_px = sxy_um / vxy
                sz_px  = sz_um  / vz
                image_channels.append({
                    'index':               len(image_channels),
                    'dye':                 ch['dye_name'],
                    'emission_nm':         ch['emission_nm'],
                    'band_nm':             ch['band_nm'],
                    'acquired_with_lasers': seq['lasers'],
                    'sigma_xy_um':         round(sxy_um, 4),
                    'sigma_xy_px':         round(sxy_px, 3),
                    'sigma_z_um':          round(sz_um, 4),
                    'sigma_z_px':          round(sz_px, 3),
                    'nyquist_xy':          sxy_px >= 2.0,
                    'nyquist_z':           sz_px  >= 2.0,
                    'deconv_mode':         '3D' if sz_px >= 2.0 else '2D-per-frame',
                })

        # Bleed-through: for each sequence, find non-target dyes that could be
        # cross-excited by the active laser(s) and emit into an active detector band.
        # Flag only when laser is bluer than the dye's excitation peak by up to 100 nm,
        # or redder by up to 30 nm (absorption tail is negligible beyond +30 nm).
        all_named_dyes = {
            c['dye_name']: _dye_emission[c['dye_name']]
            for seq in sequences for c in seq['channels']
            if c['dye_name'] in _dye_emission
        }
        bleed_through = []
        for seq in sequences:
            target_dyes = {c['dye_name'] for c in seq['channels']}
            for laser in seq['lasers']:
                for dye, em_nm in all_named_dyes.items():
                    if dye in target_dyes:
                        continue
                    ex_peak = _dye_excitation.get(dye)
                    if ex_peak is None:
                        continue
                    delta = laser - ex_peak  # negative: laser bluer than peak
                    if delta > 30 or delta < -100:
                        continue  # outside plausible cross-excitation range
                    for det_ch in seq['channels']:
                        bl, br = det_ch['band_nm']
                        if bl <= em_nm <= br:
                            severity = ('HIGH'   if abs(delta) < 30  else
                                        'MEDIUM' if abs(delta) < 70  else 'LOW')
                            tgt_idx = next((ic['index'] for ic in image_channels
                                            if ic['band_nm'] == det_ch['band_nm']), '?')
                            bleed_through.append({
                                'source_dye':                  dye,
                                'source_emission_nm':          em_nm,
                                'source_excitation_peak_nm':   ex_peak,
                                'cross_excitation_laser_nm':   laser,
                                'delta_laser_to_peak_nm':      delta,
                                'target_channel_index':        tgt_idx,
                                'target_channel_dye':          det_ch['dye_name'],
                                'target_band_nm':              (bl, br),
                                'severity':                    severity,
                            })

        scene_data = {
            'scene_name':    scene_name,
            'objective':     obj_name,
            'NA':            NA,
            'n':             n,
            'voxel_xy_um':   vxy,
            'voxel_z_um':    vz,
            'shape_zyx':     (nz, ny, nx),
            'sequences':     sequences,
            'image_channels': image_channels,
            'bleed_through': bleed_through,
        }
        all_scenes[scene_name] = scene_data

        # ── printed summary ────────────────────────────────────────────────────
        if do_print:
            W = 62
            print(f'\n{"═"*W}')
            print(f'Scene {scene_idx}: {scene_name}')
            print('═'*W)
            print(f'  Objective  : {obj_name}')
            print(f'  NA / n     : {NA} / {n}')
            print(f'  Voxel size : XY = {vxy:.4f} µm/px  |  Z = {vz:.4f} µm/step')
            print(f'  Shape      : {nz} × {ny} × {nx}  (Z × Y × X)')

            print(f'\n  {"─"*38} Acquisition sequences')
            for s_i, seq in enumerate(sequences):
                laser_str = ' + '.join(f'{l} nm' for l in seq['lasers'])
                tag = '  [simultaneous]' if len(seq['lasers']) > 1 else ''
                print(f'  Seq {s_i+1}  laser(s): {laser_str}{tag}')
                print(f'         pinhole  : {seq["pinhole_um"]:.2f} µm  ({seq["pinhole_airy"]:.3f} AU)')
                for ch in seq['channels']:
                    dye_str = ch['dye_name'] if ch['dye_name'] else '(unnamed)'
                    print(f'         det ch{ch["detector_ch"]}  '
                        f'{ch["band_nm"][0]:.0f}–{ch["band_nm"][1]:.0f} nm  '
                        f'{dye_str:<22}  em = {ch["emission_nm"]:.0f} nm')

            print(f'\n  {"─"*38} Image channels (0-indexed)')
            for ic in image_channels:
                dye_str  = ic['dye'] if ic['dye'] else '(unnamed)'
                nxy = '✓' if ic['nyquist_xy'] else '✗ sub-Nyquist'
                nz_ = '✓' if ic['nyquist_z']  else '✗ sub-Nyquist'
                print(f'  ch{ic["index"]}  {dye_str:<22}  em={ic["emission_nm"]:.0f} nm  '
                    f'σ_xy={ic["sigma_xy_px"]:.2f}px ({ic["sigma_xy_um"]:.3f}µm) [{nxy}]  '
                    f'σ_z={ic["sigma_z_px"]:.2f}px [{nz_}]  → {ic["deconv_mode"]}')

            print(f'\n  {"─"*38} Spectral bleed-through')
            if bleed_through:
                for bt in bleed_through:
                    print(f'  ⚠  [{bt["severity"]}]  {bt["source_dye"]}  →  '
                        f'ch{bt["target_channel_index"]} ({bt["target_channel_dye"]})')
                    print(f'       {bt["cross_excitation_laser_nm"]} nm laser cross-excites '
                        f'{bt["source_dye"]} (ex peak {bt["source_excitation_peak_nm"]} nm, '
                        f'Δ={bt["delta_laser_to_peak_nm"]:+d} nm)')
                    print(f'       emission at {bt["source_emission_nm"]} nm falls in detection '
                        f'band {bt["target_band_nm"][0]:.0f}–{bt["target_band_nm"][1]:.0f} nm')
            else:
                print('  None detected.')

    return all_scenes


def _gaussian_psf(sigma_xy_px, sigma_z_px=None):
    """Build a normalized Gaussian PSF kernel. 3D if sigma_z_px is given, else 2D."""
    def _odd_ceil(sigma):
        s = max(3, int(np.ceil(sigma * 6)))
        return s if s % 2 == 1 else s + 1

    if sigma_z_px is not None:
        kz, ky, kx = _odd_ceil(sigma_z_px), _odd_ceil(sigma_xy_px), _odd_ceil(sigma_xy_px)
        psf = np.zeros((kz, ky, kx), dtype=np.float64)
        psf[kz // 2, ky // 2, kx // 2] = 1.0
        psf = gaussian_filter(psf, sigma=[sigma_z_px, sigma_xy_px, sigma_xy_px])
    else:
        ky, kx = _odd_ceil(sigma_xy_px), _odd_ceil(sigma_xy_px)
        psf = np.zeros((ky, kx), dtype=np.float64)
        psf[ky // 2, kx // 2] = 1.0
        psf = gaussian_filter(psf, sigma=[sigma_xy_px, sigma_xy_px])
    return psf / psf.sum()


def deconvolve(stack, lif_path, channel, scene=0, num_iter=15, emission_nm=None, forced2d = False):
    """
    Deconvolve a confocal image using Richardson-Lucy with a theoretical
    Gaussian PSF derived from the microscope metadata in the .lif file.

    Parameters
    ----------
    stack : ndarray, shape (Z, Y, X) or (Y, X)
        Raw confocal z-stack or single 2D frame (any unsigned integer dtype).
    lif_path : str
        Path to the originating Leica .lif file.
    channel : int
        0-based channel index (selects the matching emission wavelength).
    scene : int
        Series index within the .lif file.
    num_iter : int
        Number of Richardson-Lucy iterations (10–30 is typical; more iterations
        sharpen further but amplify noise).
    emission_nm : float or None
        Known fluorophore emission peak in nm. If provided, overrides the
        detection-band midpoint read from the .lif metadata (which can be
        significantly off for wide detection windows). E.g. 519 for Alexa 488,
        573 for Alexa 546, 670 for Cy5.

    Returns
    -------
    ndarray, same shape as stack, dtype float32
        Deconvolved image scaled back to the original intensity range.

    Notes
    -----
    Automatically selects 3D or 2D-per-frame deconvolution based on Nyquist:
    if σ_z < 2 px (Z undersampled, pixel > σ_z/2), applies 2D RL to each
    frame independently using only the XY PSF.
    With coarse XY sampling (>0.5 µm/px), σ_xy may be <1 px and the effect
    will be subtle — most visible as reduced lateral blur.
    """
    params = parse_lif_psf_params(lif_path, scene)
    NA  = params['NA']
    n   = params['n']
    vz  = params['voxel_z_um']
    vxy = params['voxel_xy_um']
    # Use known fluorophore emission peak if provided, else fall back to
    # detection-band midpoint from metadata (can be ~15-20% off for wide windows).
    lam = (emission_nm if emission_nm is not None else params['emission_nm'][channel]) * 1e-3  # nm → µm

    sigma_xy_um = 0.21 * lam / NA
    sigma_z_um  = 0.66 * lam * n / (NA ** 2)
    sigma_xy_px = sigma_xy_um / vxy
    sigma_z_px  = sigma_z_um  / vz

    is_3d = stack.ndim == 3
    # Nyquist: need σ ≥ 2 px (pixel ≤ σ/2) for the axis to be adequately sampled.
    # If Z is undersampled, fall back to 2D deconvolution applied per frame.
    use_3d_psf = is_3d and (sigma_z_px >= 2.0) and not forced2d
    
    if use_3d_psf:
        deconv_type = "3d"
    else:
        deconv_type = "2d"

    if is_3d:
        
        mode = '3D' if use_3d_psf else f'2D-per-frame (σ_z={sigma_z_px:.2f} px < 2, Z undersampled)'
        print(f"PSF (ch{channel}): λ={lam*1e3:.0f} nm | "
              f"σ_xy={sigma_xy_um:.3f} µm ({sigma_xy_px:.2f} px) | "
              f"σ_z={sigma_z_um:.3f} µm ({sigma_z_px:.2f} px) → {mode}")
        psf = _gaussian_psf(sigma_xy_px, sigma_z_px if use_3d_psf else None)
    else:
       
        print(f"PSF 2D (ch{channel}): λ={lam*1e3:.0f} nm | "
              f"σ_xy={sigma_xy_um:.3f} µm ({sigma_xy_px:.2f} px)")
        psf = _gaussian_psf(sigma_xy_px)

    def _process(img):
        img = img.astype(np.float64)
        # Replace only pixels that deviate strongly from their local median (hot pixels).
        # This avoids the striping artifact caused by blanket median filtering.
        # img = median_filter(img, size=3)
        local_median = median_filter(img, size=3)
        hot_pixels = (img - local_median) > (5.0 * img.std())
        img[hot_pixels] = local_median[hot_pixels]
        scale = img.max()
        if scale > 0:
            img /= scale
        out = richardson_lucy(img, psf, num_iter=num_iter, clip=True)
        return (out * scale).astype(np.float32)

    if is_3d and not use_3d_psf:
        result = np.stack([_process(stack[z]) for z in range(stack.shape[0])])
    else:
        result = _process(stack)

    # path deconv image:
    #'gerard/data/confocal/2026_05_26_Gerardo/Project.lif'
    name2remove = lif_path.split('/')[-1]
    glob_path = lif_path.replace('/' + name2remove, '')
    date = (lif_path.split('/')[-2])
    date = date.replace(date.split('_')[-1], '') # 2026_05_26_
    
    #print(date, '<-----------')
    #new_name = glob_path + 'series_'+ str(scene) +'/'+'channel_'+ str(channel)+'/'+'deconv.tif'
    #name = 's' + str(scene) + '_ch' + str(channel) + '_deconv_iter_' + str(num_iter) + '.tif'    #'s1_ch0_deconv_iter_4.tif

    name = date + 's' + str(scene) + '_ch' + str(channel) + '_deconv'+ deconv_type +'_iter_' + str(num_iter) + '.tif' # 2026_05_26_s1_ch0_deconv_iter_4.tif
    

    print(name)
    final_path = os.path.join(glob_path, 'series_'+ str(scene),'channel_'+ str(channel), name)

    axes = 'ZYX' if is_3d else 'YX'
    imwrite(final_path, result, imagej=True, resolution=(1/vxy, 1/vxy),
        metadata={'spacing': vz, 'unit': 'um', 'axes': axes})

    # return (result * scale).astype(np.float32)
    return result, sigma_xy_px


def analyze_deconvolution_results(result, stack):
    background_noise_ratios = []
    for z in range(result.shape[0]):
        ratio = result[z].max() / stack[z].max() + 1e-9
        if ratio > 5:
            print(f'Frame{z}: peak increased {ratio:2f}x - possible noise spike')
            
        bg_mask = stack[z] < np.percentile(stack[z], 10)
        
        # bg_orig_std == 0 means a flat/dark frame (e.g. all-zero edge slices).
        # After deconv to float32 those pixels get tiny non-zero values, so the
        # ratio blows up — not a real deconvolution problem, skip these frames.
        if bg_mask.sum() < 2:
            background_noise_ratios.append(0)
            continue
        
        bg_orig_std = stack[z][bg_mask].std()          # ← only called when safe
        
        if bg_orig_std == 0:
            background_noise_ratios.append(0)
            continue

        noise_ratio = result[z][bg_mask].std() / bg_orig_std
        
        if noise_ratio > 1.5:
            background_noise_ratios.append(1)
        else:
            background_noise_ratios.append(0)
    
    inspected = [r for r in background_noise_ratios if r is not None]
    flagged = sum(inspected)/len(inspected)
    print(f'{flagged :.3f} of inspected frames flagged')
    return flagged

################################

def raw_values(series_n):
    
    series = 'series_' + str(series_n)
    s = 's' + str(series_n)
    
    results = {}
    results['brp'] = {}
    results['mito'] = {}
    results['hsp'] = {}
    
    msk_AL = f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/masks/msk_AL.tif'
    msk_MBa = f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/masks/msk_MBa.tif'
    
    
    al = signal_extractor(msk_AL, f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/projections/2025_11_29_{s}_ch0_AL.tif')
    brp = signal_extractor(msk_MBa, f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/projections/2025_11_29_{s}_ch0_MBa.tif')
    mito = signal_extractor(msk_MBa, f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/projections/2025_11_29_{s}_ch1_MBa.tif')
    hsp = signal_extractor(msk_MBa, f'/home/gerard/data/confocal/2025_11_29_Gerardo/{series}/projections/2025_11_29_{s}_ch2_MBa.tif')
    
    
    results['brp']['a1'] = brp['0']['1'], brp['0']['4']
    results['brp']['a2'] = brp['0']['2'], brp['0']['5'] 
    results['brp']['a3'] = brp['0']['3'], brp['0']['6']

    results['mito']['a1'] = mito['0']['1'], mito['0']['4']
    results['mito']['a2'] = mito['0']['2'], mito['0']['5']
    results['mito']['a3'] = mito['0']['3'], mito['0']['6']
 
    
    results['hsp']['a1'] = hsp['0']['1'], hsp['0']['4']
    results['hsp']['a2'] = hsp['0']['2'], hsp['0']['5']
    results['hsp']['a3'] = hsp['0']['3'], hsp['0']['6']
    
    
    results['al'] =  al['0']['1'], al['0']['2']
    

    
    
    return results