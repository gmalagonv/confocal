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

def loader2(date,user,split_frames=False, server=False):
    
    """_summary_
    
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
        
        for channel in range(0,num_channels):

            if split_frames == False:
                zyx = reader.get_image_data("ZYX", T=0, C=channel)
                name_c = f"{date}_s{i}_ch{channel}.tif"
                imwrite(os.path.join(series_path, name_c), zyx)
            else:
                channel_path = os.path.join(series_path,f"channel_{channel}" )
                os.makedirs(channel_path, exist_ok=True)
                num_frames = reader.dims.Z
                for frame in range(0,num_frames):
                    yx = reader.get_image_data("YX", T=0, C=channel, Z=frame)
                    name_f = f"{date}_s{i}_ch{channel}_f{frame}.tif"
                    imwrite(os.path.join(channel_path, name_f), yx)
                    
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