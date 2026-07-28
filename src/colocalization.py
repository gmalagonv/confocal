from scipy.stats import fisher_exact, spearmanr
import pandas as pd
import numpy as np



def snr_proxy(stack, p95_thresh=None):
    """(p95 - median) / std of background (voxels <= median).

    Proxy for how far the fixed-percentile mask threshold sits above the
    background noise floor, in background-sigma units. Low values mean the
    '95th percentile' mask is not distinguishable from the tail of the
    background noise distribution.
    """
    stack = np.asarray(stack)
    median = np.median(stack)
    if p95_thresh is None:
        p95_thresh = np.percentile(stack, 95)
    bg = stack[stack <= median]
    std_bg = bg.std()
    return round((p95_thresh - median) / std_bg, 3)


# snr_rows = []
# for series in series_list:
#     name = list(info.keys())[series]
#     row = {'name': name}
#     for c_str, c_name in channels.items():
#         entry = results['s_' + str(series)]['c_' + c_str][deconv_str_mask]
#         row[f'snr_{c_name}'] = snr_proxy(entry['stack'], entry['95']['threshold'])
#     snr_rows.append(row)

# df_snr = pd.DataFrame(snr_rows)

# df = pd.read_csv(path_csv)
# df = df.merge(df_snr, on='name')

# for c in comparisons:
#     chA_name = channels[str(c[0])]
#     chB_name = channels[str(c[1])]
#     snr_col = f'snr_min_{chA_name}_{chB_name}'
#     df[snr_col] = df[[f'snr_{chA_name}', f'snr_{chB_name}']].min(axis=1)

#     for metric in ['MB_total_odds_ratio', 'MB_total_enrichment']:
#         r, p = spearmanr(df[snr_col], df[metric])
#         print(f'Spearman({snr_col}, {metric}) = {r:.3f} (p={p:.3f})')

# df.to_csv(path_csv, index=False)
# df.sort_values(list(df.filter(like='snr_min').columns)[0], ascending=False)



def compute_colocalization(results, series_list, names_list, raw=False, deconv_iter=2, export=True):
    #raw = False # <---------------------------------

    #series_list = list(range(len(info.keys())))
    
    table_results = {}
    table_results['name'] = []
    table_results['mask_HSP'] = []
    table_results['mask_mito'] = []    
    table_results['mask_BRP'] = []

    table_results['prop_MB_BRP'] = []
    table_results['prop_MB_HSP'] = []
    table_results['prop_MB_Mito'] = []

    table_results['prop_MB-HSP_in_syn'] = []
    table_results['prop_MB-Mito_in_syn'] = []


    for series in series_list:
        #print(list(info.keys())[series])
        
        
        
        channels = {
            '0':'BRP',
            '1':'HSP',
            '2':'Mito',    
        }

        comparisons = [
                    # [0,1],
                    # [0,2],
                    [1,2],
                    ]


        chA_thrsh_al = '95'# ch1,HSP
        chB_thrsh_al = '95'# ch2,Mito
        brp_thrsh_al = '95' # ch0,BRP

        table_results['name'].append(names_list[series])
        table_results['mask_HSP'].append(chA_thrsh_al)
        table_results['mask_mito'].append(chB_thrsh_al)
        table_results['mask_BRP'].append(brp_thrsh_al)
        #************************************************************
        deconv_str_mask = 'deconv_iter_'+ str(deconv_iter)
        
        MB_mask = results['s_'+str(series)]['c_2'][deconv_str_mask]['mb'].astype(bool)
        
        if raw:
            
            brp_mask = results['s_'+str(series)]['c_0']['raw'][brp_thrsh_al]['masks'].astype(bool)
            mito_mask = results['s_'+str(series)]['c_2']['raw'][chB_thrsh_al]['masks'].astype(bool)
            hsp_mask = results['s_'+str(series)]['c_1']['raw'][chA_thrsh_al]['masks'].astype(bool)
        else:

            brp_mask = results['s_'+str(series)]['c_0'][deconv_str_mask][brp_thrsh_al]['masks'].astype(bool)
            mito_mask = results['s_'+str(series)]['c_2'][deconv_str_mask][chB_thrsh_al]['masks'].astype(bool)
            hsp_mask = results['s_'+str(series)]['c_1'][deconv_str_mask][chA_thrsh_al]['masks'].astype(bool)

        SYN_mask = MB_mask & brp_mask
        NON_SYN_mask = MB_mask & ~brp_mask
        
        prop_MB_BRP = (brp_mask & MB_mask).sum()/MB_mask.sum()#((brp_mask & MB_mask).sum()/MB_mask.sum())
        prop_MB_HSP = (hsp_mask & MB_mask).sum()/MB_mask.sum()
        prop_MB_Mito = (mito_mask & MB_mask).sum()/MB_mask.sum()
        
        prop_MB_HSP_in_syn = (hsp_mask & SYN_mask).sum()/(hsp_mask & MB_mask).sum()
        prop_MB_Mito_in_syn = (mito_mask & SYN_mask).sum()/(mito_mask & MB_mask).sum()

        table_results['prop_MB_BRP'].append(float(f'{prop_MB_BRP:.3f}'))
        table_results['prop_MB_HSP'].append(float(f'{prop_MB_HSP:.3f}'))
        table_results['prop_MB_Mito'].append(float(f'{prop_MB_Mito:.3f}'))

        table_results['prop_MB-HSP_in_syn'].append(float(f'{prop_MB_HSP_in_syn:.3f}'))
        table_results['prop_MB-Mito_in_syn'].append(float(f'{prop_MB_Mito_in_syn:.3f}'))



        for c in comparisons:
            
            chA_n = c[0]
            chB_n = c[1]
            
            if raw:
                mask_chA = results['s_'+str(series)]['c_'+str(chA_n)]['raw'][chA_thrsh_al]['masks'].astype(bool)
                mask_chB = results['s_'+str(series)]['c_'+str(chB_n)]['raw'][chB_thrsh_al]['masks'].astype(bool)
            else:
                mask_chA = results['s_'+str(series)]['c_'+str(chA_n)][deconv_str_mask][chA_thrsh_al]['masks'].astype(bool)
                mask_chB = results['s_'+str(series)]['c_'+str(chB_n)][deconv_str_mask][chB_thrsh_al]['masks'].astype(bool)
                    
            
            
            masks2check = {'MB_total': MB_mask,
                        'MB_syn': SYN_mask,
                        'MB_non_syn' : NON_SYN_mask,
                        }
            
            for msk in masks2check.keys():
                #print(msk)
                MASK = masks2check[msk]
                
                
                mask_chA_MB = mask_chA & MASK
                mask_chB_MB = mask_chB & MASK
                overlap = mask_chA_MB & mask_chB_MB # |A∩B|
                
            ############
                A_and_B  = (mask_chA_MB &  mask_chB_MB).sum()
                A_not_B  = (mask_chA_MB & ~mask_chB_MB).sum()
                B_not_A  = (~mask_chA_MB & mask_chB_MB).sum()
                neither  = MASK.sum() - A_and_B - A_not_B - B_not_A  # fixed


                table = [[A_and_B, A_not_B],
                        [B_not_A, neither]]

                odds_ratio, pvalue = fisher_exact(table, alternative='greater') 

                
                
                ##############

                fraction_A_in_B = overlap.sum() / mask_chA_MB.sum()   # = |A∩B| / |A|:  "fraction of A that is B-positive"
                fraction_B_in_A = overlap.sum() / mask_chB_MB.sum()   # = |A∩B| / |B|:  "fraction of A that is B-positive"

                
                pA = mask_chA_MB.sum() / MASK.sum()
                pB = mask_chB_MB.sum() / MASK.sum()

                # Expected overlap:
                observed = overlap.sum()
                expected = pA * pB * MASK.sum()

                #print(f'observed={observed}, expected={expected:.2f}')
                #print (f'the observed enrichment is {(observed/expected):.2f} bigger than expected.')
            
                column1 = msk + '_' +f'fraction of {channels[str(chA_n)]} in {channels[str(chB_n)]}'
                column2 = msk + '_' +f'fraction of {channels[str(chB_n)]} in {channels[str(chA_n)]}'
                column3 = msk + '_' + 'odds_ratio'
                column4 = msk + '_' + 'enrichment'
                
                if column1 not in table_results.keys():
                    table_results[column1] = []
                if column2 not in table_results.keys():
                    table_results[column2] = []
                if column3 not in table_results.keys():
                    table_results[column3] = []
                if column4 not in table_results.keys(): 
                    table_results[column4] = []
                
                table_results[column1].append(float(f'{fraction_A_in_B:.3f}'))
                table_results[column2].append(float(f'{fraction_B_in_A:.3f}'))
                table_results[column3].append(float(f'{odds_ratio:.3f}'))
                table_results[column4].append(float(f'{(observed/expected):.3f}'))

    df = pd.DataFrame(table_results)

################# the following part needs to be integrated in the main loop: ineficient.
    snr_rows = []
    for series in series_list:
        name = names_list[series]
        row = {'name': name}
        for c_str, c_name in channels.items():
            if raw:
                entry = results['s_' + str(series)]['c_' + c_str]['raw']
            else:
                entry = results['s_' + str(series)]['c_' + c_str][deconv_str_mask]
            row[f'snr_{c_name}'] = snr_proxy(entry['stack'], entry['95']['threshold'])
        snr_rows.append(row)

    df_snr = pd.DataFrame(snr_rows)
    df = df.merge(df_snr, on='name')

    for c in comparisons:
        chA_name = channels[str(c[0])]
        chB_name = channels[str(c[1])]
        snr_col = f'snr_min_{chA_name}_{chB_name}'
        df[snr_col] = df[[f'snr_{chA_name}', f'snr_{chB_name}']].min(axis=1)

        for metric in ['MB_total_odds_ratio', 'MB_total_enrichment']:
            r, p = spearmanr(df[snr_col], df[metric])
            print(f'Spearman({snr_col}, {metric}) = {r:.3f} (p={p:.3f})')
    
    if export:
        if raw:
            path_csv = home + date + '_' + user + '/coloc_results_raw.csv'
        else:
            path_csv = home + date + '_' + user + '/coloc_results.csv'
        df.to_csv(path_csv, index=False)
    
    return df