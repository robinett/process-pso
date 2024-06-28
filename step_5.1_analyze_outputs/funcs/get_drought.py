import scipy.stats as ss
import sys
import pandas as pd
import numpy as np
import spei
import matplotlib.pyplot as plt
import os

class drought:
    def get_spei(self,precip,evap,plots_dir=np.nan,plot_spei=False):
        p_minus_e = precip - evap
        pixels = list(p_minus_e.columns)
        spei_df = pd.DataFrame(columns=pixels)
        spei_df['time'] = list(p_minus_e.index)
        for p,pix in enumerate(pixels):
            print('working on spei for pix {}'.format(pix))
            this_p_e = p_minus_e[pix].dropna()
            if len(this_p_e) > 0:
                spei_calc = spei.spei(this_p_e,dist=ss.fisk)
                spei_calc = spei_calc.rolling(90).mean()
                print(spei_calc)
                if plot_spei:
                    plt.figure(figsize=(8,4))
                    spei.plot.si(spei_calc)
                    #plt.xlim(tmin,tmax)
                    plt.grid()
                    plt.title('SPI for pixel {}'.format(pix))
                    savename = os.path.join(
                        plots_dir,
                        'spi_{}.png'.format(pix)
                    )
                    plt.savefig(savename)
                    plt.close()
                spei_df[pix] = list(spei_calc)
                print(spei_df)
        spei_df = spei_df.set_index('time')
        return spei_df
    def rank(self,to_rank):
        cols = list(to_rank.columns)
        years = list(to_rank.index)
        ranks = pd.DataFrame(columns=cols)
        ranks['time'] = years
        for c,col in enumerate(cols):
            this_obs = np.array(to_rank[col])
            this_ranks = ss.rankdata(this_obs)
            ranks[col] = this_ranks
        ranks = ranks.set_index('time')
        return ranks
