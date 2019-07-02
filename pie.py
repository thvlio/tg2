# import necessary packages
import numpy as np
import os
from matplotlib import pyplot

# import files
import config

def main():
    ''' Main function. '''

    # pie charts
    # pies = [('rede5_media5',    ['Rede, 5%', 'Média, 5%'],              [1, 4],     [1, 4],     ['C0', 'C2']),
    #     ('rede5_media5_off',    ['Rede, 5%', 'Média, 5%', 'Nenhum'],    [0, 0, 1],  [1, 0, 0],  ['C0', 'C2', 'C7']),
    #     ('rede20_media20',      ['Rede, 20%', 'Média, 20%'],            [2, 2],     [0, 4],     ['C1', 'C3']),
    #     ('rede5_rede20',        ['Rede, 5%', 'Rede, 20%'],              [1, 1],     [0, 2],     ['C0', 'C1']),
    #     ('media5_media20',      ['Média, 5%', 'Média, 20%'],            [0, 1],     [0, 1],     ['C2', 'C3']),
    #     ('media5_media20_off',  ['Média, 5%', 'Média, 20%', 'Nenhum'],  [0, 1, 0],  [0, 0, 1],  ['C2', 'C3', 'C7'])]
    pies = [('rede5_media5',    ['R5', 'M5'],       [1, 4],     [1, 4],     ['C0', 'C2']),
        ('rede5_media5_off',    ['R5', 'M5', 'D'],  [0, 0, 1],  [1, 0, 0],  ['C0', 'C2', 'C7']),
        ('rede20_media20',      ['R20', 'M20'],     [2, 2],     [0, 4],     ['C1', 'C3']),
        ('rede5_rede20',        ['R5', 'R20'],      [1, 1],     [0, 2],     ['C0', 'C1']),
        ('media5_media20',      ['M5', 'M20'],      [0, 1],     [0, 1],     ['C2', 'C3']),
        ('media5_media20_off',  ['M5', 'M20', 'D'], [0, 1, 0],  [0, 0, 1],  ['C2', 'C3', 'C7'])]

    # plot the pie charts
    for fname, labels, sizesp, sizeso, colors in pies:

        # save the image plots
        fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, figsize=config.figsizeg, subplot_kw=dict(aspect="equal"))
        fig.suptitle(f'Preferências | {" vs. ".join(labels)}', x=0.5, y=0.95)
        wedges, texts = ax1.pie(sizesp, labels=labels, shadow=True, colors=colors,
            wedgeprops=dict(width=0.9), labeldistance=1.2, startangle=0)
        for size, text in zip(sizesp, texts):
            if size != 0:
                text.set_text(f'{text.get_text()}\n{size} pessoa{"" if size == 1 else "s"}')
                # text.set_text(f'{text.get_text()}')
                # text.set_text(f'{text.get_text()} – {size}')
                pyplot.setp(text, size=10, weight="bold")
            else:
                pyplot.setp(text, alpha=0.0)
        ax1.set_title('Prédio', x=0.5, y=1.05)
        wedges, texts = ax2.pie(sizeso, labels=labels, shadow=True, colors=colors,
            wedgeprops=dict(width=0.9), labeldistance=1.2, startangle=0)
        for size, text in zip(sizeso, texts):
            if size != 0:
                text.set_text(f'{text.get_text()}\n{size} pessoa{"" if size == 1 else "s"}')
                # text.set_text(f'{text.get_text()}')
                # text.set_text(f'{text.get_text()} – {size}')
                pyplot.setp(text, size=10, weight="bold")
            else:
                pyplot.setp(text, alpha=0.0)
        ax2.legend(wedges, labels,
            title="Método",
            loc="center",
            bbox_to_anchor=(1.0, 0, 0.5, 1))
        ax2.set_title('Oito', x=0.5, y=1.05)
        pyplot.subplots_adjust(wspace=0.4)
        pyplot.savefig(os.path.join('results', f'pie_gold_{fname}.png'), bbox_inches='tight')

# if it is the main file
if __name__ == '__main__':
    main()
