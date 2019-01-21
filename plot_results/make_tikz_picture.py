
import numpy as np


def get_tikz_picture(avgs, vars, names):
    print('\\begin{tikzpicture}[scale=\subfigscale ]')

    print('\\begin{axis}[mlineplot, ymin=\\subfigminy,')
    print('            ymax=\\subfigmaxy,')
    print('            xlabel=Train Size,')
    print('            ylabel=\\subfigylabel,')
    print('            xmin=\\subfigminx,')
    print('            xmax=\\subfigmaxx ]')

    for i in range(len(avgs)):
        print('\\addplot+[error bars/y dir=both,')
        print('             error bars/y explicit, mark=none')
        print('             ,color=\\' + names[i] + 'plotcolor, mark=\\' + names[i] + 'plotmark'
              + ', dash pattern=on \\'+ names[i] + 'plotdashpatternfirst off \\'+ names[i] + 'plotdashpatternsecond on \\'
              + names[i] + 'plotdashpatternthird off \\'+ names[i] + 'plotdashpatternfourth'
              + ', line width=\\' + names[i] + 'plotlinewidth ]')
        print('        coordinates{')
        for j in range(len(avgs[i])):
            print('       (' + str(100*(j+1)) + ', ' + str(avgs[i][j]) + ') +- (' + str(np.sqrt(vars[i][j])) + ', ' + str(np.sqrt(vars[i][j])) + ')')
        print('        };')
        print('\\addlegendentry{\\' + names[i] + 'plotlegendlabel }')
    print('\\end{axis}')
    print('\\end{tikzpicture}')

'''

avgs = [
    [ 1.04369325332, 0.79443773417, 0.671778510216, 0.559706872905, 0.516823683205, 0.504941127883, 0.461454314255, 0.451712578794, 0.441844414845, 0.432442327962]
    , [ 0.910168285764, 0.75938120699, 0.664449716453, 0.565996099696, 0.533465874839, 0.508340729681, 0.472479226492, 0.463783764184, 0.453364448642, 0.439730405756]
    , [ 0.913259494476, 0.705221910418, 0.653165745093, 0.576958173319, 0.539543959248, 0.528895694147, 0.494594485596, 0.486454305305, 0.48040532479, 0.468446658297]
         ]


vars = [
    [ 0.000594554980792, 0.000244453478604, 0.000159904353519, 1.41677976593e-05, 2.40206578442e-05, 6.8947757864e-05, 6.47776374964e-06, 6.51833987025e-06, 1.46416654379e-05, 1.03534348938e-05]
    , [ 0.00043490641153, 0.000299219409022, 0.000179076968281, 1.82628918334e-05, 4.78829092482e-05, 3.70456192772e-05, 1.01023755592e-05, 2.81234650559e-05, 1.60936317039e-05, 8.64596899545e-06]
    , [ 0.000444310716269, 7.9063824104e-05, 0.000255498160319, 0.000145035155644, 2.3126195316e-05, 4.46284628777e-05, 9.84117835347e-06, 3.23544855781e-06, 4.0083911853e-06, 9.96073887953e-07]
]

labels = ['MADE', 'MAGSDE', 'MASDE']

colors = ['blue', 'red', 'black']

marks = []

get_tikz_picture(ymin=0.1, ymax=1.1, ylabel='KL Divergence',avgs=avgs, vars=vars,labels=labels, colors=colors, marks=marks)


'''
