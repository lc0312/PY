# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

paint_points = np.linspace (-1, 1, 25)

def art_work():
    a = np.random.uniform(1, 2, size=64)[:, np.newaxis]
    paintings = a*np.power (paint_points,2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable (paintings)

G = nn.Sequential(
    nn.Linear(5, 128),
    nn.ReLU(),
    nn.Linear(128, 25)
    )

D = nn.Sequential (
    nn.Linear (25,128),
    nn.ReLU(),
    nn.Linear (128, 1),
    nn.Sigmoid()
    )

opt_D = torch.optim.Adam (D.parameters(),lr=1E-4 )
opt_G = torch.optim.Adam (G.parameters(),lr=1E-4 )

for step in range (10000):
    art_paintings = art_work()
    G_ideas = torch.randn (64, 5, requires_grad=True )
    G_paintings = G(G_ideas)

    prob_art1 = D(G_paintings)

    G_loss = torch.mean ( torch.log(1.-prob_art1))
    opt_G.zero_grad ()
    G_loss.backward ()
    opt_G.step ()

    prob_art0 = D(art_paintings)
    prob_art1 = D(G_paintings.detach())
    D_loss = -torch.mean (torch.log(prob_art0)+torch.log(1.- prob_art1))
    opt_D.zero_grad ()
    D_loss.backward (retain_graph=1)
    opt_D.step ()
    

    if step % 1000 == 0:  # plotting
        plt.cla()
        plt.plot(paint_points, G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(paint_points, 2 * np.power(paint_points, 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(paint_points, 1 * np.power(paint_points, 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_art0.data.numpy().mean(),
                 fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=12);
        plt.draw();
        plt.pause(0.01)
        plt.show()

