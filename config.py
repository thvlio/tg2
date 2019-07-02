# dataset
dataset_name = 'oito'

# how many times to train each network
num_iter = 5

# parameters
network_archs = [ # (epochs, batch_size, [layer_units])
    (200, 32, [32, 16]),
    (200, 64, [32, 16]),
    (200, 32, [64, 32]),
    (200, 64, [64, 32]),
    (200, 32, [96, 64, 32]),
    (200, 64, [96, 64, 32]),
    (200, 32, [128, 96, 64, 32]),
    (200, 64, [128, 96, 64, 32])
]

# figure sizes for saving
figsizej = (11, 7)
figsizep = (7, 7)
figsizeg = (7, 4)
figsizel = (11, 6.5)
figsizes = (18, 7)

