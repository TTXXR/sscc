conf = {
    "save_path": "/home/gameai/my_nsm/nsm_data/trained",
    "load_path": "/home/gameai/my_nsm/nsm_data/trained",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUDA_USE": [0, 1],
    "data": "/home/gameai/my_nsm/nsm_data",
    "model": {
        'model_name': 'MLP_EA',
        'epoch': 150,
        'batch_size': 1200,
        'segmentation': [0, 419, 575, 2609, 4657, 5307],
        'encoder_dim': 5307,
        'encoder_num': 4,
        'mlp_ratio': 4.,
        'encoder_dropout': 0.3,
        'decoder_dim': [5307, 4096, 2048, 1024, 618],
        'decoder_dropout': 0.3,
        'lr': 0.0001,
        'layer_num': 4
    },
}
