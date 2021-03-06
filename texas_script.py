import time
import pandas as pd
from FL_models import *
import constants

data = np.load("./texas100.npz")

imgs = data['features']
labels = data['labels']
imgs = torch.tensor(imgs, dtype=torch.float)
labels = torch.tensor(labels)
labels = torch.max(labels, dim=1).indices
rand_idx = torch.randperm(imgs.size(0))
imgs = imgs[rand_idx]
labels = labels[rand_idx]

train_imgs = imgs[:60000]
train_labels = labels[:60000]
test_imgs = imgs[60000:]
test_labels = labels[60000:]

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}, features {test_imgs.size(1)}")

print("Initializing...")
num_iter = 201
Ph = 50
hidden = 1024
malicious_factor = 0.3
for att_mode in constants.att_modes:
    for exp in constants.experiments:
        cgd = FL_torch(
            num_iter=num_iter,
            train_imgs=train_imgs,
            train_labels=train_labels,
            test_imgs=test_imgs,
            test_labels=test_labels,
            Ph=Ph,
            malicious_factor=malicious_factor,
            defender=exp['defender'],
            n_H=hidden,
            dataset="TEXAS",
            start_attack=exp['start'],
            attack_mode=att_mode,
            k_nearest=35,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(2000)
        print(f"Start {att_mode} attack to {exp['defender']}...")
        t1 = time.time()
        cgd.eq_train()
        t2 = time.time()
        print(f"{att_mode} attack to {exp['defender']} complete, time consumed {t2-t1}s")