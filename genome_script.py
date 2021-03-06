import numpy as np
from FL_models import *
import constants

data = np.load("./gnome.npz")

train_imgs = data['features']
train_labels = data['labels']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
rand_idx = torch.randperm(train_labels.size(0))
train_imgs = train_imgs[rand_idx]
train_labels = train_labels[rand_idx]

test_imgs = train_imgs[1000:]
test_labels = train_labels[1000:]
train_imgs = train_imgs[:1000]
train_labels = train_labels[:1000]

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
num_iter = 201
Ph = 50
hidden = 128
malicious_factor = 0.3
for att_mode in ["mislead", "min_max", "label_flip", "grad_ascent"]:
# for att_mode in ["scale"]:
    for exp in [constants.fang, constants.p_fang]:
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
            k_nearest=35,
            dataset="GENOME",
            start_attack=exp['start'],
            attack_mode=att_mode,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(validation_size=80)
        print(f"Start {att_mode} attack to {exp['defender']}...")
        cgd.eq_train()
        print(f"{att_mode} attack to {exp['defender']} complete")