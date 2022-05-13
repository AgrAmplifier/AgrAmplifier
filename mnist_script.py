import numpy as np
from FL_models import *
import constants

data = np.load("./mnist_train.npz")
train_imgs = data['arr_0']
train_labels = data['arr_1']
data = np.load("./mnist_test.npz")
test_imgs = data['arr_0']
test_labels = data['arr_1']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
num_iter = 201
Ph = 50
hidden = 512
malicious_factor = 0.4
for att_mode in constants.targeted_att:
    for exp in [constants.p_cos, constants.np_cos, constants.p_trust]:
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
            dataset="MNIST",
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
        cgd.eq_train()
        print(f"{att_mode} attack to {exp['defender']} complete")