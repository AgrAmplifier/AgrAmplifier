import numpy as np
from FL_models import *
import constants
import time

data = np.load("./location.npz")

imgs = data['arr_0']
labels = data['arr_1']
train_imgs = imgs[:4000]
train_labels = labels[:4000]
test_imgs = imgs[4000:]
test_labels = labels[4000:]

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
time_recorder = pd.DataFrame(index=["mislead", "min_max", "label_flip", "grad_ascent"],
                             columns=[x['defender'] for x in constants.experiments])
num_iter = 71
Ph = 50
hidden = 1024
malicious_factor = 0.3
# att_modes = ["mislead", "min_max", "label_flip", "grad_ascent"]
kernel = 3
for att_mode in ["mislead"]:
    for exp in [constants.attacked]:
        cgd = FL_torch(
            num_iter=num_iter,
            train_imgs=train_imgs,
            train_labels=train_labels,
            test_imgs=test_imgs,
            test_labels=test_labels,
            Ph=Ph,
            malicious_factor=malicious_factor,
            defender=exp['defender'],
            k_nearest=35,
            p_kernel=2,
            n_H=hidden,
            dataset="LOCATION",
            start_attack=exp['start'],
            attack_mode=att_mode,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution()
        print(f"Start {att_mode} attack to {exp['defender']}, kernel size {kernel}...")
        t1 = time.time()
        cgd.eq_train()
        t2 = time.time()
        print(f"{att_mode} attack to {exp['defender']} complete， time consumed {t2 - t1}s")
        # time_recorder.loc[att_mode][exp['defender']] = t2 - t1
# time_recorder.to_csv("./output/Location_timer.csv")
