#!/usr/bin/env python

from cnexp import redo


def main():
    crashed_runs = {
        # "ft-3": dict(
        #     path="../../experiments/cifar/dl/model:random_state=3118/"
        #     "sgd/lrcos/infonce/train:random_state=3118"
        #     "/ftmodel:freeze=1:change=lastlin:random_state=3118"
        #     "/sgd/lrcos:n_epochs=50:warmup_epochs=0/infonce"
        #     "/train/ftmodel:freeze=0/sgd:lr=0.00012/lrcos:n_epochs=450/train"
        #     "/default.run",
        #     time="24:00:00",
        # ),
        # "euc.2d-3": dict(
        #     path="../../experiments/cifar/dl/model:out_dim=2:random_state=3118"
        #     "/sgd/lrcos/infonce/train:random_state=3118/default.run",
        #     time="24:00:00",
        # ),
        # "euc-c100": dict(
        #     path="../../experiments/cifar100/dl/model:out_dim=2/sgd/lrcos/"
        #     "infonce/train/intermediates.zip",
        #     time="24:00:00",
        # ),
        "supervised": dict(
            path="../../experiments/cifar100/dl/model:out_dim=100/sgd/lrcos/"
            "ce_loss/train/intermediates.zip",
            time="24:00:00",
        ),
        # "budget500-2": dict(
        #     path="../../experiments/cifar/dl/model:random_state=704/"
        #     "sgd/lrcos:n_epochs=400/infonce/train:random_state=704/"
        #     "ftmodel:freeze=1:change=lastlin:random_state=704/"
        #     "sgd/lrcos:n_epochs=25:warmup_epochs=0/infonce/train/default.run",
        #     time="24:00:00",
        # ),
        # "ince-2-1-fixup": dict(
        #     path="../../experiments/cifar/dl/model/sgd/lrcos/"
        #     "closs:loss_mode=infonce:negative_samples=2/train/default.run",
        #     time="24:00:00",
        # ),
        # "nce-2-1-fixup": dict(
        #     path="../../experiments/cifar/dl/model/sgd/lrcos/"
        #     "closs:loss_mode=nce:negative_samples=2/train/default.run",
        #     time="24:00:00",
        # ),
        # "neg-128-3-fixup": dict(
        #     path="../../experiments/cifar/dl/model:random_state=3118/"
        #     "sgd/lrcos/closs:loss_mode=neg_sample:negative_samples=128/"
        #     "train:random_state=3118/default.run",
        #     time="24:00:00",
        # ),
    }

    redo.redo_ifchange_slurm(
        [v["path"] for v in crashed_runs.values()],
        name=[k for k in crashed_runs.keys()],
        time_str=[v["time"] for v in crashed_runs.values()],
        partition="gpu-2080ti",
    )


if __name__ == "__main__":
    main()
