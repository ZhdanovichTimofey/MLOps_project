import os

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from metrics import iou
from tqdm.notebook import tqdm


def train(
    segment_model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    state_dict_path,
    device="cpu",
    n_epochs=20,
    show_interval=20,
    savefig_dir=None,
):
    # Set history
    history = {}
    history["train_loss"] = []
    history["train_iou"] = []
    history["val_loss"] = []
    history["val_iou"] = []

    # Set start epoch number
    start_epoch = 0

    # Load state dict
    if os.path.exists(state_dict_path):
        state = torch.load(state_dict_path)
        segment_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"] + 1

    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)

    best_val_iou = 0.0

    end_epoch = start_epoch + n_epochs

    if savefig_dir is not None:
        if not os.path.exists(savefig_dir):
            os.mkdir(savefig_dir)

    for epoch in range(start_epoch, start_epoch + n_epochs):

        print(f"Epoch {epoch}/{end_epoch}")

        segment_model.train()

        train_loss = 0.0
        train_iou = 0.0

        for i, (image, mask) in enumerate(tqdm(train_dataloader)):
            image = image.to(device)
            mask = mask.to(device)

            pred = segment_model(image)
            loss = criterion(pred, mask.float())
            loss.backward()

            # будем аккумулировать градиенты так, чтобы делать 1 шаг за эпоху, поскольку у нас всего 41 изображение на эпоху
            if i == (n_train_batches - 1):
                optimizer.step()
                optimizer.zero_grad()

            loss_ = float(loss.detach().data)
            # предсказываем маску по вероятности 0.5 <=> logit > 0
            iou_ = float(iou(pred.detach() > 0.0, mask > 0).data)

            train_loss += loss_
            train_iou += iou_

        train_loss = train_loss / n_train_batches
        train_iou = train_iou / n_train_batches

        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)

        print("")
        print(f"Total Train:\tloss\t{train_loss:.5f}" f"\t\tIoU\t{train_iou:.5f}")

        segment_model.eval()

        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for image, mask in tqdm(val_dataloader):
                image, mask = image.to(device), mask.to(device)
                pred = segment_model(image)
                loss = criterion(pred, mask.float())
                loss_ = float(loss.data)
                # предсказываем маску по вероятности 0.5 <=> logit > 0
                iou_ = float(iou(pred > 0.0, mask > 0).data)

                val_iou += iou_
                val_loss += loss_

        val_loss = val_loss / n_val_batches
        val_iou = val_iou / n_val_batches

        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if history["val_iou"][-1] > best_val_iou:
            torch.save(
                {
                    "model": segment_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "epoch": epoch,
                },
                state_dict_path,
            )
            best_val_iou = history["val_iou"][-1]

        print("")
        print(
            f"Total Valid:\tloss\t{val_loss:.5f}"
            f"\t\tIoU\t{val_iou:.5f}"
            f"\t\tbest IoU\t{best_val_iou:.5f}"
        )
        print("-" * 100)

        clear_output(wait=True)

        if epoch % show_interval == show_interval - 1:
            # Отрисовка графиков
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="train")
            plt.plot(history["val_loss"], label="test")
            plt.title("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history["train_iou"], label="train")
            plt.plot(history["val_iou"], label="test")
            plt.title("IoU")
            plt.legend()

            plt.show()
