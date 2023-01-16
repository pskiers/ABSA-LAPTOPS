from typing import Callable
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Class responsible for training the model
    """

    def __init__(
            self,
            model: nn.Module,
            optim: torch.optim.Optimizer,
            loss: Callable,
            train_dataloader: data.DataLoader,
            valid_dataloader: data.DataLoader,
            test_dataloader: data.DataLoader,
            metrics: list[Callable],
            tensorboard: bool = True,
            verbose: bool = True
        ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.optim = optim
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.verbose = verbose

        self.tb_writer = None
        if tensorboard is True:
            self.tb_writer = SummaryWriter(f"./runs/{self.model._get_name()}")
            _, rev_ids, rev_mask, asp_ids, asp_mask, _ = next(iter(self.train_dataloader))
            rev_ids = rev_ids.to(self.device)
            rev_mask = rev_mask.to(self.device)
            asp_ids = asp_ids.to(self.device)
            asp_mask = asp_mask.to(self.device)
            self.tb_writer.add_graph(self.model, (rev_ids, rev_mask, asp_ids, asp_mask))

        self.history: list[tuple[float, float, float, float]] = []


    def evaluate(self, eval_dataloader: data.DataLoader) -> float:
        """
        Evaluates the model on given dataset
        """
        losses = []
        metric_scores = [[] for _ in self.metrics]
        self.model.eval()
        with torch.no_grad():
            for _, rev_ids, rev_mask, asp_ids, asp_mask, labels in eval_dataloader:
                rev_ids = rev_ids.to(self.device)
                rev_mask = rev_mask.to(self.device)
                asp_ids = asp_ids.to(self.device)
                asp_mask = asp_mask.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(rev_ids, rev_mask, asp_ids, asp_mask).squeeze(dim=1)

                loss = self.loss(predictions, labels)
                losses.append(loss.item())
                for scores, metric in zip(metric_scores, self.metrics):
                    scores.append(metric(predictions, labels))
        return sum(losses) / len(losses), [sum(scores) / len(scores) for scores in metric_scores]


    def train_one_epoch(self) -> None:
        """
        Trains one epoch
        """
        losses = []
        metric_scores = [[] for _ in self.metrics]
        self.model.train()
        for _, rev_ids, rev_mask, asp_ids, asp_mask, labels in self.train_dataloader:
            rev_ids = rev_ids.to(self.device)
            rev_mask = rev_mask.to(self.device)
            asp_ids = asp_ids.to(self.device)
            asp_mask = asp_mask.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(rev_ids, rev_mask, asp_ids, asp_mask).squeeze(dim=1)

            loss = self.loss(predictions, labels)

            self.optim.zero_grad()

            loss.backward()

            self.optim.step()

            losses.append(loss.item())
            for scores, metric in zip(metric_scores, self.metrics):
                scores.append(metric(predictions, labels))

        self.history.append(
            (sum(losses)/len(losses), [sum(scores) / len(scores) for scores in metric_scores], *self.evaluate(self.valid_dataloader))
        )

    def train(self, epochs: int, early_stoping: int = 0) -> None:
        """
        Trains model
        """
        min_loss = None
        without_progress = 0
        try:
            for epoch in range(epochs):
                self.train_one_epoch()
                train_loss, train_metric, valid_loss, valid_metrics = self.history[-1]
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("training loss", train_loss, epoch)
                    self.tb_writer.add_scalar("validation loss", valid_loss, epoch)
                    for metric, score in zip(self.metrics, train_metric):
                        self.tb_writer.add_scalar(f"training {metric.__name__}", score, epoch)
                    for metric, score in zip(self.metrics, valid_metrics):
                        self.tb_writer.add_scalar(f"validation {metric.__name__}", score, epoch)
                    self.tb_writer.close()
                if self.verbose is True:
                    train_metric_string = ""
                    for metric, score in zip(self.metrics, train_metric):
                        train_metric_string += f"{metric}: {score:.4}\t "
                    valid_metric_string = ""
                    for metric, score in zip(self.metrics, valid_metrics):
                        valid_metric_string += f"{metric}: {score:.4}\t "

                    print(
                        f"Epoch: {epoch}\t Train loss: {train_loss:.4}\t metrics: {train_metric_string}\n Validation loss: {valid_loss:.4}\t metrics: {valid_metric_string}"
                    )
                if early_stoping > 0:
                    _, _, valid_loss, _ = self.history[-1]
                    if min_loss is None:
                        min_loss = valid_loss
                    elif min_loss > valid_loss:
                        min_loss = valid_loss
                        without_progress = 0
                    else:
                        without_progress += 1

                    if without_progress >= early_stoping:
                        break
        except KeyboardInterrupt:
            pass

        if self.verbose is True:
            test_loss, test_metrics = self.evaluate(self.test_dataloader)
            metric_string = ""
            for metric, score in zip(self.metrics, test_metrics):
                metric_string += f"{metric}: {score:.4}\t"
            print(f"Test loss: {test_loss:.4}\t metrics: {metric_string}")
