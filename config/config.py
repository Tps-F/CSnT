from omegaconf import OmegaConf


class Config:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        config = OmegaConf.load("config/config.yaml")
        self.nmda_dataset = config.nmda_dataset
        self.training = config.training
        self.learning_schdule = config.learning_schedule
        self.num_epochs = config.training.num_epochs
        self.model = config.model
        self.device = "cuda"

        # define schedule
        config.schedules.loss_weights_per_epoch = [
            [x[0], x[1], x[2] * x[3]] for x in config.schedules.loss_weights_per_epoch
        ]

        self.schedule = [
            [
                config.schedules["epochs_change"][i],
                config.schedules["learning_rate_per_epoch"][i],
                config.schedules["loss_weights_per_epoch"][i],
            ]
            for i in range(len(config.schedules["learning_rate_per_epoch"]))
        ]

        self.batch_size_per_epoch = [8] * self.num_epochs

        self.learning_rate_per_epoch = [
            next(
                lr for start_epoch, lr, _ in reversed(self.schedule) if i >= start_epoch
            )
            for i in range(self.num_epochs)
        ]

        self.loss_weights_per_epoch = [
            next(
                loss_weights
                for start_epoch, _, loss_weights in reversed(self.schedule)
                if i >= start_epoch
            )
            for i in range(self.num_epochs)
        ]

        self.learning_schedule = {
            "batch_size_per_epoch": self.batch_size_per_epoch,
            "loss_weights_per_epoch": self.loss_weights_per_epoch,
            "learning_rate_per_epoch": self.learning_rate_per_epoch,
            "num_train_steps_per_epoch": [100] * self.num_epochs,
            **config.learning_schedule,
        }


if __name__ == "__main__":
    config = Config()
