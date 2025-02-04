from typing import Optional, Dict, Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.components.stocks_dataframe import StocksDataFrame

# Configure logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StocksDataModule(LightningDataModule):
    """LightningDataModule for stock market data.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader)
        - test_dataloader (the test dataloader)
    """

    def __init__(
        self,
        index: str,
        scraping_date: list,
        train_date: list,
        val_date: list,
        test_date: list,
        interval: str,
        market: str,
        sequence_length: int,
        batch_size: int,
        cache_dir: str,
        use_cache: bool,
        provider: str,
        openbb_output_type: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        selected_tickers: list = None,
    ):
        super().__init__()

        # Save all params to hparams
        self.save_hyperparameters(logger=False)
        logger.info(
            f"Initializing StocksDataModule with index: {index}, market: {market}"
        )
        logger.debug(
            f"Training period: {train_date}, Validation period: {val_date}, Test period: {test_date}"
        )

        self.data_train: Optional[StocksDataFrame] = None
        self.data_val: Optional[StocksDataFrame] = None
        self.data_test: Optional[StocksDataFrame] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        logger.info("Preparing data for download...")
        try:
            StocksDataFrame(
                index=self.hparams.index,
                scraping_date=self.hparams.scraping_date,
                train_date=self.hparams.train_date,
                val_date=self.hparams.val_date,
                test_date=self.hparams.test_date,
                interval=self.hparams.interval,
                market=self.hparams.market,
                sequence_length=self.hparams.sequence_length,
                cache_dir=self.hparams.cache_dir,
                use_cache=self.hparams.use_cache,
                provider=self.hparams.provider,
                openbb_output_type=self.hparams.openbb_output_type,
                train=True,
                validation=False,
                selected_tickers=self.hparams.selected_tickers,
            )
            logger.info("Data preparation completed successfully")
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        logger.info(f"Setting up data for stage: {stage}")

        if stage == "fit" or stage is None:
            if self.data_train is None:
                logger.info("Loading training dataset...")
                try:
                    self.data_train = StocksDataFrame(
                        index=self.hparams.index,
                        scraping_date=self.hparams.scraping_date,
                        train_date=self.hparams.train_date,
                        val_date=self.hparams.val_date,
                        test_date=self.hparams.test_date,
                        interval=self.hparams.interval,
                        market=self.hparams.market,
                        sequence_length=self.hparams.sequence_length,
                        cache_dir=self.hparams.cache_dir,
                        use_cache=self.hparams.use_cache,
                        provider=self.hparams.provider,
                        openbb_output_type=self.hparams.openbb_output_type,
                        selected_tickers=self.hparams.selected_tickers,
                        train=True,
                        validation=False,
                    )
                    logger.info(
                        f"Training dataset loaded successfully. Size: {len(self.data_train)}"
                    )
                except Exception as e:
                    logger.error(f"Error loading training dataset: {str(e)}")
                    raise

            if self.data_val is None:
                logger.info("Loading validation dataset...")
                try:
                    self.data_val = StocksDataFrame(
                        index=self.hparams.index,
                        scraping_date=self.hparams.scraping_date,
                        train_date=self.hparams.train_date,
                        val_date=self.hparams.val_date,
                        test_date=self.hparams.test_date,
                        interval=self.hparams.interval,
                        market=self.hparams.market,
                        sequence_length=self.hparams.sequence_length,
                        cache_dir=self.hparams.cache_dir,
                        use_cache=self.hparams.use_cache,
                        provider=self.hparams.provider,
                        openbb_output_type=self.hparams.openbb_output_type,
                        selected_tickers=self.hparams.selected_tickers,
                        train=True,
                        validation=True,
                    )
                    logger.info(
                        f"Validation dataset loaded successfully. Size: {len(self.data_val)}"
                    )
                except Exception as e:
                    logger.error(f"Error loading validation dataset: {str(e)}")
                    raise

        if stage == "test" or stage is None:
            if self.data_test is None:
                logger.info("Loading test dataset...")
                try:
                    self.data_test = StocksDataFrame(
                        index=self.hparams.index,
                        scraping_date=self.hparams.scraping_date,
                        train_date=self.hparams.train_date,
                        val_date=self.hparams.val_date,
                        test_date=self.hparams.test_date,
                        interval=self.hparams.interval,
                        market=self.hparams.market,
                        sequence_length=self.hparams.sequence_length,
                        cache_dir=self.hparams.cache_dir,
                        use_cache=self.hparams.use_cache,
                        provider=self.hparams.provider,
                        openbb_output_type=self.hparams.openbb_output_type,
                        selected_tickers=self.hparams.selected_tickers,
                        train=False,
                        validation=False,
                    )
                    logger.info(
                        f"Test dataset loaded successfully. Size: {len(self.data_test)}"
                    )
                except Exception as e:
                    logger.error(f"Error loading test dataset: {str(e)}")
                    raise

    def train_dataloader(self):
        logger.debug("Creating training dataloader")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        logger.debug("Creating validation dataloader")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        logger.debug("Creating test dataloader")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
