from sciq_dataset import SciQDataset
from binary_operation_dataset import (SubtractionDataset, 
                                                                AdditionDataset,
                                                                MultiplicationDataset, 
                                                                ModularAdditionDataset)
from cities_dataset import PopulationDataset, CapitalsDataset, HemisphereDataset
from nli_dataset import NliDataset
from sentiment_dataset import SentimentDataset
from books_dataset import AuthorsDataset, BookRatingDataset
from unary_operation_dataset import SquaringDataset


ds_classes = [
    # (PopulationDataset, 4000),
    # (CapitalsDataset, 2000),
    # (HemisphereDataset, 4000),
    # (AuthorsDataset, 4000),
    # (BookRatingDataset, 4000),
    # (SentimentDataset, 8000),
    # (NliDataset, 4000),
    # (SciQDataset, 4000),
    (AdditionDataset, 8000),
    (SubtractionDataset, 8000),
    (MultiplicationDataset, 8000),
    (ModularAdditionDataset, 8000),
    (SquaringDataset, 8000),
]
if __name__ == "__main__":

    for ds_class, n_val_test in ds_classes:
        ds = ds_class(working_dir="weak_lm_datasets", verbose=True)
        print("Creating dataset", ds.name)
        ds.save_quirky_dataset(difficulty_model_names=[], push_to_hub=True, n_train=-1, n_val=n_val_test, n_test=n_val_test)
