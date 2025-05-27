We want our data to be separated from model training for better organization, reusability and modularity - we can edit and write processing scripts, change data sources columns etc without having to change the model code.

We have two data primitives:
1. torch.utils.data.Dataset - an abstract class representing a dataset. It is used to define how to access and process the data. It pre-loads datasets and organizes them into samples and labels. This is map style access, meaning that it allows you to access the data by index. It is used to define how to load and process the data, and it can be used to create custom datasets by subclassing it. Required methods are:
   - `__len__`: returns the number of samples in the dataset.
   - `__getitem__`: Loads and returns a sample (tensor) from the dataset at a given index. Based on the index, it identifies the sample, locaates it, reads it, gets a corresponding label, applied any transformations and returns the sample and label as a tuple. This method is called when you access the dataset by index, e.g., `dataset[i]`.

   Example usage:
   ```python
   class MyDataset(torch.utils.data.Dataset):
       def __init__(self, data, labels, transform=None, target_transform=None):

           self.data = data
           self.labels = labels
           self.transform = transform
           self.target_transform = target_transform

       def __len__(self):
           return len(self.data)

       def __getitem__(self, idx):
           sample = self.data[idx]
           label = self.labels[idx]
           if self.transform:
               sample = self.transform(sample)
           if self.target_transform:
               label = self.target_transform(label)
           return sample, label
   ```

It also has a Iterable style access for streaming data which is useful for large datasets that cannot fit into memory. In this case, the `__iter__` method is implemented to yield samples one by one.
Example usage:
```python
from torch.utils.data import IterableDataset
class MyIterableDataset(IterableDataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        for item in self.data_source:
            yield item
```
1. torch.utils.data.DataLoader - a class that provides an iterable over the dataset. It is used to load the data in samples, collate them into batches, shuffle at every epoch, and apply transformations. It is a wrapper (a simple api that returns and iterbale and abstrcts away the complexity of using multiprocessing (workers) for this.)  It takes a dataset as input and returns an iterable that yields batches of data. It can also be used to parallelize data loading by using multiple workers. Required parameters are:
   - `dataset`: the dataset to load.
   - `batch_size`: the number of samples in each batch.
   - `shuffle`: whether to shuffle the data.
   - `num_workers`: the number of subprocesses to use for data loading.
   - `pin_memory`: whether to load the data into memory.
  
  Example usage:
  ```python
    from torch.utils.data import DataLoader
    dataset = MyDataset(data, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    for batch in dataloader:
        inputs, labels = batch
        # process the batch
    ```

    Transforms:
   - `transform`: a function/transform to apply to each sample.
   - `target_transform`: a function/transform to apply to each target.

    - `collate_fn`: a function to merge a list of samples into a batch. It is used to customize how the samples are combined into a batch. By default, it uses the default collate function which stacks the samples into a tensor.

    Sampler:
    - `sampler`: a sampler that defines the strategy to draw samples from the dataset. It is used to customize how the samples are drawn from the dataset. It can be used to implement custom sampling strategies, such as weighted sampling or stratified sampling.


## Background Concepts:
1. Lazy Loading: Data is not loaded into memory until __getitem__ is called.
2. Dataset objects are pickled when using multiprocessing, so they should not contain any non-picklable objects.